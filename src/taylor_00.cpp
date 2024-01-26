// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Small helper to deduce the number of parameters
// present in the Taylor decomposition of an ODE system.
// NOTE: this will also include the functions of state variables,
// as they are part of the decomposition.
// NOTE: the first few entries in the decomposition are the mapping
// u variables -> state variables. These never contain any param
// by construction.
std::uint32_t n_pars_in_dc(const taylor_dc_t &dc)
{
    std::uint32_t retval = 0;

    for (const auto &p : dc) {
        retval = std::max(retval, get_param_size(p.first));
    }

    return retval;
}

// Add to s an adaptive timestepper function with support for events. This timestepper will *not*
// propagate the state of the system. Instead, its output will be the jet of derivatives
// of all state variables and event equations, and the deduced timestep value(s).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
taylor_dc_t taylor_add_adaptive_step_with_events(llvm_state &s, llvm::Type *ext_fp_t, llvm::Type *fp_t,
                                                 const std::string &name,
                                                 const std::vector<std::pair<expression, expression>> &sys,
                                                 std::uint32_t batch_size, bool compact_mode,
                                                 const std::vector<expression> &evs, bool high_accuracy,
                                                 bool parallel_mode, std::uint32_t order)
{
    assert(!s.is_compiled());
    assert(batch_size != 0u);

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto [dc, ev_dc] = taylor_decompose(sys, evs);

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();
    auto &context = s.context();
    auto &md = s.module();

    // Prepare the function prototype. The arguments are:
    // - pointer to the output jet of derivative (write only),
    // - pointer to the current state vector (read only),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the max_abs_state output variable (write only).
    // These pointers cannot overlap.
    const std::vector<llvm::Type *> fargs(6, llvm::PointerType::getUnqual(ext_fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);
    // NOTE: a step function cannot call itself recursively.
    f->addFnAttr(llvm::Attribute::NoRecurse);

    // Set the names/attributes of the function arguments.
    auto *jet_ptr = f->args().begin();
    jet_ptr->setName("jet_ptr");
    jet_ptr->addAttr(llvm::Attribute::NoCapture);
    jet_ptr->addAttr(llvm::Attribute::NoAlias);
    jet_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *state_ptr = jet_ptr + 1;
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::NoAlias);
    state_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *par_ptr = state_ptr + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = par_ptr + 1;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = time_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *max_abs_state_ptr = h_ptr + 1;
    max_abs_state_ptr->setName("max_abs_state_ptr");
    max_abs_state_ptr->addAttr(llvm::Attribute::NoCapture);
    max_abs_state_ptr->addAttr(llvm::Attribute::NoAlias);
    max_abs_state_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Create a global read-only array containing the values in ev_dc, if there
    // are any and we are in compact mode (otherwise, svf_ptr will be null).
    auto *svf_ptr = compact_mode ? taylor_c_make_sv_funcs_arr(s, ev_dc) : nullptr;

    // Compute the jet of derivatives at the given order.
    auto diff_variant = taylor_compute_jet(s, fp_t, state_ptr, par_ptr, time_ptr, dc, ev_dc, n_eq, n_uvars, order,
                                           batch_size, compact_mode, high_accuracy, parallel_mode);

    // Determine the integration timestep.
    auto *h = taylor_determine_h(s, fp_t, diff_variant, ev_dc, svf_ptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                 max_abs_state_ptr);

    // Store h to memory.
    ext_store_vector_to_memory(s, h_ptr, h);

    // Copy the jet of derivatives to jet_ptr.
    taylor_write_tc(s, fp_t, diff_variant, ev_dc, svf_ptr, jet_ptr, n_eq, n_uvars, order, batch_size);

    // End the lifetime of the array of derivatives, if we are in compact mode.
    if (compact_mode) {
        builder.CreateLifetimeEnd(std::get<0>(diff_variant).first,
                                  builder.getInt64(get_size(md, std::get<0>(diff_variant).second)));
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    return dc;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
taylor_dc_t taylor_add_adaptive_step(llvm_state &s, llvm::Type *ext_fp_t, llvm::Type *fp_t, const std::string &name,
                                     const std::vector<std::pair<expression, expression>> &sys,
                                     std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                                     bool parallel_mode, std::uint32_t order)
{
    assert(!s.is_compiled());
    assert(batch_size > 0u);

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    // NOTE: no sv_funcs needed for this stepper.
    auto [dc, sv_funcs_dc] = taylor_decompose(sys, {});

    assert(sv_funcs_dc.empty());

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();
    auto &context = s.context();
    auto &md = s.module();

    // Prepare the function prototype. The arguments are:
    // - pointer to the current state vector (read & write),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the Taylor coefficients output (write only).
    // These pointers cannot overlap.
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);
    const std::vector<llvm::Type *> fargs(5, llvm::PointerType::getUnqual(ext_fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &md);
    // NOTE: a step function cannot call itself recursively.
    f->addFnAttr(llvm::Attribute::NoRecurse);

    // Set the names/attributes of the function arguments.
    auto *state_ptr = f->args().begin();
    state_ptr->setName("state_ptr");
    state_ptr->addAttr(llvm::Attribute::NoCapture);
    state_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *par_ptr = state_ptr + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = par_ptr + 1;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = time_ptr + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tc_ptr = h_ptr + 1;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Compute the jet of derivatives at the given order.
    auto diff_variant = taylor_compute_jet(s, fp_t, state_ptr, par_ptr, time_ptr, dc, {}, n_eq, n_uvars, order,
                                           batch_size, compact_mode, high_accuracy, parallel_mode);

    // Determine the integration timestep.
    auto *h = taylor_determine_h(s, fp_t, diff_variant, sv_funcs_dc, nullptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                 nullptr);

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    auto new_state_var
        = high_accuracy ? taylor_run_ceval(s, fp_t, diff_variant, h, n_eq, n_uvars, order, high_accuracy, batch_size)
                        : taylor_run_multihorner(s, fp_t, diff_variant, h, n_eq, n_uvars, order, batch_size);

    // Store the new state.
    // NOTE: no need to perform overflow check on n_eq * batch_size,
    // as in taylor_compute_jet() we already checked.
    if (compact_mode) {
        auto *new_state = std::get<llvm::Value *>(new_state_var);

        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            auto *val = builder.CreateLoad(fp_vec_t, builder.CreateInBoundsGEP(fp_vec_t, new_state, cur_var_idx));
            ext_store_vector_to_memory(
                s,
                builder.CreateInBoundsGEP(ext_fp_t, state_ptr,
                                          builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))),
                val);
        });
    } else {
        const auto &new_state = std::get<std::vector<llvm::Value *>>(new_state_var);

        assert(new_state.size() == n_eq);

        for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
            ext_store_vector_to_memory(
                s, builder.CreateInBoundsGEP(ext_fp_t, state_ptr, builder.getInt32(var_idx * batch_size)),
                new_state[var_idx]);
        }
    }

    // Store the timesteps that were used.
    ext_store_vector_to_memory(s, h_ptr, h);

    // Write the Taylor coefficients, if requested.
    auto *nptr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ext_fp_t));
    llvm_if_then_else(
        s, builder.CreateICmpNE(tc_ptr, nptr),
        [&]() {
            // tc_ptr is not null: copy the Taylor coefficients
            // for the state variables.
            taylor_write_tc(s, fp_t, diff_variant, {}, nullptr, tc_ptr, n_eq, n_uvars, order, batch_size);
        },
        []() {
            // Taylor coefficients were not requested,
            // don't do anything in this branch.
        });

    // End the lifetime of the array of derivatives, if we are in compact mode.
    if (compact_mode) {
        builder.CreateLifetimeEnd(std::get<0>(diff_variant).first,
                                  builder.getInt64(get_size(md, std::get<0>(diff_variant).second)));
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    return dc;
}

namespace
{

// Helper to compute max(x_v, abs(y_v)) in the Taylor stepper implementation.
llvm::Value *taylor_step_maxabs(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
    return llvm_max(s, x_v, llvm_abs(s, y_v));
}

// Helper to compute min(x_v, abs(y_v)) in the Taylor stepper implementation.
llvm::Value *taylor_step_minabs(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
    return llvm_min(s, x_v, llvm_abs(s, y_v));
}

// Helper to compute the scaling + safety factor
// in taylor_determine_h().
number taylor_determine_h_rhofac(llvm_state &s, llvm::Type *fp_t, std::uint32_t order)
{
    assert(fp_t != nullptr);
    assert(order > 0u);

    const auto const_m7_10 = number_like(s, fp_t, -7.) / number_like(s, fp_t, 10.);
    const auto const_e2 = exp(number_like(s, fp_t, 1.)) * exp(number_like(s, fp_t, 1.));
    const auto const_om1 = number_like(s, fp_t, static_cast<double>(order - 1u));

    return exp(const_m7_10 / const_om1) / const_e2;
}

} // namespace

// Helper to generate the LLVM code to determine the timestep in an adaptive Taylor integrator,
// following Jorba's prescription. diff_variant is the output of taylor_compute_jet(), and it contains
// the jet of derivatives for the state variables and the sv_funcs. h_ptr is an external pointer containing
// the clamping values for the timesteps. svf_ptr is a pointer to the first element of an LLVM array containing the
// values in sv_funcs_dc. If max_abs_state_ptr is not nullptr, the computed norm infinity of the
// state vector (including sv_funcs, if any) will be written into it (max_abs_state_ptr is an external pointer).
llvm::Value *
taylor_determine_h(llvm_state &s, llvm::Type *fp_t,
                   const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &diff_variant,
                   const std::vector<std::uint32_t> &sv_funcs_dc,
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   llvm::Value *svf_ptr, llvm::Value *h_ptr,
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                   llvm::Value *max_abs_state_ptr)
{
    assert(batch_size != 0u);
#if !defined(NDEBUG)
    if (diff_variant.index() == 0u) {
        // Compact mode.
        assert(sv_funcs_dc.empty() == !svf_ptr);
    } else {
        // Non-compact mode.
        assert(svf_ptr == nullptr);
    }
#endif

    using std::exp;

    auto &builder = s.builder();

    llvm::Value *max_abs_state = nullptr, *max_abs_diff_o = nullptr, *max_abs_diff_om1 = nullptr;

    // Fetch the vector type.
    auto *vec_t = make_vector_type(fp_t, batch_size);

    if (diff_variant.index() == 0u) {
        // Compact mode.
        auto *diff_arr = std::get<0>(diff_variant).first;

        // These will end up containing the norm infinity of the state vector + sv_funcs and the
        // norm infinity of the derivatives at orders order and order - 1.
        max_abs_state = builder.CreateAlloca(vec_t);
        max_abs_diff_o = builder.CreateAlloca(vec_t);
        max_abs_diff_om1 = builder.CreateAlloca(vec_t);

        // Initialise with the abs(derivatives) of the first state variable at orders 0, 'order' and 'order - 1'.
        builder.CreateStore(
            llvm_abs(s, taylor_c_load_diff(s, vec_t, diff_arr, n_uvars, builder.getInt32(0), builder.getInt32(0))),
            max_abs_state);
        builder.CreateStore(
            llvm_abs(s, taylor_c_load_diff(s, vec_t, diff_arr, n_uvars, builder.getInt32(order), builder.getInt32(0))),
            max_abs_diff_o);
        builder.CreateStore(llvm_abs(s, taylor_c_load_diff(s, vec_t, diff_arr, n_uvars, builder.getInt32(order - 1u),
                                                           builder.getInt32(0))),
                            max_abs_diff_om1);

        // Iterate over the variables to compute the norm infinities.
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n_eq), [&](llvm::Value *cur_idx) {
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_state),
                                   taylor_c_load_diff(s, vec_t, diff_arr, n_uvars, builder.getInt32(0), cur_idx)),
                max_abs_state);
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_o),
                                   taylor_c_load_diff(s, vec_t, diff_arr, n_uvars, builder.getInt32(order), cur_idx)),
                max_abs_diff_o);
            builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_om1),
                                                   taylor_c_load_diff(s, vec_t, diff_arr, n_uvars,
                                                                      builder.getInt32(order - 1u), cur_idx)),
                                max_abs_diff_om1);
        });

        if (svf_ptr != nullptr) {
            // Consider also the functions of state variables for
            // the computation of the timestep.
            llvm_loop_u32(
                s, builder.getInt32(0), builder.getInt32(boost::numeric_cast<std::uint32_t>(sv_funcs_dc.size())),
                [&](llvm::Value *arr_idx) {
                    // Fetch the index value from the array.
                    auto *cur_idx = builder.CreateLoad(
                        builder.getInt32Ty(), builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                    builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_state),
                                                           taylor_c_load_diff(s, vec_t, diff_arr, n_uvars,
                                                                              builder.getInt32(0), cur_idx)),
                                        max_abs_state);
                    builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_o),
                                                           taylor_c_load_diff(s, vec_t, diff_arr, n_uvars,
                                                                              builder.getInt32(order), cur_idx)),
                                        max_abs_diff_o);
                    builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_om1),
                                                           taylor_c_load_diff(s, vec_t, diff_arr, n_uvars,
                                                                              builder.getInt32(order - 1u), cur_idx)),
                                        max_abs_diff_om1);
                });
        }

        // Load the values for later use.
        max_abs_state = builder.CreateLoad(vec_t, max_abs_state);
        max_abs_diff_o = builder.CreateLoad(vec_t, max_abs_diff_o);
        max_abs_diff_om1 = builder.CreateLoad(vec_t, max_abs_diff_om1);
    } else {
        // Non-compact mode.
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_variant);

        const auto n_sv_funcs = static_cast<std::uint32_t>(sv_funcs_dc.size());

        // Compute the norm infinity of the state vector and the norm infinity of the derivatives
        // at orders order and order - 1. We first create vectors of absolute values and then
        // compute their maxima.
        std::vector<llvm::Value *> v_max_abs_state, v_max_abs_diff_o, v_max_abs_diff_om1;

        // NOTE: iterate up to n_eq + n_sv_funcs in order to
        // consider also the functions of state variables for
        // the computation of the timestep.
        for (std::uint32_t i = 0; i < n_eq + n_sv_funcs; ++i) {
            v_max_abs_state.push_back(llvm_abs(s, diff_arr[i]));
            // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
            // state variables and sv funcs (not all u vars), hence the indexing is
            // order * (n_eq + n_sv_funcs).
            v_max_abs_diff_o.push_back(llvm_abs(s, diff_arr[order * (n_eq + n_sv_funcs) + i]));
            v_max_abs_diff_om1.push_back(llvm_abs(s, diff_arr[(order - 1u) * (n_eq + n_sv_funcs) + i]));
        }

        // Find the maxima via pairwise reduction.
        auto reducer = [&s](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return llvm_max(s, a, b); };
        max_abs_state = pairwise_reduce(v_max_abs_state, reducer);
        max_abs_diff_o = pairwise_reduce(v_max_abs_diff_o, reducer);
        max_abs_diff_om1 = pairwise_reduce(v_max_abs_diff_om1, reducer);
    }

    // Store max_abs_state, if requested.
    if (max_abs_state_ptr != nullptr) {
        ext_store_vector_to_memory(s, max_abs_state_ptr, max_abs_state);
    }

    // Determine if we are in absolute or relative tolerance mode.
    auto *abs_or_rel = llvm_fcmp_ole(s, max_abs_state, llvm_constantfp(s, vec_t, 1.));

    // Estimate rho at orders order - 1 and order.
    auto *num_rho = builder.CreateSelect(abs_or_rel, llvm_constantfp(s, vec_t, 1.), max_abs_state);

    // NOTE: it is fine here to static_cast<double>(order), as order is a 32-bit integer
    // and double is a IEEE double-precision type (i.e., 53 bits).
    auto *rho_o = llvm_pow(
        s, llvm_fdiv(s, num_rho, max_abs_diff_o),
        vector_splat(builder,
                     llvm_codegen(s, fp_t, number_like(s, fp_t, 1.) / number_like(s, fp_t, static_cast<double>(order))),
                     batch_size));
    auto *rho_om1 = llvm_pow(
        s, llvm_fdiv(s, num_rho, max_abs_diff_om1),
        vector_splat(
            builder,
            llvm_codegen(s, fp_t, number_like(s, fp_t, 1.) / number_like(s, fp_t, static_cast<double>(order - 1u))),
            batch_size));

    // Take the minimum.
    auto *rho_m = llvm_min(s, rho_o, rho_om1);

    // Compute the scaling + safety factor.
    const auto rhofac = taylor_determine_h_rhofac(s, fp_t, order);

    // Determine the step size in absolute value.
    auto *h = llvm_fmul(s, rho_m, vector_splat(builder, llvm_codegen(s, fp_t, rhofac), batch_size));

    // Ensure that the step size does not exceed the limit in absolute value.
    auto *max_h_vec = ext_load_vector_from_memory(s, fp_t, h_ptr, batch_size);
    h = taylor_step_minabs(s, h, max_h_vec);

    // Handle backwards propagation.
    auto *backward = llvm_fcmp_olt(s, max_h_vec, llvm_constantfp(s, vec_t, 0.));
    auto *h_fac = builder.CreateSelect(backward, llvm_constantfp(s, vec_t, -1.), llvm_constantfp(s, vec_t, 1.));
    h = llvm_fmul(s, h_fac, h);

    return h;
}

} // namespace detail

HEYOKA_END_NAMESPACE
