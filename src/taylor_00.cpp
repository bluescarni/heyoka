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

} // namespace detail

HEYOKA_END_NAMESPACE
