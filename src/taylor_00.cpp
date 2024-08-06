// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

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

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

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

// Run the Horner scheme to propagate an ODE state via the evaluation of the Taylor polynomials.
// diff_var contains either the derivatives for all u variables (in compact mode) or only
// for the state variables (non-compact mode). The evaluation point (i.e., the timestep)
// is h. The evaluation is run in parallel over the polynomials of all the state
// variables.
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_multihorner(llvm_state &s, llvm::Type *fp_t,
                       const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &diff_var,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       llvm::Value *h, std::uint32_t n_eq, std::uint32_t n_uvars,
                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                       std::uint32_t order, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (diff_var.index() == 0u) {
        // Compact mode.
        auto *diff_arr = std::get<0>(diff_var).first;

        // Create the array storing the results of the evaluation.
        auto *fp_vec_t = make_vector_type(fp_t, batch_size);
        auto *array_type = llvm::ArrayType::get(fp_vec_t, n_eq);
        auto *array_inst = builder.CreateAlloca(array_type);
        auto *res_arr = builder.CreateInBoundsGEP(array_type, array_inst, {builder.getInt32(0), builder.getInt32(0)});

        // Init the return value, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr and store it in res_arr.
            builder.CreateStore(
                taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, builder.getInt32(order), cur_var_idx),
                builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx));
        });

        // Run the evaluation.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                      [&](llvm::Value *cur_order) {
                          llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                              // Load the current poly coeff from diff_arr.
                              // NOTE: we are loading the coefficients backwards wrt the order, hence
                              // we specify order - cur_order.
                              auto *cf = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars,
                                                            builder.CreateSub(builder.getInt32(order), cur_order),
                                                            cur_var_idx);

                              // Accumulate in res_arr.
                              auto *res_ptr = builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx);
                              builder.CreateStore(
                                  llvm_fadd(s, cf, llvm_fmul(s, builder.CreateLoad(fp_vec_t, res_ptr), h)), res_ptr);
                          });
                      });

        return res_arr;
    } else {
        // Non-compact mode.
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_var);

        // Init the return value, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        std::vector<llvm::Value *> res_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            res_arr.push_back(diff_arr[(n_eq * order) + i]);
        }

        // Run the Horner scheme simultaneously for all polynomials.
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                res_arr[j] = llvm_fadd(s, diff_arr[(order - i) * n_eq + j], llvm_fmul(s, res_arr[j], h));
            }
        }

        return res_arr;
    }
}

// Same as taylor_run_multihorner(), but instead of the Horner scheme this implementation uses
// a compensated summation over the naive evaluation of monomials.
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_ceval(llvm_state &s, llvm::Type *fp_t,
                 const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &diff_var,
                 llvm::Value *h,
                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                 std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, bool, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    if (diff_var.index() == 0u) {
        // Compact mode.
        auto *diff_arr = std::get<0>(diff_var).first;

        // Create the arrays storing the results of the evaluation and the running compensations.
        auto *fp_vec_t = make_vector_type(fp_t, batch_size);
        auto *array_type = llvm::ArrayType::get(fp_vec_t, n_eq);
        auto *res_arr_inst = builder.CreateAlloca(array_type);
        auto *comp_arr_inst = builder.CreateAlloca(array_type);
        auto *res_arr = builder.CreateInBoundsGEP(array_type, res_arr_inst, {builder.getInt32(0), builder.getInt32(0)});
        auto *comp_arr
            = builder.CreateInBoundsGEP(array_type, comp_arr_inst, {builder.getInt32(0), builder.getInt32(0)});

        // Init res_arr with the order-0 coefficients, and the running
        // compensations with zero.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr.
            auto *val = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, builder.getInt32(0), cur_var_idx);

            // Store it in res_arr.
            builder.CreateStore(val, builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx));

            // Zero-init the element in comp_arr.
            builder.CreateStore(llvm_constantfp(s, fp_vec_t, 0.),
                                builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx));
        });

        // Init the running updater for the powers of h.
        auto *cur_h = builder.CreateAlloca(h->getType());
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(order + 1u), [&](llvm::Value *cur_order) {
            // Load the current power of h.
            auto *cur_h_val = builder.CreateLoad(fp_vec_t, cur_h);

            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                // Evaluate the current monomial.
                auto *cf = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, cur_order, cur_var_idx);
                auto *tmp = llvm_fmul(s, cf, cur_h_val);

                // Compute the quantities for the compensation.
                auto *comp_ptr = builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx);
                auto *res_ptr = builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx);
                auto *y = llvm_fsub(s, tmp, builder.CreateLoad(fp_vec_t, comp_ptr));
                auto *cur_res = builder.CreateLoad(fp_vec_t, res_ptr);
                auto *t = llvm_fadd(s, cur_res, y);

                // Update the compensation and the return value.
                builder.CreateStore(llvm_fsub(s, llvm_fsub(s, t, cur_res), y), comp_ptr);
                builder.CreateStore(t, res_ptr);
            });

            // Update the value of h.
            builder.CreateStore(llvm_fmul(s, cur_h_val, h), cur_h);
        });

        return res_arr;
    } else {
        // Non-compact mode.
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_var);

        // Init the return values with the order-0 monomials, and the running
        // compensations with zero.
        std::vector<llvm::Value *> res_arr, comp_arr;
        for (std::uint32_t i = 0; i < n_eq; ++i) {
            res_arr.push_back(diff_arr[i]);
            comp_arr.push_back(llvm_constantfp(s, diff_arr[i]->getType(), 0.));
        }

        // Evaluate and sum.
        auto *cur_h = h;
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                // Evaluate the current monomial.
                auto *tmp = llvm_fmul(s, diff_arr[i * n_eq + j], cur_h);

                // Compute the quantities for the compensation.
                auto *y = llvm_fsub(s, tmp, comp_arr[j]);
                auto *t = llvm_fadd(s, res_arr[j], y);

                // Update the compensation and the return value.
                comp_arr[j] = llvm_fsub(s, llvm_fsub(s, t, res_arr[j]), y);
                res_arr[j] = t;
            }

            // Update the power of h.
            cur_h = llvm_fmul(s, cur_h, h);
        }

        return res_arr;
    }
}

// Helper to generate the LLVM code to store the Taylor coefficients of the state variables and
// the sv funcs into an external array. The Taylor polynomials are stored in row-major order,
// first the state variables and after the sv funcs. For use in the adaptive timestepper implementations.
// tc_ptr is an external pointer.
void taylor_write_tc(
    llvm_state &s, llvm::Type *fp_t,
    const std::variant<std::pair<llvm::Value *, llvm::Type *>, std::vector<llvm::Value *>> &diff_variant,
    const std::vector<std::uint32_t> &sv_funcs_dc, llvm::Value *svf_ptr, llvm::Value *tc_ptr,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size)
{
    // LCOV_EXCL_START
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
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the vector type.
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Convert to std::uint32_t for overflow checking and use below.
    const auto n_sv_funcs = boost::numeric_cast<std::uint32_t>(sv_funcs_dc.size());

    // Overflow checking: ensure we can index into
    // tc_ptr using std::uint32_t.
    // LCOV_EXCL_START
    if (order == std::numeric_limits<std::uint32_t>::max()
        || (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || n_eq > std::numeric_limits<std::uint32_t>::max() - n_sv_funcs
        || n_eq + n_sv_funcs > std::numeric_limits<std::uint32_t>::max() / ((order + 1u) * batch_size)) {
        throw std::overflow_error("An overflow condition was detected while generating the code for writing the Taylor "
                                  "polynomials of an ODE system into the output array");
    }
    // LCOV_EXCL_STOP

    if (diff_variant.index() == 0u) {
        // Compact mode.

        auto *diff_arr = std::get<0>(diff_variant).first;

        // Write out the Taylor coefficients for the state variables.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var) {
            llvm_loop_u32(
                s, builder.getInt32(0), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                [&](llvm::Value *cur_order) {
                    // Load the value of the derivative from diff_arr.
                    auto *diff_val = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, cur_order, cur_var);

                    // Compute the index in the output pointer.
                    auto *out_idx
                        = builder.CreateAdd(builder.CreateMul(builder.getInt32((order + 1u) * batch_size), cur_var),
                                            builder.CreateMul(cur_order, builder.getInt32(batch_size)));

                    // Store into tc_ptr.
                    ext_store_vector_to_memory(s, builder.CreateInBoundsGEP(ext_fp_t, tc_ptr, out_idx), diff_val);
                });
        });

        // Write out the Taylor coefficients for the sv funcs, if necessary.
        if (svf_ptr != nullptr) {
            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_sv_funcs), [&](llvm::Value *arr_idx) {
                // Fetch the u var index from svf_ptr.
                auto *cur_idx = builder.CreateLoad(builder.getInt32Ty(),
                                                   builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                llvm_loop_u32(
                    s, builder.getInt32(0), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                    [&](llvm::Value *cur_order) {
                        // Load the derivative value from diff_arr.
                        auto *diff_val = taylor_c_load_diff(s, fp_vec_t, diff_arr, n_uvars, cur_order, cur_idx);

                        // Compute the index in the output pointer.
                        auto *out_idx
                            = builder.CreateAdd(builder.CreateMul(builder.getInt32((order + 1u) * batch_size),
                                                                  builder.CreateAdd(builder.getInt32(n_eq), arr_idx)),
                                                builder.CreateMul(cur_order, builder.getInt32(batch_size)));

                        // Store into tc_ptr.
                        ext_store_vector_to_memory(s, builder.CreateInBoundsGEP(ext_fp_t, tc_ptr, out_idx), diff_val);
                    });
            });
        }
    } else {
        // Non-compact mode.

        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_variant);

        for (std::uint32_t j = 0; j < n_eq + n_sv_funcs; ++j) {
            for (decltype(diff_arr.size()) cur_order = 0; cur_order <= order; ++cur_order) {
                // Index in the jet of derivatives.
                // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
                // state variables and sv_variable (not all u vars), hence the indexing
                // is cur_order * (n_eq + n_sv_funcs) + j.
                const auto arr_idx = cur_order * (n_eq + n_sv_funcs) + j;
                assert(arr_idx < diff_arr.size()); // LCOV_EXCL_LINE
                auto *const val = diff_arr[arr_idx];

                // Index in tc_ptr.
                const auto out_idx
                    = static_cast<decltype(diff_arr.size())>(order + 1u) * batch_size * j + cur_order * batch_size;

                // Write to tc_ptr.
                auto *out_ptr = builder.CreateInBoundsGEP(ext_fp_t, tc_ptr,
                                                          builder.getInt32(static_cast<std::uint32_t>(out_idx)));
                ext_store_vector_to_memory(s, out_ptr, val);
            }
        }
    }
}

} // namespace

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
    auto [dc, ev_dc] = taylor_decompose_sys(sys, evs);

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
    auto [dc, sv_funcs_dc] = taylor_decompose_sys(sys, {});

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

    return dc;
}

} // namespace detail

HEYOKA_END_NAMESPACE
