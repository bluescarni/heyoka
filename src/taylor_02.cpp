// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka::detail
{

// Function to split the central part of the decomposition (i.e., the definitions of the u variables
// that do not represent state variables) into parallelisable segments. Within a segment,
// the definition of a u variable does not depend on any u variable defined within that segment.
// NOTE: the hidden deps are not considered as dependencies.
// NOTE: the segments in the return value will contain shallow copies of the
// expressions in dc.
std::vector<taylor_dc_t> taylor_segment_dc(const taylor_dc_t &dc, std::uint32_t n_eq)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Helper that takes in input the definition ex of a u variable, and returns
    // in output the list of indices of the u variables on which ex depends.
    auto udef_args_indices = [](const expression &ex) -> std::vector<std::uint32_t> {
        return std::visit(
            [](const auto &v) -> std::vector<std::uint32_t> {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        std::visit(
                            [&retval](const auto &x) {
                                using tp = detail::uncvref_t<decltype(x)>;

                                if constexpr (std::is_same_v<tp, variable>) {
                                    retval.push_back(uname_to_index(x.name()));
                                } else if constexpr (!std::is_same_v<tp, number> && !std::is_same_v<tp, param>) {
                                    throw std::invalid_argument(
                                        "Invalid argument encountered in an element of a Taylor decomposition: the "
                                        "argument is not a variable or a number/param");
                                }
                            },
                            arg.value());
                    }

                    return retval;
                } else {
                    throw std::invalid_argument("Invalid expression encountered in a Taylor decomposition: the "
                                                "expression is not a function");
                }
            },
            ex.value());
    };

    // Init the return value.
    std::vector<taylor_dc_t> s_dc;

    // cur_limit_idx is initially the index of the first
    // u variable which is not a state variable.
    auto cur_limit_idx = n_eq;
    for (std::uint32_t i = n_eq; i < dc.size() - n_eq; ++i) {
        // NOTE: at the very first iteration of this for loop,
        // no block has been created yet. Do it now.
        if (i == n_eq) {
            assert(s_dc.empty());
            s_dc.emplace_back();
        } else {
            assert(!s_dc.empty());
        }

        const auto &[ex, deps] = dc[i];

        // Determine the u indices on which ex depends.
        const auto u_indices = udef_args_indices(ex);

        if (std::any_of(u_indices.begin(), u_indices.end(),
                        [cur_limit_idx](auto idx) { return idx >= cur_limit_idx; })) {
            // The current expression depends on one or more variables
            // within the current block. Start a new block and
            // update cur_limit_idx with the start index of the new block.
            s_dc.emplace_back();
            cur_limit_idx = i;
        }

        // Append ex to the current block.
        s_dc.back().emplace_back(ex, deps);
    }

#if !defined(NDEBUG)
    // Verify s_dc.

    decltype(dc.size()) counter = 0;
    for (const auto &s : s_dc) {
        // No segment can be empty.
        assert(!s.empty());

        for (const auto &[ex, _] : s) {
            // All the indices in the definitions of the
            // u variables in the current block must be
            // less than counter + n_eq (which is the starting
            // index of the block).
            const auto u_indices = udef_args_indices(ex);
            assert(std::all_of(u_indices.begin(), u_indices.end(),
                               [idx_limit = counter + n_eq](auto idx) { return idx < idx_limit; }));
        }

        // Update the counter.
        counter += s.size();
    }

    assert(counter == dc.size() - n_eq * 2u);
#endif

    get_logger()->debug("Taylor N of segments: {}", s_dc.size());
    get_logger()->trace("Taylor segment runtime: {}", sw);

    return s_dc;
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

// Helper to compute pow(x_v, y_v) in the Taylor stepper implementation.
llvm::Value *taylor_step_pow(llvm_state &s, llvm::Value *x_v, llvm::Value *y_v)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Determine the scalar type of the vector arguments.
    auto *x_t = x_v->getType()->getScalarType();

    if (x_t == llvm::Type::getFP128Ty(s.context())) {
        return call_extern_vec(s, x_v, y_v, "powq");
    } else {
#endif
        // If we are operating on SIMD vectors, try to see if we have a sleef
        // function available for pow().
        if (auto *vec_t = llvm::dyn_cast<llvm_vector_type>(x_v->getType())) {
            // NOTE: if sfn ends up empty, we will be falling through
            // below and use the LLVM intrinsic instead.
            if (const auto sfn = sleef_function_name(s.context(), "pow", vec_t->getElementType(),
                                                     boost::numeric_cast<std::uint32_t>(vec_t->getNumElements()));
                !sfn.empty()) {
                return llvm_invoke_external(
                    s, sfn, vec_t, {x_v, y_v},
                    // NOTE: in theory we may add ReadNone here as well,
                    // but for some reason, at least up to LLVM 10,
                    // this causes strange codegen issues. Revisit
                    // in the future.
                    {llvm::Attribute::NoUnwind, llvm::Attribute::Speculatable, llvm::Attribute::WillReturn});
            }
        }

        return llvm_invoke_intrinsic(s, "llvm.pow", {x_v->getType()}, {x_v, y_v});
#if defined(HEYOKA_HAVE_REAL128)
    }
#endif
}

} // namespace

// Helper to generate the LLVM code to determine the timestep in an adaptive Taylor integrator,
// following Jorba's prescription. diff_variant is the output of taylor_compute_jet(), and it contains
// the jet of derivatives for the state variables and the sv_funcs. h_ptr is a pointer containing
// the clamping values for the timesteps. svf_ptr is a pointer to the first element of an LLVM array containing the
// values in sv_funcs_dc. If max_abs_state_ptr is not nullptr, the computed norm infinity of the
// state vector (including sv_funcs, if any) will be written into it.
template <typename T>
llvm::Value *taylor_determine_h(llvm_state &s,
                                const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_variant,
                                const std::vector<std::uint32_t> &sv_funcs_dc, llvm::Value *svf_ptr, llvm::Value *h_ptr,
                                std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order,
                                std::uint32_t batch_size, llvm::Value *max_abs_state_ptr)
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
    auto &context = s.context();

    llvm::Value *max_abs_state = nullptr, *max_abs_diff_o = nullptr, *max_abs_diff_om1 = nullptr;

    if (diff_variant.index() == 0u) {
        // Compact mode.
        auto *diff_arr = std::get<llvm::Value *>(diff_variant);

        // These will end up containing the norm infinity of the state vector + sv_funcs and the
        // norm infinity of the derivatives at orders order and order - 1.
        auto vec_t = to_llvm_vector_type<T>(context, batch_size);
        max_abs_state = builder.CreateAlloca(vec_t);
        max_abs_diff_o = builder.CreateAlloca(vec_t);
        max_abs_diff_om1 = builder.CreateAlloca(vec_t);

        // Initialise with the abs(derivatives) of the first state variable at orders 0, 'order' and 'order - 1'.
        builder.CreateStore(
            llvm_abs(s, taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), builder.getInt32(0))),
            max_abs_state);
        builder.CreateStore(
            llvm_abs(s, taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order), builder.getInt32(0))),
            max_abs_diff_o);
        builder.CreateStore(
            llvm_abs(s, taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order - 1u), builder.getInt32(0))),
            max_abs_diff_om1);

        // Iterate over the variables to compute the norm infinities.
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n_eq), [&](llvm::Value *cur_idx) {
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_state),
                                   taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), cur_idx)),
                max_abs_state);
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_o),
                                   taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order), cur_idx)),
                max_abs_diff_o);
            builder.CreateStore(
                taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_om1),
                                   taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order - 1u), cur_idx)),
                max_abs_diff_om1);
        });

        if (svf_ptr != nullptr) {
            // Consider also the functions of state variables for
            // the computation of the timestep.
            llvm_loop_u32(
                s, builder.getInt32(0), builder.getInt32(boost::numeric_cast<std::uint32_t>(sv_funcs_dc.size())),
                [&](llvm::Value *arr_idx) {
                    // Fetch the index value from the array.
                    assert(llvm_depr_GEP_type_check(svf_ptr, builder.getInt32Ty())); // LCOV_EXCL_LINE
                    auto cur_idx = builder.CreateLoad(
                        builder.getInt32Ty(), builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                    builder.CreateStore(
                        taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_state),
                                           taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), cur_idx)),
                        max_abs_state);
                    builder.CreateStore(
                        taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_o),
                                           taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order), cur_idx)),
                        max_abs_diff_o);
                    builder.CreateStore(taylor_step_maxabs(s, builder.CreateLoad(vec_t, max_abs_diff_om1),
                                                           taylor_c_load_diff(s, diff_arr, n_uvars,
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
        store_vector_to_memory(builder, max_abs_state_ptr, max_abs_state);
    }

    // Determine if we are in absolute or relative tolerance mode.
    auto abs_or_rel
        = builder.CreateFCmpOLE(max_abs_state, vector_splat(builder, codegen<T>(s, number{1.}), batch_size));

    // Estimate rho at orders order - 1 and order.
    auto num_rho
        = builder.CreateSelect(abs_or_rel, vector_splat(builder, codegen<T>(s, number{1.}), batch_size), max_abs_state);
    auto rho_o = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_o),
                                 vector_splat(builder, codegen<T>(s, number{T(1) / order}), batch_size));
    auto rho_om1 = taylor_step_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_om1),
                                   vector_splat(builder, codegen<T>(s, number{T(1) / (order - 1u)}), batch_size));

    // Take the minimum.
    auto rho_m = llvm_min(s, rho_o, rho_om1);

    // Compute the scaling + safety factor.
    const auto rhofac = exp((T(-7) / T(10)) / (order - 1u)) / (exp(T(1)) * exp(T(1)));

    // Determine the step size in absolute value.
    auto h = builder.CreateFMul(rho_m, vector_splat(builder, codegen<T>(s, number{rhofac}), batch_size));

    // Ensure that the step size does not exceed the limit in absolute value.
    auto *max_h_vec = load_vector_from_memory(builder, h_ptr, batch_size);
    h = taylor_step_minabs(s, h, max_h_vec);

    // Handle backwards propagation.
    auto backward = builder.CreateFCmpOLT(max_h_vec, vector_splat(builder, codegen<T>(s, number{0.}), batch_size));
    auto h_fac = builder.CreateSelect(backward, vector_splat(builder, codegen<T>(s, number{-1.}), batch_size),
                                      vector_splat(builder, codegen<T>(s, number{1.}), batch_size));
    h = builder.CreateFMul(h_fac, h);

    return h;
}

// Instantiate the required versions of taylor_determine_h().
template llvm::Value *taylor_determine_h<double>(llvm_state &,
                                                 const std::variant<llvm::Value *, std::vector<llvm::Value *>> &,
                                                 const std::vector<std::uint32_t> &, llvm::Value *, llvm::Value *,
                                                 std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                                                 llvm::Value *);

template llvm::Value *taylor_determine_h<long double>(llvm_state &,
                                                      const std::variant<llvm::Value *, std::vector<llvm::Value *>> &,
                                                      const std::vector<std::uint32_t> &, llvm::Value *, llvm::Value *,
                                                      std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                                                      llvm::Value *);

#if defined(HEYOKA_HAVE_REAL128)

template llvm::Value *taylor_determine_h<mppp::real128>(llvm_state &,
                                                        const std::variant<llvm::Value *, std::vector<llvm::Value *>> &,
                                                        const std::vector<std::uint32_t> &, llvm::Value *,
                                                        llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                                        std::uint32_t, llvm::Value *);

#endif

// Create a global read-only array containing the values in sv_funcs_dc, if any
// (otherwise, the return value will be null). This is for use in the adaptive steppers
// when employing compact mode.
llvm::Value *taylor_c_make_sv_funcs_arr(llvm_state &s, const std::vector<std::uint32_t> &sv_funcs_dc)
{
    auto &builder = s.builder();

    if (sv_funcs_dc.empty()) {
        return nullptr;
    } else {
        auto *arr_type
            = llvm::ArrayType::get(builder.getInt32Ty(), boost::numeric_cast<std::uint64_t>(sv_funcs_dc.size()));
        std::vector<llvm::Constant *> sv_funcs_dc_const;
        sv_funcs_dc_const.reserve(sv_funcs_dc.size());
        for (auto idx : sv_funcs_dc) {
            sv_funcs_dc_const.emplace_back(builder.getInt32(idx));
        }
        auto *sv_funcs_dc_arr = llvm::ConstantArray::get(arr_type, sv_funcs_dc_const);
        auto *g_sv_funcs_dc = new llvm::GlobalVariable(s.module(), sv_funcs_dc_arr->getType(), true,
                                                       llvm::GlobalVariable::InternalLinkage, sv_funcs_dc_arr);

        // Get out a pointer to the beginning of the array.
        assert(llvm_depr_GEP_type_check(g_sv_funcs_dc, arr_type)); // LCOV_EXCL_LINE
        return builder.CreateInBoundsGEP(arr_type, g_sv_funcs_dc, {builder.getInt32(0), builder.getInt32(0)});
    }
}

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number/param), n_uvars the number of variables in
// the decomposition, arr the array containing the derivatives of all u variables
// up to order - 1.
template <typename T>
llvm::Value *taylor_compute_sv_diff(llvm_state &s, const expression &ex, const std::vector<llvm::Value *> &arr,
                                    llvm::Value *par_ptr, std::uint32_t n_uvars, std::uint32_t order,
                                    std::uint32_t batch_size)
{
    assert(order > 0u);

    auto &builder = s.builder();

    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                // Extract the index of the u variable in the expression
                // of the first-order derivative.
                const auto u_idx = uname_to_index(v.name());

                // Fetch from arr the derivative
                // of order 'order - 1' of the u variable at u_idx. The index is:
                // (order - 1) * n_uvars + u_idx.
                auto ret = taylor_fetch_diff(arr, u_idx, order - 1u, n_uvars);

                // We have to divide the derivative by order
                // to get the normalised derivative of the state variable.
                return builder.CreateFDiv(
                    ret, vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size));
            } else if constexpr (std::is_same_v<type, number> || std::is_same_v<type, param>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0. No need for normalization as the only
                // nonzero value that can be produced here is the first-order
                // derivative.
                if (order == 1u) {
                    return taylor_codegen_numparam<T>(s, v, par_ptr, batch_size);
                } else {
                    return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
                }
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

// Instantiate the required versions of taylor_compute_sv_diff().
template llvm::Value *taylor_compute_sv_diff<double>(llvm_state &, const expression &,
                                                     const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                                     std::uint32_t, std::uint32_t);

template llvm::Value *taylor_compute_sv_diff<long double>(llvm_state &, const expression &,
                                                          const std::vector<llvm::Value *> &, llvm::Value *,
                                                          std::uint32_t, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

template llvm::Value *taylor_compute_sv_diff<mppp::real128>(llvm_state &, const expression &,
                                                            const std::vector<llvm::Value *> &, llvm::Value *,
                                                            std::uint32_t, std::uint32_t, std::uint32_t);

#endif

// Helper to construct the global arrays needed for the computation of the
// derivatives of the state variables in compact mode. The first part of the
// return value is a set of 6 arrays:
// - the indices of the state variables whose time derivative is a u variable, paired to
// - the indices of the u variables appearing in the derivatives, and
// - the indices of the state variables whose time derivative is a constant, paired to
// - the values of said constants, and
// - the indices of the state variables whose time derivative is a param, paired to
// - the indices of the params.
// The second part of the return value is a boolean flag that will be true if
// the time derivatives of all state variables are u variables, false otherwise.
template <typename T>
std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
taylor_c_make_sv_diff_globals(llvm_state &s, const taylor_dc_t &dc, std::uint32_t n_uvars)
{
    auto &context = s.context();
    auto &builder = s.builder();
    auto &md = s.module();

    // Build iteratively the output values as vectors of constants.
    std::vector<llvm::Constant *> var_indices, vars, num_indices, nums, par_indices, pars;

    // Keep track of how many time derivatives
    // of the state variables are u variables.
    std::uint32_t n_der_vars = 0;

    // NOTE: the derivatives of the state variables are at the end of the decomposition.
    for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    ++n_der_vars;
                    // NOTE: remove from i the n_uvars offset to get the
                    // true index of the state variable.
                    var_indices.push_back(builder.getInt32(i - n_uvars));
                    vars.push_back(builder.getInt32(uname_to_index(v.name())));
                } else if constexpr (std::is_same_v<type, number>) {
                    num_indices.push_back(builder.getInt32(i - n_uvars));
                    nums.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, v)));
                } else if constexpr (std::is_same_v<type, param>) {
                    par_indices.push_back(builder.getInt32(i - n_uvars));
                    pars.push_back(builder.getInt32(v.idx()));
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            dc[i].first.value());
    }

    // Flag to signal that the time derivatives of all state variables are u variables.
    assert(dc.size() >= n_uvars); // LCOV_EXCL_LINE
    const auto all_der_vars = (n_der_vars == (dc.size() - n_uvars));

    assert(var_indices.size() == vars.size()); // LCOV_EXCL_LINE
    assert(num_indices.size() == nums.size()); // LCOV_EXCL_LINE
    assert(par_indices.size() == pars.size()); // LCOV_EXCL_LINE

    // Turn the vectors into global read-only LLVM arrays.

    // Variables.
    auto *var_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(var_indices.size()));

    auto *var_indices_arr = llvm::ConstantArray::get(var_arr_type, var_indices);
    auto *g_var_indices = new llvm::GlobalVariable(md, var_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, var_indices_arr);

    auto *vars_arr = llvm::ConstantArray::get(var_arr_type, vars);
    auto *g_vars
        = new llvm::GlobalVariable(md, vars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, vars_arr);

    // Numbers.
    auto *num_indices_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(num_indices.size()));
    auto *num_indices_arr = llvm::ConstantArray::get(num_indices_arr_type, num_indices);
    auto *g_num_indices = new llvm::GlobalVariable(md, num_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, num_indices_arr);

    auto nums_arr_type
        = llvm::ArrayType::get(to_llvm_type<T>(context), boost::numeric_cast<std::uint64_t>(nums.size()));
    auto nums_arr = llvm::ConstantArray::get(nums_arr_type, nums);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *g_nums
        = new llvm::GlobalVariable(md, nums_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, nums_arr);

    // Params.
    auto *par_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(par_indices.size()));

    auto *par_indices_arr = llvm::ConstantArray::get(par_arr_type, par_indices);
    auto *g_par_indices = new llvm::GlobalVariable(md, par_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, par_indices_arr);

    auto *pars_arr = llvm::ConstantArray::get(par_arr_type, pars);
    auto *g_pars
        = new llvm::GlobalVariable(md, pars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, pars_arr);

    return std::pair{std::array{g_var_indices, g_vars, g_num_indices, g_nums, g_par_indices, g_pars}, all_der_vars};
}

// Instantiate the required versions of taylor_c_make_sv_diff_globals().
template std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
taylor_c_make_sv_diff_globals<double>(llvm_state &, const taylor_dc_t &, std::uint32_t);

template std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
taylor_c_make_sv_diff_globals<long double>(llvm_state &, const taylor_dc_t &, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

template std::pair<std::array<llvm::GlobalVariable *, 6>, bool>
taylor_c_make_sv_diff_globals<mppp::real128>(llvm_state &, const taylor_dc_t &, std::uint32_t);

#endif

namespace
{

// Small helper to compute the size of a global array.
std::uint32_t taylor_c_gl_arr_size(llvm::Value *v)
{
    assert(llvm::isa<llvm::GlobalVariable>(v)); // LCOV_EXCL_LINE

    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::PointerType>(v->getType())->getElementType())->getNumElements());
}

} // namespace

// Helper to compute and store the derivatives of the state variables in compact mode at order 'order'.
// svd_gl is the return value of taylor_c_make_sv_diff_globals(), which contains
// the indices/constants necessary for the computation.
void taylor_c_compute_sv_diffs(llvm_state &s, const std::pair<std::array<llvm::GlobalVariable *, 6>, bool> &svd_gl,
                               llvm::Value *diff_arr, llvm::Value *par_ptr, std::uint32_t n_uvars, llvm::Value *order,
                               std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    // Fetch the global arrays and
    // the all_der_vars flag.
    const auto &sv_diff_gl = svd_gl.first;
    const auto all_der_vars = svd_gl.second;

    auto &builder = s.builder();

    // Recover the number of state variables whose derivatives are given
    // by u variables, numbers and params.
    const auto n_vars = taylor_c_gl_arr_size(sv_diff_gl[0]);
    const auto n_nums = taylor_c_gl_arr_size(sv_diff_gl[2]);
    const auto n_pars = taylor_c_gl_arr_size(sv_diff_gl[4]);

    // Fetch the type stored in the array of derivatives.
    auto *fp_vec_t = pointee_type(diff_arr);

    // Handle the u variables definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_vars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        // NOTE: if the time derivatives of all state variables are u variables, there's
        // no need to lookup the index in the global array (which will just contain
        // the values in the [0, n_vars] range).
        auto *sv_idx = all_der_vars
                           ? cur_idx
                           : builder.CreateLoad(builder.getInt32Ty(),
                                                builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[0]), sv_diff_gl[0],
                                                                          {builder.getInt32(0), cur_idx}));

        // Fetch the index of the u variable.
        auto *u_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[1]), sv_diff_gl[1], {builder.getInt32(0), cur_idx}));

        // Fetch from diff_arr the derivative of order 'order - 1' of the u variable u_idx.
        auto *ret = taylor_c_load_diff(s, diff_arr, n_uvars, builder.CreateSub(order, builder.getInt32(1)), u_idx);

        // We have to divide the derivative by 'order' in order
        // to get the normalised derivative of the state variable.
        ret = builder.CreateFDiv(
            ret, vector_splat(builder, builder.CreateUIToFP(order, fp_vec_t->getScalarType()), batch_size));

        // Store the derivative.
        taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx, ret);
    });

    // Handle the number definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_nums), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto *sv_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[2]), sv_diff_gl[2], {builder.getInt32(0), cur_idx}));

        // Fetch the constant.
        auto *num = builder.CreateLoad(
            fp_vec_t->getScalarType(),
            builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[3]), sv_diff_gl[3], {builder.getInt32(0), cur_idx}));

        // If the first-order derivative is being requested,
        // do the codegen for the constant itself, otherwise
        // return 0. No need for normalization as the only
        // nonzero value that can be produced here is the first-order
        // derivative.
        auto *cmp_cond = builder.CreateICmpEQ(order, builder.getInt32(1));
        auto ret = builder.CreateSelect(cmp_cond, vector_splat(builder, num, batch_size),
                                        llvm::ConstantFP::get(fp_vec_t, 0.));

        // Store the derivative.
        taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx, ret);
    });

    // Handle the param definitions.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_pars), [&](llvm::Value *cur_idx) {
        // Fetch the index of the state variable.
        auto *sv_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[4]), sv_diff_gl[4], {builder.getInt32(0), cur_idx}));

        // Fetch the index of the param.
        auto *par_idx = builder.CreateLoad(
            builder.getInt32Ty(),
            builder.CreateInBoundsGEP(pointee_type(sv_diff_gl[5]), sv_diff_gl[5], {builder.getInt32(0), cur_idx}));

        // If the first-order derivative is being requested,
        // do the codegen for the constant itself, otherwise
        // return 0. No need for normalization as the only
        // nonzero value that can be produced here is the first-order
        // derivative.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(order, builder.getInt32(1)),
            [&]() {
                // Derivative of order 1. Fetch the value from par_ptr.
                // NOTE: param{0} is unused, its only purpose is type tagging.
                taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx,
                                    taylor_c_diff_numparam_codegen(s, param{0}, par_idx, par_ptr, batch_size));
            },
            [&]() {
                // Derivative of order > 1, return 0.
                taylor_c_store_diff(s, diff_arr, n_uvars, order, sv_idx, llvm::ConstantFP::get(fp_vec_t, 0.));
            });
    });
}

} // namespace heyoka::detail
