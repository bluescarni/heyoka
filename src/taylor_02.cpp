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
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/llvm_vector_type.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/sleef.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

// NOTE: GCC warns about use of mismatched new/delete
// when creating global variables. I am not sure this is
// a real issue, as it looks like we are adopting the "canonical"
// approach for the creation of global variables (at least
// according to various sources online)
// and clang is not complaining. But let us revisit
// this issue in later LLVM versions.
#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

#endif

namespace heyoka
{

namespace detail
{

namespace
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
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, func>) {
                    std::vector<std::uint32_t> retval;

                    for (const auto &arg : v.args()) {
                        std::visit(
                            [&retval](const auto &x) {
                                using tp = uncvref_t<decltype(x)>;

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

    assert(counter == dc.size() - static_cast<decltype(dc.size())>(n_eq) * 2u);
#endif

    get_logger()->debug("Taylor decomposition N of segments: {}", s_dc.size());
    get_logger()->trace("Taylor decomposition segment runtime: {}", sw);

    return s_dc;
}

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

    auto *vec_t = to_llvm_vector_type<T>(context, batch_size);

    if (diff_variant.index() == 0u) {
        // Compact mode.
        auto *diff_arr = std::get<llvm::Value *>(diff_variant);

        // These will end up containing the norm infinity of the state vector + sv_funcs and the
        // norm infinity of the derivatives at orders order and order - 1.
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
                    auto *cur_idx = builder.CreateLoad(
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
    auto abs_or_rel = builder.CreateFCmpOLE(max_abs_state, llvm::ConstantFP::get(vec_t, 1.));

    // Estimate rho at orders order - 1 and order.
    auto num_rho = builder.CreateSelect(abs_or_rel, llvm::ConstantFP::get(vec_t, 1.), max_abs_state);
    auto rho_o = llvm_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_o),
                          vector_splat(builder, codegen<T>(s, number{static_cast<T>(1) / order}), batch_size));
    auto rho_om1 = llvm_pow(s, builder.CreateFDiv(num_rho, max_abs_diff_om1),
                            vector_splat(builder, codegen<T>(s, number{static_cast<T>(1) / (order - 1u)}), batch_size));

    // Take the minimum.
    auto rho_m = llvm_min(s, rho_o, rho_om1);

    // Compute the scaling + safety factor.
    const auto rhofac = exp((static_cast<T>(-7) / static_cast<T>(10)) / (order - 1u))
                        / (exp(static_cast<T>(1)) * exp(static_cast<T>(1)));

    // Determine the step size in absolute value.
    auto h = builder.CreateFMul(rho_m, vector_splat(builder, codegen<T>(s, number{rhofac}), batch_size));

    // Ensure that the step size does not exceed the limit in absolute value.
    auto *max_h_vec = load_vector_from_memory(builder, h_ptr, batch_size);
    h = taylor_step_minabs(s, h, max_h_vec);

    // Handle backwards propagation.
    auto backward = builder.CreateFCmpOLT(max_h_vec, llvm::ConstantFP::get(vec_t, 0.));
    auto h_fac = builder.CreateSelect(backward, llvm::ConstantFP::get(vec_t, -1.), llvm::ConstantFP::get(vec_t, 1.));
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

namespace
{

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
                    return llvm::ConstantFP::get(to_llvm_vector_type<T>(s.context(), batch_size), 0.);
                }
            } else {
                assert(false);
                return nullptr;
            }
        },
        ex.value());
}

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

// Small helper to compute the size of a global array.
std::uint32_t taylor_c_gl_arr_size(llvm::Value *v)
{
    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::GlobalVariable>(v)->getValueType())->getNumElements());
}

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
        auto *ret = builder.CreateSelect(cmp_cond, vector_splat(builder, num, batch_size),
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

// Helper to check if a vector of indices consists of consecutive values:
// [n, n + 1, n + 2, ...]
// NOTE: requires a non-empty vector.
bool is_consecutive(const std::vector<std::uint32_t> &v)
{
    assert(!v.empty());

    for (decltype(v.size()) i = 1; i < v.size(); ++i) {
        // NOTE: the first check is to avoid potential
        // negative overflow in the second check.
        if (v[i] <= v[i - 1u] || v[i] - v[i - 1u] != 1u) {
            return false;
        }
    }

    return true;
}

// Functions the create the arguments generators for the functions that compute
// the Taylor derivatives in compact mode. The generators are created from vectors
// of either u var indices (taylor_c_make_arg_gen_vidx()) or floating-point constants
// (taylor_c_make_arg_gen_vc()).
// NOTE: in these two functions we ensure that the return values do not capture
// any LLVM value except for types and global variables. This way we ensure that
// the generators do not rely on LLVM values created at the current insertion point.
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vidx(llvm_state &s,
                                                                       const std::vector<std::uint32_t> &ind)
{
    assert(!ind.empty()); // LCOV_EXCL_LINE

    auto &builder = s.builder();

    // Check if all indices in ind are the same.
    if (std::all_of(ind.begin() + 1, ind.end(), [&ind](const auto &n) { return n == ind[0]; })) {
        // If all indices are the same, don't construct an array, just always return
        // the same value.
        return [&builder, num = ind[0]](llvm::Value *) -> llvm::Value * { return builder.getInt32(num); };
    }

    // If ind consists of consecutive indices, we can replace
    // the index array with a simple offset computation.
    if (is_consecutive(ind)) {
        return [&builder, start_idx = ind[0]](llvm::Value *cur_call_idx) -> llvm::Value * {
            return builder.CreateAdd(builder.getInt32(start_idx), cur_call_idx);
        };
    }

    // Check if ind consists of a repeated pattern like [a, a, a, b, b, b, c, c, c, ...],
    // that is, [a X n, b X n, c X n, ...], such that [a, b, c, ...] are consecutive numbers.
    if (ind.size() > 1u) {
        // Determine the candidate number of repetitions.
        decltype(ind.size()) n_reps = 1;
        for (decltype(ind.size()) i = 1; i < ind.size(); ++i) {
            if (ind[i] == ind[i - 1u]) {
                ++n_reps;
            } else {
                break;
            }
        }

        if (n_reps > 1u && (ind.size() % n_reps) == 0u) {
            // There is an initial number of repetitions
            // and the vector size is a multiple of that.
            // See if the repetitions continue, and keep
            // track of the repeated indices.
            std::vector<std::uint32_t> rep_indices{ind[0]};

            bool rep_flag = true;

            // Iterate over the blocks of repetitions.
            for (decltype(ind.size()) rep_idx = 1; rep_idx < ind.size() / n_reps; ++rep_idx) {
                for (decltype(ind.size()) i = 1; i < n_reps; ++i) {
                    const auto cur_idx = rep_idx * n_reps + i;

                    if (ind[cur_idx] != ind[cur_idx - 1u]) {
                        rep_flag = false;
                        break;
                    }
                }

                if (rep_flag) {
                    rep_indices.push_back(ind[rep_idx * n_reps]);
                } else {
                    break;
                }
            }

            if (rep_flag && is_consecutive(rep_indices)) {
                // The pattern is  [a X n, b X n, c X n, ...] and [a, b, c, ...]
                // are consecutive numbers. The m-th value in the array can thus
                // be computed as a + floor(m / n).

#if !defined(NDEBUG)
                // Double-check the result in debug mode.
                std::vector<std::uint32_t> checker;
                for (decltype(ind.size()) i = 0; i < ind.size(); ++i) {
                    checker.push_back(boost::numeric_cast<std::uint32_t>(ind[0] + i / n_reps));
                }
                assert(checker == ind); // LCOV_EXCL_LINE
#endif

                return [&builder, start_idx = rep_indices[0], n_reps = boost::numeric_cast<std::uint32_t>(n_reps)](
                           llvm::Value *cur_call_idx) -> llvm::Value * {
                    return builder.CreateAdd(builder.getInt32(start_idx),
                                             builder.CreateUDiv(cur_call_idx, builder.getInt32(n_reps)));
                };
            }
        }
    }

    auto &md = s.module();

    // Generate the array of indices as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    tmp_c_vec.reserve(ind.size());
    for (const auto &val : ind) {
        tmp_c_vec.push_back(builder.getInt32(val));
    }

    // Create the array type.
    auto *arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(ind.size()));
    assert(arr_type != nullptr); // LCOV_EXCL_LINE

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr); // LCOV_EXCL_LINE
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar
        = new llvm::GlobalVariable(md, const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, const_arr);

    // Return the generator.
    return [&builder, gvar, arr_type](llvm::Value *cur_call_idx) -> llvm::Value * {
        assert(llvm_depr_GEP_type_check(gvar, arr_type)); // LCOV_EXCL_LINE
        return builder.CreateLoad(builder.getInt32Ty(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

template <typename T>
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vc(llvm_state &s, const std::vector<number> &vc)
{
    assert(!vc.empty()); // LCOV_EXCL_LINE

    // Check if all the numbers are the same.
    // NOTE: the comparison operator of number will consider two numbers of different
    // type but equal value to be equal.
    if (std::all_of(vc.begin() + 1, vc.end(), [&vc](const auto &n) { return n == vc[0]; })) {
        // If all constants are the same, don't construct an array, just always return
        // the same value.
        return [&s, num = vc[0]](llvm::Value *) -> llvm::Value * { return codegen<T>(s, num); };
    }

    // Generate the array of constants as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    tmp_c_vec.reserve(vc.size());
    for (const auto &val : vc) {
        tmp_c_vec.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, val)));
    }

    // Create the array type.
    auto *arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(vc.size()));
    assert(arr_type != nullptr); // LCOV_EXCL_LINE

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr); // LCOV_EXCL_LINE
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar = new llvm::GlobalVariable(s.module(), const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage,
                                          const_arr);

    // Return the generator.
    return [&s, gvar, arr_type](llvm::Value *cur_call_idx) -> llvm::Value * {
        auto &builder = s.builder();

        assert(llvm_depr_GEP_type_check(gvar, arr_type)); // LCOV_EXCL_LINE
        return builder.CreateLoad(arr_type->getArrayElementType(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

// For each segment in s_dc, this function will return a dict mapping an LLVM function
// f for the computation of a Taylor derivative to a size and a vector of std::functions. For example, one entry
// in the return value will read something like:
// {f : (2, [g_0, g_1, g_2])}
// The meaning in this example is that the arity of f is 3 and it will be called with 2 different
// sets of arguments. The g_i functions are expected to be called with input argument j in [0, 1]
// to yield the value of the i-th function argument for f at the j-th invocation.
template <typename T>
auto taylor_build_function_maps(llvm_state &s, const std::vector<taylor_dc_t> &s_dc, std::uint32_t n_eq,
                                std::uint32_t n_uvars, std::uint32_t batch_size, bool high_accuracy)
{
    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    // Init the return value.
    // NOTE: use maps with name-based comparison for the functions. This ensures that the order in which these
    // functions are invoked in taylor_compute_jet_compact_mode() is always the same. If we used directly pointer
    // comparisons instead, the order could vary across different executions and different platforms. The name
    // mangling we do when creating the function names should ensure that there are no possible name collisions.
    std::vector<
        std::map<llvm::Function *, std::pair<std::uint32_t, std::vector<std::function<llvm::Value *(llvm::Value *)>>>,
                 llvm_func_name_compare>>
        retval;

    // Variable to keep track of the u variable
    // on whose definition we are operating.
    auto cur_u_idx = n_eq;
    for (const auto &seg : s_dc) {
        // This structure maps an LLVM function to sets of arguments
        // with which the function is to be called. For instance, if function
        // f(x, y, z) is to be called as f(a, b, c) and f(d, e, f), then tmp_map
        // will contain {f : [[a, b, c], [d, e, f]]}.
        // After construction, we have verified that for each function
        // in the map the sets of arguments have all the same size.
        std::unordered_map<llvm::Function *, std::vector<std::vector<std::variant<std::uint32_t, number>>>> tmp_map;

        for (const auto &ex : seg) {
            // Get the function for the computation of the derivative.
            auto *func = taylor_c_diff_func<T>(s, ex.first, n_uvars, batch_size, high_accuracy);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty()); // LCOV_EXCL_LINE

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto cdiff_args = udef_to_variants(ex.first, ex.second);

            if (!is_new_func && it->second.back().size() - 1u != cdiff_args.size()) {
                throw std::invalid_argument(
                    fmt::format("Inconsistent arity detected in a Taylor derivative function in compact "
                                "mode: the same function is being called with both {} and {} arguments",
                                it->second.back().size() - 1u, cdiff_args.size()));
            }

            // Add the new set of arguments.
            it->second.emplace_back();
            // Add the idx of the u variable.
            it->second.back().emplace_back(cur_u_idx);
            // Add the actual function arguments.
            it->second.back().insert(it->second.back().end(), cdiff_args.begin(), cdiff_args.end());

            ++cur_u_idx;
        }

        // Now we build the transposition of tmp_map: from {f : [[a, b, c], [d, e, f]]}
        // to {f : [[a, d], [b, e], [c, f]]}.
        std::unordered_map<llvm::Function *, std::vector<std::variant<std::vector<std::uint32_t>, std::vector<number>>>>
            tmp_map_transpose;
        for (const auto &[func, vv] : tmp_map) {
            assert(!vv.empty()); // LCOV_EXCL_LINE

            // Add the function.
            const auto [it, ins_status] = tmp_map_transpose.try_emplace(func);
            assert(ins_status); // LCOV_EXCL_LINE

            const auto n_calls = vv.size();
            const auto n_args = vv[0].size();
            // NOTE: n_args must be at least 1 because the u idx
            // is prepended to the actual function arguments in
            // the tmp_map entries.
            assert(n_args >= 1u); // LCOV_EXCL_LINE

            for (decltype(vv[0].size()) i = 0; i < n_args; ++i) {
                // Build the vector of values corresponding
                // to the current argument index.
                std::vector<std::variant<std::uint32_t, number>> tmp_c_vec;
                for (decltype(vv.size()) j = 0; j < n_calls; ++j) {
                    tmp_c_vec.push_back(vv[j][i]);
                }

                // Turn tmp_c_vec (a vector of variants) into a variant
                // of vectors, and insert the result.
                it->second.push_back(vv_transpose(tmp_c_vec));
            }
        }

        // Add a new entry in retval for the current segment.
        retval.emplace_back();
        auto &a_map = retval.back();

        for (const auto &[func, vv] : tmp_map_transpose) {
            assert(!vv.empty()); // LCOV_EXCL_LINE

            // Add the function.
            const auto [it, ins_status] = a_map.try_emplace(func);
            assert(ins_status); // LCOV_EXCL_LINE

            // Set the number of calls for this function.
            it->second.first
                = std::visit([](const auto &x) { return boost::numeric_cast<std::uint32_t>(x.size()); }, vv[0]);
            assert(it->second.first > 0u); // LCOV_EXCL_LINE

            // Create the g functions for each argument.
            for (const auto &v : vv) {
                it->second.second.push_back(std::visit(
                    [&s](const auto &x) {
                        using type = uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<type, std::vector<std::uint32_t>>) {
                            return taylor_c_make_arg_gen_vidx(s, x);
                        } else {
                            return taylor_c_make_arg_gen_vc<T>(s, x);
                        }
                    },
                    v));
            }
        }
    }

    get_logger()->trace("Taylor build function maps runtime: {}", sw);

    // LCOV_EXCL_START
    // Log a breakdown of the return value in trace mode.
    if (get_logger()->should_log(spdlog::level::trace)) {
        std::vector<std::vector<std::uint32_t>> fm_bd;

        for (const auto &m : retval) {
            fm_bd.emplace_back();

            for (const auto &p : m) {
                fm_bd.back().push_back(p.second.first);
            }
        }

        get_logger()->trace("Taylor function maps breakdown: {}", fm_bd);
    }
    // LCOV_EXCL_STOP

    return retval;
}

// Helper to create a global zero-inited array variable in the module m
// with type t. The array is mutable and with internal linkage.
llvm::GlobalVariable *make_global_zero_array(llvm::Module &m, llvm::ArrayType *t)
{
    assert(t != nullptr); // LCOV_EXCL_LINE

    // Make the global array.
    auto *gl_arr = new llvm::GlobalVariable(m, t, false, llvm::GlobalVariable::InternalLinkage,
                                            llvm::ConstantAggregateZero::get(t));

    // Return it.
    return gl_arr;
}

} // namespace

// Helper for the computation of a jet of derivatives in compact mode,
// used in taylor_compute_jet().
template <typename T>
llvm::Value *taylor_compute_jet_compact_mode(llvm_state &s, llvm::Value *order0, llvm::Value *par_ptr,
                                             llvm::Value *time_ptr, const taylor_dc_t &dc,
                                             const std::vector<std::uint32_t> &sv_funcs_dc, std::uint32_t n_eq,
                                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                             bool high_accuracy, bool parallel_mode)
{
    auto &builder = s.builder();
    auto &context = s.context();
    auto &md = s.module();

    // Split dc into segments.
    const auto s_dc = taylor_segment_dc(dc, n_eq);

    // Generate the function maps.
    const auto f_maps = taylor_build_function_maps<T>(s, s_dc, n_eq, n_uvars, batch_size, high_accuracy);

    // Log the runtime of IR construction in trace mode.
    spdlog::stopwatch sw;

    // Generate the global arrays for the computation of the derivatives
    // of the state variables.
    const auto svd_gl = taylor_c_make_sv_diff_globals<T>(s, dc, n_uvars);

    // Determine the maximum u variable index appearing in sv_funcs_dc, or zero
    // if sv_funcs_dc is empty.
    const auto max_svf_idx = sv_funcs_dc.empty() ? static_cast<std::uint32_t>(0)
                                                 : *std::max_element(sv_funcs_dc.begin(), sv_funcs_dc.end());

    // Prepare the array that will contain the jet of derivatives.
    // We will be storing all the derivatives of the u variables
    // up to order 'order - 1', the derivatives of order
    // 'order' of the state variables and the derivatives
    // of order 'order' of the sv_funcs.
    // NOTE: the array size is specified as a 64-bit integer in the
    // LLVM API.
    // NOTE: fp_type is the original, scalar floating-point type.
    // It will be turned into a vector type (if necessary) by
    // make_vector_type() below.
    // NOTE: if sv_funcs_dc is empty, or if all its indices are not greater
    // than the indices of the state variables, then we don't need additional
    // slots after the sv derivatives. If we need additional slots, allocate
    // another full column of derivatives, as it is complicated at this stage
    // to know exactly how many slots we will need.
    auto *fp_type = to_llvm_type<T>(context);
    auto *fp_vec_type = make_vector_type(fp_type, batch_size);
    auto *array_type
        = llvm::ArrayType::get(fp_vec_type, (max_svf_idx < n_eq) ? (n_uvars * order + n_eq) : (n_uvars * (order + 1u)));

    // Make the global array and fetch a pointer to its first element.
    // NOTE: we use a global array rather than a local one here because
    // its size can grow quite large, which can lead to stack overflow issues.
    // This has of course consequences in terms of thread safety, which
    // we will have to document.
    auto diff_arr_gvar = make_global_zero_array(md, array_type);
    assert(llvm_depr_GEP_type_check(diff_arr_gvar, array_type)); // LCOV_EXCL_LINE
    auto *diff_arr = builder.CreateInBoundsGEP(array_type, diff_arr_gvar, {builder.getInt32(0), builder.getInt32(0)});

    // Copy over the order-0 derivatives of the state variables.
    // NOTE: overflow checking is already done in the parent function.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
        // Fetch the pointer from order0.
        assert(llvm_depr_GEP_type_check(order0, fp_type)); // LCOV_EXCL_LINE
        auto *ptr
            = builder.CreateInBoundsGEP(fp_type, order0, builder.CreateMul(cur_var_idx, builder.getInt32(batch_size)));

        // Load as a vector.
        auto *vec = load_vector_from_memory(builder, ptr, batch_size);

        // Store into diff_arr.
        taylor_c_store_diff(s, diff_arr, n_uvars, builder.getInt32(0), cur_var_idx, vec);
    });

    // NOTE: these are used only in parallel mode.
    std::vector<std::vector<llvm::AllocaInst *>> par_funcs_ptrs;
    llvm::Value *gl_par_data = nullptr;
    llvm::Type *par_data_t = nullptr;

    if (parallel_mode) {
        // Fetch the LLVM version of T *.
        auto *scal_ptr_t = llvm::PointerType::getUnqual(to_llvm_type<T>(context));

        // NOTE: we will use a global variable with these fields:
        //
        // - int32 (current Taylor order),
        // - T * (pointer to the runtime parameters),
        // - T * (pointer to the time coordinate(s)),
        //
        // to pass the data necessary to the parallel workers.
        par_data_t = llvm::StructType::get(context, {builder.getInt32Ty(), scal_ptr_t, scal_ptr_t});
        gl_par_data = new llvm::GlobalVariable(md, par_data_t, false, llvm::GlobalVariable::InternalLinkage,
                                               llvm::ConstantAggregateZero::get(par_data_t));

        // Write the par/time pointers into the global struct (unlike the current order, this needs
        // to be done only once).
        assert(llvm_depr_GEP_type_check(gl_par_data, par_data_t)); // LCOV_EXCL_LINE
        builder.CreateStore(
            par_ptr, builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(1)}));
        builder.CreateStore(
            time_ptr, builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(2)}));

        // Fetch the function types for the parallel worker and the wrapper.
        auto *worker_t
            = llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty(), builder.getInt32Ty()}, false);
        assert(worker_t != nullptr); // LCOV_EXCL_LINE

        auto *wrapper_t = llvm::FunctionType::get(builder.getVoidTy(), {}, false);
        assert(wrapper_t != nullptr); // LCOV_EXCL_LINE

        for (const auto &map : f_maps) {
            par_funcs_ptrs.emplace_back();

            for (const auto &p : map) {
                // The LLVM function for the computation of the
                // derivative in compact mode.
                const auto &func = p.first;

                // The number of func calls.
                const auto ncalls = p.second.first;

                // The generators for the arguments of func.
                const auto &gens = p.second.second;

                // Fetch the current insertion block.
                auto *orig_bb = builder.GetInsertBlock();

                // Create the worker function.
                auto *worker = llvm::Function::Create(worker_t, llvm::Function::InternalLinkage, "", &md);
                assert(worker != nullptr); // LCOV_EXCL_LINE

                // Fetch the function arguments.
                auto *b_idx = worker->args().begin();
                auto *e_idx = worker->args().begin() + 1;

                // Create a new basic block to start insertion into.
                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", worker));

                // Load the order and par/time pointers from the global variable.
                auto *cur_order = builder.CreateLoad(
                    builder.getInt32Ty(),
                    builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(0)}));
                auto par_arg = builder.CreateLoad(
                    scal_ptr_t,
                    builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(1)}));
                auto time_arg = builder.CreateLoad(
                    scal_ptr_t,
                    builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(2)}));

                // Iterate over the range.
                llvm_loop_u32(s, b_idx, e_idx, [&](llvm::Value *cur_call_idx) {
                    // Create the u variable index from the first generator.
                    auto u_idx = gens[0](cur_call_idx);

                    // Initialise the vector of arguments with which func must be called. The following
                    // initial arguments are always present:
                    // - current Taylor order,
                    // - u index of the variable,
                    // - array of derivatives,
                    // - pointer to the param values,
                    // - pointer to the time value(s).
                    std::vector<llvm::Value *> args{cur_order, u_idx, diff_arr, par_arg, time_arg};

                    // Create the other arguments via the generators.
                    for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                        args.push_back(gens[i](cur_call_idx));
                    }

                    // Calculate the derivative and store the result.
                    taylor_c_store_diff(s, diff_arr, n_uvars, cur_order, u_idx, builder.CreateCall(func, args));
                });

                // Return.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(worker);

                // Create the wrapper function. This will execute multiple calls
                // to the worker in parallel, until the entire range [0, ncalls) has
                // been consumed.
                auto *wrapper = llvm::Function::Create(wrapper_t, llvm::Function::InternalLinkage, "", &md);
                assert(wrapper != nullptr); // LCOV_EXCL_LINE

                // Create a new basic block to start insertion into.
                builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", wrapper));

                // Invoke the parallel looper.
                llvm_invoke_external(s, "heyoka_cm_par_looper", builder.getVoidTy(), {builder.getInt32(ncalls), worker},
                                     {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});

                // Return.
                builder.CreateRetVoid();

                // Verify.
                s.verify_function(wrapper);

                // Restore the original insertion block.
                builder.SetInsertPoint(orig_bb);

                // Add a pointer to the wrapper to par_funcs_ptrs.
                auto *f_ptr = builder.CreateAlloca(wrapper->getType());
                builder.CreateStore(wrapper, f_ptr);
                par_funcs_ptrs.back().push_back(f_ptr);
            }
        }
    }

    // Helper to compute the Taylor derivatives for a block.
    // func is the LLVM function for the computation of the Taylor derivative in the block,
    // ncalls the number of times it must be called, gens the generators for the
    // function arguments and cur_order the order of the derivative.
    auto block_diff = [&](const auto &func, const auto &ncalls, const auto &gens, llvm::Value *cur_order) {
        // LCOV_EXCL_START
        assert(ncalls > 0u);
        assert(!gens.empty());
        assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));
        // LCOV_EXCL_STOP

        // Loop over the number of calls.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
            // Create the u variable index from the first generator.
            auto u_idx = gens[0](cur_call_idx);

            // Initialise the vector of arguments with which func must be called. The following
            // initial arguments are always present:
            // - current Taylor order,
            // - u index of the variable,
            // - array of derivatives,
            // - pointer to the param values,
            // - pointer to the time value(s).
            std::vector<llvm::Value *> args{cur_order, u_idx, diff_arr, par_ptr, time_ptr};

            // Create the other arguments via the generators.
            for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                args.push_back(gens[i](cur_call_idx));
            }

            // Calculate the derivative and store the result.
            taylor_c_store_diff(s, diff_arr, n_uvars, cur_order, u_idx, builder.CreateCall(func, args));
        });
    };

    // Helper to compute concurrently all the derivatives
    // in a segment using the parallel wrappers.
    auto parallel_segment_diff = [&](const auto &pfptrs) {
        assert(!pfptrs.empty()); // LCOV_EXCL_LINE

        // NOTE: we can invoke in parallel only up to a fixed number
        // of wrappers. Thus, we process them in chunks.

        // The remaining number of wrappers to invoke.
        auto rem = pfptrs.size();

        // Starting index in pfptrs.
        decltype(rem) start_idx = 0;

        while (rem != 0u) {
            // Current chunk size.
            const auto cur_size = std::min(static_cast<decltype(rem)>(HEYOKA_CM_PAR_MAX_INVOKE_N), rem);

            // Setup the function name.
            const auto fname = fmt::format("heyoka_cm_par_invoke_{}", cur_size);

            // Setup the function arguments.
            std::vector<llvm::Value *> args;
            for (auto i = start_idx; i < start_idx + cur_size; ++i) {
                assert(i < pfptrs.size()); // LCOV_EXCL_LINE
                auto *ptr = pfptrs[i];
                args.push_back(builder.CreateLoad(ptr->getAllocatedType(), ptr));
            }

            // Invoke.
            llvm_invoke_external(s, fname, builder.getVoidTy(), args,
                                 {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn});

            // Update rem and start_idx.
            rem -= cur_size;
            start_idx += cur_size;
        }
    };

    // Helper to compute and store the derivatives of order cur_order
    // of the u variables which are not state variables.
    auto compute_u_diffs = [&](llvm::Value *cur_order) {
        if (parallel_mode) {
            // Store the current order in the global struct.
            builder.CreateStore(cur_order, builder.CreateInBoundsGEP(par_data_t, gl_par_data,
                                                                     {builder.getInt32(0), builder.getInt32(0)}));

            // For each segment, invoke the wrapper functions concurrently.
            for (const auto &pfptrs : par_funcs_ptrs) {
                parallel_segment_diff(pfptrs);
            }
        } else {
            // For each block in each segment, compute the derivatives
            // of order cur_order serially.
            for (const auto &map : f_maps) {
                for (const auto &p : map) {
                    block_diff(p.first, p.second.first, p.second.second, cur_order);
                }
            }
        }
    };

    // Compute the order-0 derivatives (i.e., the initial values)
    // for all u variables which are not state variables.
    compute_u_diffs(builder.getInt32(0));

    // Compute all derivatives up to order 'order - 1'.
    llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(order), [&](llvm::Value *cur_order) {
        // State variables first.
        taylor_c_compute_sv_diffs(s, svd_gl, diff_arr, par_ptr, n_uvars, cur_order, batch_size);

        // The other u variables.
        compute_u_diffs(cur_order);
    });

    // Compute the last-order derivatives for the state variables.
    taylor_c_compute_sv_diffs(s, svd_gl, diff_arr, par_ptr, n_uvars, builder.getInt32(order), batch_size);

    // Compute the last-order derivatives for the sv_funcs, if any. Because the sv funcs
    // correspond to u variables in the decomposition, we will have to compute the
    // last-order derivatives of the u variables until we are sure all sv_funcs derivatives
    // have been properly computed.
    if (max_svf_idx >= n_eq) {
        // Monitor the starting index of the current
        // segment while iterating on the segments.
        auto cur_start_u_idx = n_eq;

        if (parallel_mode) {
            // Store the derivative order in the global struct.
            builder.CreateStore(
                builder.getInt32(order),
                builder.CreateInBoundsGEP(par_data_t, gl_par_data, {builder.getInt32(0), builder.getInt32(0)}));

            for (decltype(f_maps.size()) i = 0; i < f_maps.size(); ++i) {
                if (cur_start_u_idx > max_svf_idx) {
                    // We computed all the necessary derivatives, break out.
                    break;
                }

                // Compute the derivatives for the current segment.
                parallel_segment_diff(par_funcs_ptrs[i]);

                // Update cur_start_u_idx, taking advantage of the fact
                // that each block in a segment processes the derivatives
                // of exactly ncalls u variables.
                for (const auto &p : f_maps[i]) {
                    const auto ncalls = p.second.first;
                    cur_start_u_idx += ncalls;
                }
            }
        } else {
            for (const auto &map : f_maps) {
                if (cur_start_u_idx > max_svf_idx) {
                    // We computed all the necessary derivatives, break out.
                    break;
                }

                // Compute the derivatives of all the blocks in the segment.
                for (const auto &p : map) {
                    const auto ncalls = p.second.first;

                    block_diff(p.first, ncalls, p.second.second, builder.getInt32(order));

                    // Update cur_start_u_idx taking advantage of the fact
                    // that each block in a segment processes the derivatives
                    // of exactly ncalls u variables.
                    cur_start_u_idx += ncalls;
                }
            }
        }
    }

    get_logger()->trace("Taylor IR creation compact mode runtime: {}", sw);

    // Return the array of derivatives of the u variables.
    return diff_arr;
}

// Explicit instantiations of taylor_compute_jet_compact_mode().
template llvm::Value *taylor_compute_jet_compact_mode<double>(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *,
                                                              const taylor_dc_t &, const std::vector<std::uint32_t> &,
                                                              std::uint32_t, std::uint32_t, std::uint32_t,
                                                              std::uint32_t, bool, bool);

template llvm::Value *taylor_compute_jet_compact_mode<long double>(llvm_state &, llvm::Value *, llvm::Value *,
                                                                   llvm::Value *, const taylor_dc_t &,
                                                                   const std::vector<std::uint32_t> &, std::uint32_t,
                                                                   std::uint32_t, std::uint32_t, std::uint32_t, bool,
                                                                   bool);

#if defined(HEYOKA_HAVE_REAL128)

template llvm::Value *taylor_compute_jet_compact_mode<mppp::real128>(llvm_state &, llvm::Value *, llvm::Value *,
                                                                     llvm::Value *, const taylor_dc_t &,
                                                                     const std::vector<std::uint32_t> &, std::uint32_t,
                                                                     std::uint32_t, std::uint32_t, std::uint32_t, bool,
                                                                     bool);

#endif

namespace
{

// Given an input pointer 'in', load the first n * batch_size values in it as n vectors
// with size batch_size. If batch_size is 1, the values will be loaded as scalars.
auto taylor_load_values(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    auto &builder = s.builder();

    std::vector<llvm::Value *> retval;
    for (std::uint32_t i = 0; i < n; ++i) {
        // Fetch the pointer from in.
        // NOTE: overflow checking is done in the parent function.
        assert(llvm_depr_GEP_type_check(in, pointee_type(in))); // LCOV_EXCL_LINE
        auto *ptr = builder.CreateInBoundsGEP(pointee_type(in), in, builder.getInt32(i * batch_size));

        // Load the value in vector mode.
        retval.push_back(load_vector_from_memory(builder, ptr, batch_size));
    }

    return retval;
}

} // namespace

// Helper function to compute the jet of Taylor derivatives up to a given order. n_eq
// is the number of equations/variables in the ODE sys, dc its Taylor decomposition,
// n_uvars the total number of u variables in the decomposition.
// order is the max derivative order desired, batch_size the batch size.
// order0 is a pointer to an array of (at least) n_eq * batch_size scalar elements
// containing the derivatives of order 0. par_ptr is a pointer to an array containing
// the numerical values of the parameters, time_ptr a pointer to the time value(s).
// sv_funcs are the indices, in the decomposition, of the functions of state
// variables.
//
// The return value is a variant containing either:
// - in compact mode, the array containing the derivatives of all u variables,
// - otherwise, the jet of derivatives of the state variables and sv_funcs
//   up to order 'order'.
template <typename T>
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_compute_jet(llvm_state &s, llvm::Value *order0, llvm::Value *par_ptr, llvm::Value *time_ptr,
                   const taylor_dc_t &dc, const std::vector<std::uint32_t> &sv_funcs_dc, std::uint32_t n_eq,
                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size, bool compact_mode,
                   bool high_accuracy, bool parallel_mode)
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(n_eq > 0u);
    assert(order > 0u);
    // LCOV_EXCL_STOP

    // Make sure we can represent n_uvars * (order + 1) as a 32-bit
    // unsigned integer. This is the maximum total number of derivatives we will have to compute
    // and store, with the +1 taking into account the extra slots that might be needed by sv_funcs_dc.
    // If sv_funcs_dc is empty, we need only n_uvars * order + n_eq derivatives.
    // LCOV_EXCL_START
    if (order == std::numeric_limits<std::uint32_t>::max()
        || n_uvars > std::numeric_limits<std::uint32_t>::max() / (order + 1u)) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }

    // We also need to be able to index up to n_eq * batch_size in order0.
    if (n_eq > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error(
            "An overflow condition was detected in the computation of a jet of Taylor derivatives");
    }
    // LCOV_EXCL_STOP

    if (compact_mode) {
        // In compact mode, we need to ensure that we can index into par_ptr using std::uint32_t.
        // NOTE: in default mode the check is done inside taylor_codegen_numparam()
        // during the construction of the IR code.
        // In compact mode we cannot do that, as the determination of the index into
        // par_ptr is done *within* the IR code (compare taylor_codegen_numparam()
        // to taylor_c_diff_numparam_codegen()).

        // Deduce the size of the param array from the expressions in the decomposition.
        const auto param_size = n_pars_in_dc(dc);
        // LCOV_EXCL_START
        if (param_size > std::numeric_limits<std::uint32_t>::max() / batch_size) {
            throw std::overflow_error(
                "An overflow condition was detected in the computation of a jet of Taylor derivatives in compact mode");
        }
        // LCOV_EXCL_STOP

        return taylor_compute_jet_compact_mode<T>(s, order0, par_ptr, time_ptr, dc, sv_funcs_dc, n_eq, n_uvars, order,
                                                  batch_size, high_accuracy, parallel_mode);
    } else {
        // Log the runtime of IR construction in trace mode.
        spdlog::stopwatch sw;

        // Init the derivatives array with the order 0 of the state variables.
        auto diff_arr = taylor_load_values(s, order0, n_eq, batch_size);

        // Compute the order-0 derivatives of the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            diff_arr.push_back(taylor_diff<T>(s, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars, 0, i,
                                              batch_size, high_accuracy));
        }

        // Compute the derivatives order by order, starting from 1 to order excluded.
        // We will compute the highest derivatives of the state variables separately
        // in the last step.
        for (std::uint32_t cur_order = 1; cur_order < order; ++cur_order) {
            // Begin with the state variables.
            // NOTE: the derivatives of the state variables
            // are at the end of the decomposition vector.
            for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
                diff_arr.push_back(
                    taylor_compute_sv_diff<T>(s, dc[i].first, diff_arr, par_ptr, n_uvars, cur_order, batch_size));
            }

            // Now the other u variables.
            for (auto i = n_eq; i < n_uvars; ++i) {
                diff_arr.push_back(taylor_diff<T>(s, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars,
                                                  cur_order, i, batch_size, high_accuracy));
            }
        }

        // Compute the last-order derivatives for the state variables.
        for (auto i = n_uvars; i < boost::numeric_cast<std::uint32_t>(dc.size()); ++i) {
            diff_arr.push_back(
                taylor_compute_sv_diff<T>(s, dc[i].first, diff_arr, par_ptr, n_uvars, order, batch_size));
        }

        // If there are sv funcs, we need to compute their last-order derivatives too:
        // we will need to compute the derivatives of the u variables up to
        // the maximum index in sv_funcs_dc.
        const auto max_svf_idx = sv_funcs_dc.empty() ? static_cast<std::uint32_t>(0)
                                                     : *std::max_element(sv_funcs_dc.begin(), sv_funcs_dc.end());

        // NOTE: if there are no sv_funcs, max_svf_idx is set to zero
        // above, thus we never enter the loop.
        // NOTE: <= because max_svf_idx is an index, not a size.
        for (std::uint32_t i = n_eq; i <= max_svf_idx; ++i) {
            diff_arr.push_back(taylor_diff<T>(s, dc[i].first, dc[i].second, diff_arr, par_ptr, time_ptr, n_uvars, order,
                                              i, batch_size, high_accuracy));
        }

#if !defined(NDEBUG)
        if (sv_funcs_dc.empty()) {
            assert(diff_arr.size() == static_cast<decltype(diff_arr.size())>(n_uvars) * order + n_eq);
        } else {
            // NOTE: we use std::max<std::uint32_t>(n_eq, max_svf_idx + 1u) here because
            // the sv funcs could all be aliases of the state variables themselves,
            // in which case in the previous loop we ended up appending nothing.
            assert(diff_arr.size()
                   == static_cast<decltype(diff_arr.size())>(n_uvars) * order
                          + std::max<std::uint32_t>(n_eq, max_svf_idx + 1u));
        }
#endif

        // Extract the derivatives of the state variables and sv_funcs from diff_arr.
        std::vector<llvm::Value *> retval;
        for (std::uint32_t o = 0; o <= order; ++o) {
            for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
                retval.push_back(taylor_fetch_diff(diff_arr, var_idx, o, n_uvars));
            }
            for (auto idx : sv_funcs_dc) {
                retval.push_back(taylor_fetch_diff(diff_arr, idx, o, n_uvars));
            }
        }

        get_logger()->trace("Taylor IR creation default mode runtime: {}", sw);

        return retval;
    }
}

// Explicit instantiations of taylor_compute_jet_().
template std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_compute_jet<double>(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *, const taylor_dc_t &,
                           const std::vector<std::uint32_t> &, std::uint32_t, std::uint32_t, std::uint32_t,
                           std::uint32_t, bool, bool, bool);

template std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_compute_jet<long double>(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *, const taylor_dc_t &,
                                const std::vector<std::uint32_t> &, std::uint32_t, std::uint32_t, std::uint32_t,
                                std::uint32_t, bool, bool, bool);

#if defined(HEYOKA_HAVE_REAL128)

template std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_compute_jet<mppp::real128>(llvm_state &, llvm::Value *, llvm::Value *, llvm::Value *, const taylor_dc_t &,
                                  const std::vector<std::uint32_t> &, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, bool, bool, bool);

#endif

// Helper to generate the LLVM code to store the Taylor coefficients of the state variables and
// the sv funcs into an external array. The Taylor polynomials are stored in row-major order,
// first the state variables and after the sv funcs. For use in the adaptive timestepper implementations.
void taylor_write_tc(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_variant,
                     const std::vector<std::uint32_t> &sv_funcs_dc, llvm::Value *svf_ptr, llvm::Value *tc_ptr,
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

    // Convert to std::uint32_t for overflow checking and use below.
    const auto n_sv_funcs = boost::numeric_cast<std::uint32_t>(sv_funcs_dc.size());

    // Overflow checking: ensure we can index into
    // tc_ptr using std::uint32_t.
    // NOTE: this is the same check done in taylor_add_jet_impl().
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

        auto *diff_arr = std::get<llvm::Value *>(diff_variant);

        // Write out the Taylor coefficients for the state variables.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var) {
            llvm_loop_u32(s, builder.getInt32(0), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                          [&](llvm::Value *cur_order) {
                              // Load the value of the derivative from diff_arr.
                              auto *diff_val = taylor_c_load_diff(s, diff_arr, n_uvars, cur_order, cur_var);

                              // Compute the index in the output pointer.
                              auto *out_idx = builder.CreateAdd(
                                  builder.CreateMul(builder.getInt32((order + 1u) * batch_size), cur_var),
                                  builder.CreateMul(cur_order, builder.getInt32(batch_size)));

                              // Store into tc_ptr.
                              // LCOV_EXCL_START
                              assert(llvm_depr_GEP_type_check(tc_ptr, pointee_type(tc_ptr)));
                              // LCOV_EXCL_STOP
                              store_vector_to_memory(
                                  builder, builder.CreateInBoundsGEP(pointee_type(tc_ptr), tc_ptr, out_idx), diff_val);
                          });
        });

        // Write out the Taylor coefficients for the sv funcs, if necessary.
        if (svf_ptr != nullptr) {
            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_sv_funcs), [&](llvm::Value *arr_idx) {
                // Fetch the u var index from svf_ptr.
                assert(llvm_depr_GEP_type_check(svf_ptr, builder.getInt32Ty())); // LCOV_EXCL_LINE
                auto *cur_idx = builder.CreateLoad(builder.getInt32Ty(),
                                                   builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                llvm_loop_u32(
                    s, builder.getInt32(0), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
                    [&](llvm::Value *cur_order) {
                        // Load the derivative value from diff_arr.
                        auto *diff_val = taylor_c_load_diff(s, diff_arr, n_uvars, cur_order, cur_idx);

                        // Compute the index in the output pointer.
                        auto *out_idx
                            = builder.CreateAdd(builder.CreateMul(builder.getInt32((order + 1u) * batch_size),
                                                                  builder.CreateAdd(builder.getInt32(n_eq), arr_idx)),
                                                builder.CreateMul(cur_order, builder.getInt32(batch_size)));

                        // Store into tc_ptr.
                        // LCOV_EXCL_START
                        assert(llvm_depr_GEP_type_check(tc_ptr, pointee_type(tc_ptr)));
                        // LCOV_EXCL_STOP
                        store_vector_to_memory(
                            builder, builder.CreateInBoundsGEP(pointee_type(tc_ptr), tc_ptr, out_idx), diff_val);
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
                const auto out_idx = (order + 1u) * batch_size * j + cur_order * batch_size;

                // Write to tc_ptr.
                assert(llvm_depr_GEP_type_check(tc_ptr, pointee_type(tc_ptr))); // LCOV_EXCL_LINE
                auto *out_ptr = builder.CreateInBoundsGEP(pointee_type(tc_ptr), tc_ptr,
                                                          builder.getInt32(static_cast<std::uint32_t>(out_idx)));
                store_vector_to_memory(builder, out_ptr, val);
            }
        }
    }
}

// Run the Horner scheme to propagate an ODE state via the evaluation of the Taylor polynomials.
// diff_var contains either the derivatives for all u variables (in compact mode) or only
// for the state variables (non-compact mode). The evaluation point (i.e., the timestep)
// is h. The evaluation is run in parallel over the polynomials of all the state
// variables.
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_multihorner(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_var,
                       llvm::Value *h, std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                       bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.
        auto *diff_arr = std::get<llvm::Value *>(diff_var);

        // Create the array storing the results of the evaluation.
        auto *fp_vec_t = pointee_type(diff_arr);
        auto *array_type = llvm::ArrayType::get(fp_vec_t, n_eq);
        auto *array_inst = builder.CreateAlloca(array_type);
        assert(llvm_depr_GEP_type_check(array_inst, array_type)); // LCOV_EXCL_LINE
        auto *res_arr = builder.CreateInBoundsGEP(array_type, array_inst, {builder.getInt32(0), builder.getInt32(0)});

        // Init the return value, filling it with the values of the
        // coefficients of the highest-degree monomial in each polynomial.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr and store it in res_arr.
            assert(llvm_depr_GEP_type_check(res_arr, fp_vec_t)); // LCOV_EXCL_LINE
            builder.CreateStore(taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(order), cur_var_idx),
                                builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx));
        });

        // Run the evaluation.
        llvm_loop_u32(
            s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
                    // Load the current poly coeff from diff_arr.
                    // NOTE: we are loading the coefficients backwards wrt the order, hence
                    // we specify order - cur_order.
                    auto *cf = taylor_c_load_diff(s, diff_arr, n_uvars,
                                                  builder.CreateSub(builder.getInt32(order), cur_order), cur_var_idx);

                    // Accumulate in res_arr.
                    assert(llvm_depr_GEP_type_check(res_arr, fp_vec_t)); // LCOV_EXCL_LINE
                    auto *res_ptr = builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx);
                    builder.CreateStore(
                        builder.CreateFAdd(cf, builder.CreateFMul(builder.CreateLoad(fp_vec_t, res_ptr), h)), res_ptr);
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
                res_arr[j] = builder.CreateFAdd(diff_arr[(order - i) * n_eq + j], builder.CreateFMul(res_arr[j], h));
            }
        }

        return res_arr;
    }
}

// Same as taylor_run_multihorner(), but instead of the Horner scheme this implementation uses
// a compensated summation over the naive evaluation of monomials.
std::variant<llvm::Value *, std::vector<llvm::Value *>>
taylor_run_ceval(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_var, llvm::Value *h,
                 std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, bool, bool compact_mode)
{
    auto &builder = s.builder();

    if (compact_mode) {
        // Compact mode.
        auto *diff_arr = std::get<llvm::Value *>(diff_var);

        // Create the arrays storing the results of the evaluation and the running compensations.
        auto *fp_vec_t = pointee_type(diff_arr);
        auto *array_type = llvm::ArrayType::get(fp_vec_t, n_eq);
        auto *res_arr_inst = builder.CreateAlloca(array_type);
        auto *comp_arr_inst = builder.CreateAlloca(array_type);
        // LCOV_EXCL_START
        assert(llvm_depr_GEP_type_check(res_arr_inst, array_type));
        assert(llvm_depr_GEP_type_check(comp_arr_inst, array_type));
        // LCOV_EXCL_STOP
        auto *res_arr = builder.CreateInBoundsGEP(array_type, res_arr_inst, {builder.getInt32(0), builder.getInt32(0)});
        auto *comp_arr
            = builder.CreateInBoundsGEP(array_type, comp_arr_inst, {builder.getInt32(0), builder.getInt32(0)});

        // Init res_arr with the order-0 coefficients, and the running
        // compensations with zero.
        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            // Load the value from diff_arr.
            auto *val = taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), cur_var_idx);

            // Store it in res_arr.
            assert(llvm_depr_GEP_type_check(res_arr, fp_vec_t)); // LCOV_EXCL_LINE
            builder.CreateStore(val, builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx));

            // Zero-init the element in comp_arr.
            assert(llvm_depr_GEP_type_check(comp_arr, fp_vec_t)); // LCOV_EXCL_LINE
            builder.CreateStore(llvm::ConstantFP::get(fp_vec_t, 0.),
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
                auto *cf = taylor_c_load_diff(s, diff_arr, n_uvars, cur_order, cur_var_idx);
                auto *tmp = builder.CreateFMul(cf, cur_h_val);

                // Compute the quantities for the compensation.
                assert(llvm_depr_GEP_type_check(comp_arr, fp_vec_t)); // LCOV_EXCL_LINE
                auto *comp_ptr = builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx);
                assert(llvm_depr_GEP_type_check(res_arr, fp_vec_t)); // LCOV_EXCL_LINE
                auto *res_ptr = builder.CreateInBoundsGEP(fp_vec_t, res_arr, cur_var_idx);
                auto *y = builder.CreateFSub(tmp, builder.CreateLoad(fp_vec_t, comp_ptr));
                auto *cur_res = builder.CreateLoad(fp_vec_t, res_ptr);
                auto *t = builder.CreateFAdd(cur_res, y);

                // Update the compensation and the return value.
                builder.CreateStore(builder.CreateFSub(builder.CreateFSub(t, cur_res), y), comp_ptr);
                builder.CreateStore(t, res_ptr);
            });

            // Update the value of h.
            builder.CreateStore(builder.CreateFMul(cur_h_val, h), cur_h);
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
            comp_arr.push_back(llvm::ConstantFP::get(diff_arr[i]->getType(), 0.));
        }

        // Evaluate and sum.
        auto *cur_h = h;
        for (std::uint32_t i = 1; i <= order; ++i) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                // Evaluate the current monomial.
                auto *tmp = builder.CreateFMul(diff_arr[i * n_eq + j], cur_h);

                // Compute the quantities for the compensation.
                auto *y = builder.CreateFSub(tmp, comp_arr[j]);
                auto *t = builder.CreateFAdd(res_arr[j], y);

                // Update the compensation and the return value.
                comp_arr[j] = builder.CreateFSub(builder.CreateFSub(t, res_arr[j]), y);
                res_arr[j] = t;
            }

            // Update the power of h.
            cur_h = builder.CreateFMul(cur_h, h);
        }

        return res_arr;
    }
}

namespace
{

// NOTE: in compact mode, care must be taken when adding multiple jet functions to the same llvm state
// with the same floating-point type, batch size and number of u variables. The potential issue there
// is that when the first jet is added, the compact mode AD functions are created and then optimised.
// The optimisation pass might alter the functions in a way that makes them incompatible with subsequent
// uses in the second jet (e.g., an argument might be removed from the signature because it is a
// compile-time constant). A workaround to avoid issues is to set the optimisation level to zero
// in the state, add the 2 jets and then run a single optimisation pass.
// NOTE: document this eventually.
template <typename T, typename U>
auto taylor_add_jet_impl(llvm_state &s, const std::string &name, const U &sys, std::uint32_t order,
                         std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                         const std::vector<expression> &sv_funcs, bool parallel_mode)
{
    if (s.is_compiled()) {
        throw std::invalid_argument("A function for the computation of the jet of Taylor derivatives cannot be added "
                                    "to an llvm_state after compilation");
    }

    if (order == 0u) {
        throw std::invalid_argument("The order of a Taylor jet cannot be zero");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size of a Taylor jet cannot be zero");
    }

#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    auto &builder = s.builder();
    auto &context = s.context();

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Record the number of sv_funcs before consuming it.
    const auto n_sv_funcs = boost::numeric_cast<std::uint32_t>(sv_funcs.size());

    // Decompose the system of equations.
    // NOTE: don't use structured bindings due to the
    // usual issues with lambdas.
    const auto td_res = taylor_decompose(sys, sv_funcs);
    const auto &dc = td_res.first;
    const auto &sv_funcs_dc = td_res.second;

    assert(sv_funcs_dc.size() == n_sv_funcs); // LCOV_EXCL_LINE

    // Compute the number of u variables.
    assert(dc.size() > n_eq); // LCOV_EXCL_LINE
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // Prepare the function prototype. The first argument is a float pointer to the in/out array,
    // the second argument a const float pointer to the pars, the third argument
    // a float pointer to the time. These arrays cannot overlap.
    auto *fp_t = to_llvm_type<T>(context);
    std::vector<llvm::Type *> fargs(3, llvm::PointerType::getUnqual(fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument(fmt::format(
            "Unable to create a function for the computation of the jet of Taylor derivatives with name '{}'", name));
    }

    // Set the names/attributes of the function arguments.
    auto *in_out = f->args().begin();
    in_out->setName("in_out");
    in_out->addAttr(llvm::Attribute::NoCapture);
    in_out->addAttr(llvm::Attribute::NoAlias);

    auto *par_ptr = in_out + 1;
    par_ptr->setName("par_ptr");
    par_ptr->addAttr(llvm::Attribute::NoCapture);
    par_ptr->addAttr(llvm::Attribute::NoAlias);
    par_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *time_ptr = par_ptr + 1;
    time_ptr->setName("time_ptr");
    time_ptr->addAttr(llvm::Attribute::NoCapture);
    time_ptr->addAttr(llvm::Attribute::NoAlias);
    time_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Compute the jet of derivatives.
    auto diff_variant = taylor_compute_jet<T>(s, in_out, par_ptr, time_ptr, dc, sv_funcs_dc, n_eq, n_uvars, order,
                                              batch_size, compact_mode, high_accuracy, parallel_mode);

    // Write the derivatives to in_out.
    // NOTE: overflow checking. We need to be able to index into the jet array
    // (size (n_eq + n_sv_funcs) * (order + 1) * batch_size)
    // using uint32_t.
    // LCOV_EXCL_START
    if (order == std::numeric_limits<std::uint32_t>::max()
        || (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || n_eq > std::numeric_limits<std::uint32_t>::max() - n_sv_funcs
        || n_eq + n_sv_funcs > std::numeric_limits<std::uint32_t>::max() / ((order + 1u) * batch_size)) {
        throw std::overflow_error("An overflow condition was detected while adding a Taylor jet");
    }
    // LCOV_EXCL_STOP

    if (compact_mode) {
        auto diff_arr = std::get<llvm::Value *>(diff_variant);

        // Create a global read-only array containing the values in sv_funcs_dc, if any
        // (otherwise, svf_ptr will be null).
        auto svf_ptr = taylor_c_make_sv_funcs_arr(s, sv_funcs_dc);

        // Write the order 0 of the sv_funcs, if needed.
        if (svf_ptr != nullptr) {
            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_sv_funcs), [&](llvm::Value *arr_idx) {
                // Fetch the u var index from svf_ptr.
                assert(llvm_depr_GEP_type_check(svf_ptr, builder.getInt32Ty())); // LCOV_EXCL_LINE
                auto cur_idx = builder.CreateLoad(builder.getInt32Ty(),
                                                  builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                // Load the derivative value from diff_arr.
                auto diff_val = taylor_c_load_diff(s, diff_arr, n_uvars, builder.getInt32(0), cur_idx);

                // Compute the index in the output pointer.
                auto out_idx = builder.CreateMul(builder.CreateAdd(builder.getInt32(n_eq), arr_idx),
                                                 builder.getInt32(batch_size));

                // Store into in_out.
                assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
                store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, in_out, out_idx), diff_val);
            });
        }

        // Write the other orders.
        llvm_loop_u32(
            s, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_idx) {
                    // Load the derivative value from diff_arr.
                    auto diff_val = taylor_c_load_diff(s, diff_arr, n_uvars, cur_order, cur_idx);

                    // Compute the index in the output pointer.
                    auto out_idx = builder.CreateAdd(
                        builder.CreateMul(builder.getInt32((n_eq + n_sv_funcs) * batch_size), cur_order),
                        builder.CreateMul(cur_idx, builder.getInt32(batch_size)));

                    // Store into in_out.
                    assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
                    store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, in_out, out_idx), diff_val);
                });

                if (svf_ptr != nullptr) {
                    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_sv_funcs), [&](llvm::Value *arr_idx) {
                        // Fetch the u var index from svf_ptr.
                        assert(llvm_depr_GEP_type_check(svf_ptr, builder.getInt32Ty())); // LCOV_EXCL_LINE
                        auto cur_idx = builder.CreateLoad(
                            builder.getInt32Ty(), builder.CreateInBoundsGEP(builder.getInt32Ty(), svf_ptr, arr_idx));

                        // Load the derivative value from diff_arr.
                        auto diff_val = taylor_c_load_diff(s, diff_arr, n_uvars, cur_order, cur_idx);

                        // Compute the index in the output pointer.
                        auto out_idx = builder.CreateAdd(
                            builder.CreateMul(builder.getInt32((n_eq + n_sv_funcs) * batch_size), cur_order),
                            builder.CreateMul(builder.CreateAdd(builder.getInt32(n_eq), arr_idx),
                                              builder.getInt32(batch_size)));

                        // Store into in_out.
                        assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
                        store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, in_out, out_idx), diff_val);
                    });
                }
            });
    } else {
        const auto &diff_arr = std::get<std::vector<llvm::Value *>>(diff_variant);

        // Write the order 0 of the sv_funcs.
        for (std::uint32_t j = 0; j < n_sv_funcs; ++j) {
            // Index in the jet of derivatives.
            // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
            // state variables and sv_funcs (not all u vars), hence the indexing is
            // n_eq + j.
            const auto arr_idx = n_eq + j;
            assert(arr_idx < diff_arr.size()); // LCOV_EXCL_LINE
            const auto val = diff_arr[arr_idx];

            // Index in the output array.
            const auto out_idx = (n_eq + j) * batch_size;

            assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
            auto *out_ptr
                = builder.CreateInBoundsGEP(fp_t, in_out, builder.getInt32(static_cast<std::uint32_t>(out_idx)));
            store_vector_to_memory(builder, out_ptr, val);
        }

        for (decltype(diff_arr.size()) cur_order = 1; cur_order <= order; ++cur_order) {
            for (std::uint32_t j = 0; j < n_eq; ++j) {
                // Index in the jet of derivatives.
                // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
                // state variables and sv_funcs (not all u vars), hence the indexing is
                // cur_order * (n_eq + n_sv_funcs) + j.
                const auto arr_idx = cur_order * (n_eq + n_sv_funcs) + j;
                assert(arr_idx < diff_arr.size()); // LCOV_EXCL_LINE
                const auto val = diff_arr[arr_idx];

                // Index in the output array.
                const auto out_idx = (n_eq + n_sv_funcs) * batch_size * cur_order + j * batch_size;

                assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
                auto *out_ptr
                    = builder.CreateInBoundsGEP(fp_t, in_out, builder.getInt32(static_cast<std::uint32_t>(out_idx)));
                store_vector_to_memory(builder, out_ptr, val);
            }

            for (std::uint32_t j = 0; j < n_sv_funcs; ++j) {
                const auto arr_idx = cur_order * (n_eq + n_sv_funcs) + n_eq + j;
                assert(arr_idx < diff_arr.size()); // LCOV_EXCL_LINE
                const auto val = diff_arr[arr_idx];

                const auto out_idx = (n_eq + n_sv_funcs) * batch_size * cur_order + (n_eq + j) * batch_size;

                assert(llvm_depr_GEP_type_check(in_out, fp_t)); // LCOV_EXCL_LINE
                auto *out_ptr
                    = builder.CreateInBoundsGEP(fp_t, in_out, builder.getInt32(static_cast<std::uint32_t>(out_idx)));
                store_vector_to_memory(builder, out_ptr, val);
            }
        }
    }

    // Finish off the function.
    builder.CreateRetVoid();

    // Verify it.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // Run the optimisation pass.
    s.optimise();

    return dc;
}

} // namespace

} // namespace detail

template <typename T>
taylor_dc_t taylor_add_jet(llvm_state &s, const std::string &name, const std::vector<expression> &sys,
                           std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                           const std::vector<expression> &sv_funcs, bool parallel_mode)
{
    return detail::taylor_add_jet_impl<T>(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs,
                                          parallel_mode);
}

template <typename T>
taylor_dc_t taylor_add_jet(llvm_state &s, const std::string &name,
                           const std::vector<std::pair<expression, expression>> &sys, std::uint32_t order,
                           std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                           const std::vector<expression> &sv_funcs, bool parallel_mode)
{
    return detail::taylor_add_jet_impl<T>(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs,
                                          parallel_mode);
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet<double>(llvm_state &, const std::string &,
                                                              const std::vector<expression> &, std::uint32_t,
                                                              std::uint32_t, bool, bool,
                                                              const std::vector<expression> &, bool);

template HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet<double>(llvm_state &, const std::string &,
                                                              const std::vector<std::pair<expression, expression>> &,
                                                              std::uint32_t, std::uint32_t, bool, bool,
                                                              const std::vector<expression> &, bool);

template HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet<long double>(llvm_state &, const std::string &,
                                                                   const std::vector<expression> &, std::uint32_t,
                                                                   std::uint32_t, bool, bool,
                                                                   const std::vector<expression> &, bool);

template HEYOKA_DLL_PUBLIC taylor_dc_t
taylor_add_jet<long double>(llvm_state &, const std::string &, const std::vector<std::pair<expression, expression>> &,
                            std::uint32_t, std::uint32_t, bool, bool, const std::vector<expression> &, bool);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC taylor_dc_t taylor_add_jet<mppp::real128>(llvm_state &, const std::string &,
                                                                     const std::vector<expression> &, std::uint32_t,
                                                                     std::uint32_t, bool, bool,
                                                                     const std::vector<expression> &, bool);

template HEYOKA_DLL_PUBLIC taylor_dc_t
taylor_add_jet<mppp::real128>(llvm_state &, const std::string &, const std::vector<std::pair<expression, expression>> &,
                              std::uint32_t, std::uint32_t, bool, bool, const std::vector<expression> &, bool);

#endif

} // namespace heyoka

#if defined(__GNUC__) && (__GNUC__ >= 11)

#pragma GCC diagnostic pop

#endif
