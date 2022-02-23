// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
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
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

namespace
{

// RAII helper to temporarily set the opt level to 0 in an llvm_state.
struct opt_disabler {
    llvm_state &m_s;
    unsigned m_orig_opt_level;

    explicit opt_disabler(llvm_state &s) : m_s(s), m_orig_opt_level(s.opt_level())
    {
        // Disable optimisations.
        m_s.opt_level() = 0;
    }

    opt_disabler(const opt_disabler &) = delete;
    opt_disabler(opt_disabler &&) noexcept = delete;
    opt_disabler &operator=(const opt_disabler &) = delete;
    opt_disabler &operator=(opt_disabler &&) noexcept = delete;

    ~opt_disabler()
    {
        // Restore the original optimisation level.
        m_s.opt_level() = m_orig_opt_level;
    }
};

// Helper to determine the optimal Taylor order for a given tolerance,
// following Jorba's prescription.
template <typename T>
std::uint32_t taylor_order_from_tol(T tol)
{
    using std::ceil;
    using std::isfinite;
    using std::log;

    // Determine the order from the tolerance.
    auto order_f = ceil(-log(tol) / 2 + 1);
    // LCOV_EXCL_START
    if (!isfinite(order_f)) {
        throw std::invalid_argument(
            "The computation of the Taylor order in an adaptive Taylor stepper produced a non-finite value");
    }
    // LCOV_EXCL_STOP
    // NOTE: min order is 2.
    order_f = std::max(T(2), order_f);

    // NOTE: cast to double as that ensures that the
    // max of std::uint32_t is exactly representable.
    // LCOV_EXCL_START
    if (order_f > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("The computation of the Taylor order in an adaptive Taylor stepper resulted "
                                  "in an overflow condition");
    }
    // LCOV_EXCL_STOP
    return static_cast<std::uint32_t>(order_f);
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

// Small helper to compute the size of a global array.
std::uint32_t taylor_c_gl_arr_size(llvm::Value *v)
{
    assert(llvm::isa<llvm::GlobalVariable>(v));

    return boost::numeric_cast<std::uint32_t>(
        llvm::cast<llvm::ArrayType>(llvm::cast<llvm::PointerType>(v->getType())->getElementType())->getNumElements());
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
    auto &module = s.module();

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
                    assert(false);
                }
            },
            dc[i].first.value());
    }

    // Flag to signal that the time derivatives of all state variables are u variables.
    assert(dc.size() >= n_uvars);
    const auto all_der_vars = (n_der_vars == (dc.size() - n_uvars));

    assert(var_indices.size() == vars.size());
    assert(num_indices.size() == nums.size());
    assert(par_indices.size() == pars.size());

    // Turn the vectors into global read-only LLVM arrays.

    // Variables.
    auto *var_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(var_indices.size()));

    auto *var_indices_arr = llvm::ConstantArray::get(var_arr_type, var_indices);
    auto *g_var_indices = new llvm::GlobalVariable(module, var_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, var_indices_arr);

    auto *vars_arr = llvm::ConstantArray::get(var_arr_type, vars);
    auto *g_vars
        = new llvm::GlobalVariable(module, vars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, vars_arr);

    // Numbers.
    auto *num_indices_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(num_indices.size()));
    auto *num_indices_arr = llvm::ConstantArray::get(num_indices_arr_type, num_indices);
    auto *g_num_indices = new llvm::GlobalVariable(module, num_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, num_indices_arr);

    auto nums_arr_type
        = llvm::ArrayType::get(to_llvm_type<T>(context), boost::numeric_cast<std::uint64_t>(nums.size()));
    auto nums_arr = llvm::ConstantArray::get(nums_arr_type, nums);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *g_nums
        = new llvm::GlobalVariable(module, nums_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, nums_arr);

    // Params.
    auto *par_arr_type
        = llvm::ArrayType::get(llvm::Type::getInt32Ty(context), boost::numeric_cast<std::uint64_t>(par_indices.size()));

    auto *par_indices_arr = llvm::ConstantArray::get(par_arr_type, par_indices);
    auto *g_par_indices = new llvm::GlobalVariable(module, par_indices_arr->getType(), true,
                                                   llvm::GlobalVariable::InternalLinkage, par_indices_arr);

    auto *pars_arr = llvm::ConstantArray::get(par_arr_type, pars);
    auto *g_pars
        = new llvm::GlobalVariable(module, pars_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, pars_arr);

    return std::pair{std::array{g_var_indices, g_vars, g_num_indices, g_nums, g_par_indices, g_pars}, all_der_vars};
}

// Helper to compute and store the derivatives of the state variables in compact mode at order 'order'.
// svd_gl is the return value of taylor_c_make_sv_diff_globals(), which contains
// the indices/constants necessary for the computation.
void taylor_c_compute_sv_diffs(llvm_state &s, const std::pair<std::array<llvm::GlobalVariable *, 6>, bool> &svd_gl,
                               llvm::Value *diff_arr, llvm::Value *par_ptr, std::uint32_t n_uvars, llvm::Value *order,
                               std::uint32_t batch_size)
{
    assert(batch_size > 0u);

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

// Helper to convert the arguments of the definition of a u variable
// into a vector of variants. u variables will be converted to their indices,
// numbers will be unchanged, parameters will be converted to their indices.
// The hidden deps will also be converted to indices.
auto taylor_udef_to_variants(const expression &ex, const std::vector<std::uint32_t> &deps)
{
    return std::visit(
        [&deps](const auto &v) -> std::vector<std::variant<std::uint32_t, number>> {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, func>) {
                std::vector<std::variant<std::uint32_t, number>> retval;

                for (const auto &arg : v.args()) {
                    std::visit(
                        [&retval](const auto &x) {
                            using tp = detail::uncvref_t<decltype(x)>;

                            if constexpr (std::is_same_v<tp, variable>) {
                                retval.emplace_back(uname_to_index(x.name()));
                            } else if constexpr (std::is_same_v<tp, number>) {
                                retval.emplace_back(x);
                            } else if constexpr (std::is_same_v<tp, param>) {
                                retval.emplace_back(x.idx());
                            } else {
                                throw std::invalid_argument(
                                    "Invalid argument encountered in an element of a Taylor decomposition: the "
                                    "argument is not a variable or a number");
                            }
                        },
                        arg.value());
                }

                // Handle the hidden deps.
                for (auto idx : deps) {
                    retval.emplace_back(idx);
                }

                return retval;
            } else {
                throw std::invalid_argument("Invalid expression encountered in a Taylor decomposition: the "
                                            "expression is not a function");
            }
        },
        ex.value());
}

// Helper to convert a vector of variants into a variant of vectors.
// All elements of v must be of the same type, and v cannot be empty.
template <typename... T>
auto taylor_c_vv_transpose(const std::vector<std::variant<T...>> &v)
{
    assert(!v.empty());

    // Init the return value based on the type
    // of the first element of v.
    auto retval = std::visit(
        [size = v.size()](const auto &x) {
            using type = detail::uncvref_t<decltype(x)>;

            std::vector<type> tmp;
            tmp.reserve(boost::numeric_cast<decltype(tmp.size())>(size));
            tmp.push_back(x);

            return std::variant<std::vector<T>...>(std::move(tmp));
        },
        v[0]);

    // Append the other values from v.
    for (decltype(v.size()) i = 1; i < v.size(); ++i) {
        std::visit(
            [&retval](const auto &x) {
                std::visit(
                    [&x](auto &vv) {
                        // The value type of retval.
                        using scal_t = typename detail::uncvref_t<decltype(vv)>::value_type;

                        // The type of the current element of v.
                        using x_t = detail::uncvref_t<decltype(x)>;

                        if constexpr (std::is_same_v<scal_t, x_t>) {
                            vv.push_back(x);
                        } else {
                            throw std::invalid_argument(
                                "Inconsistent types detected while building the transposed sets of "
                                "arguments for the Taylor derivative functions in compact mode");
                        }
                    },
                    retval);
            },
            v[i]);
    }

    return retval;
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
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vidx(llvm_state &s,
                                                                       const std::vector<std::uint32_t> &ind)
{
    assert(!ind.empty());

    auto &builder = s.builder();

    // Check if all indices in ind are the same.
    if (std::all_of(ind.begin() + 1, ind.end(), [&ind](const auto &n) { return n == ind[0]; })) {
        // If all indices are the same, don't construct an array, just always return
        // the same value.
        return [num = builder.getInt32(ind[0])](llvm::Value *) -> llvm::Value * { return num; };
    }

    // If ind consists of consecutive indices, we can replace
    // the index array with a simple offset computation.
    if (is_consecutive(ind)) {
        return [&builder, start_idx = builder.getInt32(ind[0])](llvm::Value *cur_call_idx) -> llvm::Value * {
            return builder.CreateAdd(start_idx, cur_call_idx);
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
                assert(checker == ind);
#endif

                return [&builder, start_idx = builder.getInt32(rep_indices[0]),
                        n_reps = builder.getInt32(boost::numeric_cast<std::uint32_t>(n_reps))](
                           llvm::Value *cur_call_idx) -> llvm::Value * {
                    return builder.CreateAdd(start_idx, builder.CreateUDiv(cur_call_idx, n_reps));
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
    assert(arr_type != nullptr);

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr);
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar
        = new llvm::GlobalVariable(md, const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage, const_arr);

    // Return the generator.
    return [gvar, arr_type, &builder](llvm::Value *cur_call_idx) -> llvm::Value * {
        assert(llvm_depr_GEP_type_check(gvar, arr_type)); // LCOV_EXCL_LINE
        return builder.CreateLoad(builder.getInt32Ty(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

template <typename T>
std::function<llvm::Value *(llvm::Value *)> taylor_c_make_arg_gen_vc(llvm_state &s, const std::vector<number> &vc)
{
    assert(!vc.empty());

    // Check if all the numbers are the same.
    // NOTE: the comparison operator of number will consider two numbers of different
    // type but equal value to be equal.
    if (std::all_of(vc.begin() + 1, vc.end(), [&vc](const auto &n) { return n == vc[0]; })) {
        // If all constants are the same, don't construct an array, just always return
        // the same value.
        return [num = codegen<T>(s, vc[0])](llvm::Value *) -> llvm::Value * { return num; };
    }

    // Generate the array of constants as llvm constants.
    std::vector<llvm::Constant *> tmp_c_vec;
    tmp_c_vec.reserve(vc.size());
    for (const auto &val : vc) {
        tmp_c_vec.push_back(llvm::cast<llvm::Constant>(codegen<T>(s, val)));
    }

    // Create the array type.
    auto *arr_type = llvm::ArrayType::get(tmp_c_vec[0]->getType(), boost::numeric_cast<std::uint64_t>(vc.size()));
    assert(arr_type != nullptr);

    // Create the constant array as a global read-only variable.
    auto *const_arr = llvm::ConstantArray::get(arr_type, tmp_c_vec);
    assert(const_arr != nullptr);
    // NOTE: naked new here is fine, gvar will be registered in the module
    // object and cleaned up when the module is destroyed.
    auto *gvar = new llvm::GlobalVariable(s.module(), const_arr->getType(), true, llvm::GlobalVariable::InternalLinkage,
                                          const_arr);

    // Return the generator.
    return [gvar, arr_type, &s](llvm::Value *cur_call_idx) -> llvm::Value * {
        auto &builder = s.builder();

        assert(llvm_depr_GEP_type_check(gvar, arr_type)); // LCOV_EXCL_LINE
        return builder.CreateLoad(arr_type->getArrayElementType(),
                                  builder.CreateInBoundsGEP(arr_type, gvar, {builder.getInt32(0), cur_call_idx}));
    };
}

// Comparision operator for LLVM functions based on their names.
struct llvm_func_name_compare {
    bool operator()(const llvm::Function *f0, const llvm::Function *f1) const
    {
        return f0->getName() < f1->getName();
    }
};

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
            auto func = taylor_c_diff_func<T>(s, ex.first, n_uvars, batch_size, high_accuracy);

            // Insert the function into tmp_map.
            const auto [it, is_new_func] = tmp_map.try_emplace(func);

            assert(is_new_func || !it->second.empty());

            // Convert the variables/constants in the current dc
            // element into a set of indices/constants.
            const auto cdiff_args = taylor_udef_to_variants(ex.first, ex.second);

            if (!is_new_func && it->second.back().size() - 1u != cdiff_args.size()) {
                throw std::invalid_argument(
                    "Inconsistent arity detected in a Taylor derivative function in compact "
                    "mode: the same function is being called with both {} and {} arguments"_format(
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
            assert(!vv.empty());

            // Add the function.
            const auto [it, ins_status] = tmp_map_transpose.try_emplace(func);
            assert(ins_status);

            const auto n_calls = vv.size();
            const auto n_args = vv[0].size();
            // NOTE: n_args must be at least 1 because the u idx
            // is prepended to the actual function arguments in
            // the tmp_map entries.
            assert(n_args >= 1u);

            for (decltype(vv[0].size()) i = 0; i < n_args; ++i) {
                // Build the vector of values corresponding
                // to the current argument index.
                std::vector<std::variant<std::uint32_t, number>> tmp_c_vec;
                for (decltype(vv.size()) j = 0; j < n_calls; ++j) {
                    tmp_c_vec.push_back(vv[j][i]);
                }

                // Turn tmp_c_vec (a vector of variants) into a variant
                // of vectors, and insert the result.
                it->second.push_back(taylor_c_vv_transpose(tmp_c_vec));
            }
        }

        // Add a new entry in retval for the current segment.
        retval.emplace_back();
        auto &a_map = retval.back();

        for (const auto &[func, vv] : tmp_map_transpose) {
            assert(!vv.empty());

            // Add the function.
            const auto [it, ins_status] = a_map.try_emplace(func);
            assert(ins_status);

            // Set the number of calls for this function.
            it->second.first
                = std::visit([](const auto &x) { return boost::numeric_cast<std::uint32_t>(x.size()); }, vv[0]);
            assert(it->second.first > 0u);

            // Create the g functions for each argument.
            for (const auto &v : vv) {
                it->second.second.push_back(std::visit(
                    [&s](const auto &x) {
                        using type = detail::uncvref_t<decltype(x)>;

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

    return retval;
}

// Helper for the computation of a jet of derivatives in compact mode,
// used in taylor_compute_jet() below.
template <typename T>
llvm::Value *taylor_compute_jet_compact_mode(llvm_state &s, llvm::Value *order0, llvm::Value *par_ptr,
                                             llvm::Value *time_ptr, const taylor_dc_t &dc,
                                             const std::vector<std::uint32_t> &sv_funcs_dc, std::uint32_t n_eq,
                                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size,
                                             bool high_accuracy)
{
    auto &builder = s.builder();

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
    const auto max_svf_idx
        = sv_funcs_dc.empty() ? std::uint32_t(0) : *std::max_element(sv_funcs_dc.begin(), sv_funcs_dc.end());

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
    auto *fp_type = llvm::cast<llvm::PointerType>(order0->getType())->getElementType();
    auto *fp_vec_type = make_vector_type(fp_type, batch_size);
    auto *array_type
        = llvm::ArrayType::get(fp_vec_type, (max_svf_idx < n_eq) ? (n_uvars * order + n_eq) : (n_uvars * (order + 1u)));

    // Make the global array and fetch a pointer to its first element.
    // NOTE: we use a global array rather than a local one here because
    // its size can grow quite large, which can lead to stack overflow issues.
    // This has of course consequences in terms of thread safety, which
    // we will have to document.
    auto diff_arr_gvar = make_global_zero_array(s.module(), array_type);
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

    // Helper to compute and store the derivatives of order cur_order
    // of the u variables which are not state variables.
    auto compute_u_diffs = [&](llvm::Value *cur_order) {
        for (const auto &map : f_maps) {
            for (const auto &p : map) {
                // The LLVM function for the computation of the
                // derivative in compact mode.
                const auto &func = p.first;

                // The number of func calls.
                const auto ncalls = p.second.first;

                // The generators for the arguments of func.
                const auto &gens = p.second.second;

                assert(ncalls > 0u);
                assert(!gens.empty());
                assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));

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

    // Monitor the starting index of the current
    // segment while iterating on f_maps.
    auto cur_start_u_idx = n_eq;

    // NOTE: this is a slight repetition of compute_u_diffs() with minor modifications.
    for (const auto &map : f_maps) {
        if (cur_start_u_idx > max_svf_idx) {
            // We computed all the necessary derivatives, break out.
            // NOTE: if we did not have sv_funcs to begin with,
            // max_svf_idx is zero and we exit at the first iteration
            // without doing anything. If all sv funcs alias state variables,
            // then max_svf_idx < n_eq and thus we also exit immediately
            // at the first iteration.
            break;
        }

        for (const auto &p : map) {
            const auto &func = p.first;
            const auto ncalls = p.second.first;
            const auto &gens = p.second.second;

            assert(ncalls > 0u);
            assert(!gens.empty());
            assert(std::all_of(gens.begin(), gens.end(), [](const auto &f) { return static_cast<bool>(f); }));

            llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(ncalls), [&](llvm::Value *cur_call_idx) {
                auto u_idx = gens[0](cur_call_idx);

                std::vector<llvm::Value *> args{builder.getInt32(order), u_idx, diff_arr, par_ptr, time_ptr};

                for (decltype(gens.size()) i = 1; i < gens.size(); ++i) {
                    args.push_back(gens[i](cur_call_idx));
                }

                taylor_c_store_diff(s, diff_arr, n_uvars, builder.getInt32(order), u_idx,
                                    builder.CreateCall(func, args));
            });

            // Update cur_start_u_idx taking advantage of the fact
            // that each block in a segment processes the derivatives
            // of exactly ncalls u variables.
            cur_start_u_idx += ncalls;
        }
    }

    get_logger()->trace("Taylor IR creation compact mode runtime: {}", sw);

    // Return the array of derivatives of the u variables.
    return diff_arr;
}

// Given an input pointer 'in', load the first n * batch_size values in it as n vectors
// with size batch_size. If batch_size is 1, the values will be loaded as scalars.
auto taylor_load_values(llvm_state &s, llvm::Value *in, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

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

// Small helper to deduce the number of parameters
// present in the Taylor decomposition of an ODE system.
// NOTE: this will also include the functions of state variables,
// as they are part of the decomposition.
// NOTE: the first few entries in the decomposition are the mapping
// u variables -> state variables. These never contain any param
// by construction.
template <typename T>
std::uint32_t n_pars_in_dc(const T &dc)
{
    std::uint32_t retval = 0;

    for (const auto &p : dc) {
        retval = std::max(retval, get_param_size(p.first));
    }

    return retval;
}

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
                   bool high_accuracy)
{
    assert(batch_size > 0u);
    assert(n_eq > 0u);
    assert(order > 0u);

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
        // In compact mode, let's ensure that we can index into par_ptr using std::uint32_t.
        // NOTE: in default mode the check is done inside taylor_codegen_numparam_par().

        // Deduce the size of the param array from the expressions in the decomposition.
        const auto param_size = n_pars_in_dc(dc);
        // LCOV_EXCL_START
        if (param_size > std::numeric_limits<std::uint32_t>::max() / batch_size) {
            throw std::overflow_error(
                "An overflow condition was detected in the computation of a jet of Taylor derivatives in compact mode");
        }
        // LCOV_EXCL_STOP

        return taylor_compute_jet_compact_mode<T>(s, order0, par_ptr, time_ptr, dc, sv_funcs_dc, n_eq, n_uvars, order,
                                                  batch_size, high_accuracy);
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
        const auto max_svf_idx
            = sv_funcs_dc.empty() ? std::uint32_t(0) : *std::max_element(sv_funcs_dc.begin(), sv_funcs_dc.end());

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

// Helper to generate the LLVM code to store the Taylor coefficients of the state variables and
// the sv funcs into an external array. The Taylor polynomials are stored in row-major order,
// first the state variables and after the sv funcs. For use in the adaptive timestepper implementations.
void taylor_write_tc(llvm_state &s, const std::variant<llvm::Value *, std::vector<llvm::Value *>> &diff_variant,
                     const std::vector<std::uint32_t> &sv_funcs_dc, llvm::Value *svf_ptr, llvm::Value *tc_ptr,
                     std::uint32_t n_eq, std::uint32_t n_uvars, std::uint32_t order, std::uint32_t batch_size)
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

        for (std::uint32_t j = 0; j < n_eq; ++j) {
            for (decltype(diff_arr.size()) cur_order = 0; cur_order <= order; ++cur_order) {
                // Index in the jet of derivatives.
                // NOTE: in non-compact mode, diff_arr contains the derivatives only of the
                // state variables and sv_variable (not all u vars), hence the indexing
                // is cur_order * (n_eq + n_sv_funcs) + j.
                const auto arr_idx = cur_order * (n_eq + n_sv_funcs) + j;
                assert(arr_idx < diff_arr.size());
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

        for (std::uint32_t j = 0; j < n_sv_funcs; ++j) {
            for (decltype(diff_arr.size()) cur_order = 0; cur_order <= order; ++cur_order) {
                // Index in the jet of derivatives.
                const auto arr_idx = cur_order * (n_eq + n_sv_funcs) + n_eq + j;
                assert(arr_idx < diff_arr.size());
                auto *const val = diff_arr[arr_idx];

                // Index in tc_ptr.
                const auto out_idx = (order + 1u) * batch_size * (n_eq + j) + cur_order * batch_size;

                // Write to tc_ptr.
                assert(llvm_depr_GEP_type_check(tc_ptr, pointee_type(tc_ptr))); // LCOV_EXCL_LINE
                auto *out_ptr = builder.CreateInBoundsGEP(pointee_type(tc_ptr), tc_ptr,
                                                          builder.getInt32(static_cast<std::uint32_t>(out_idx)));
                store_vector_to_memory(builder, out_ptr, val);
            }
        }
    }
}

// Add to s an adaptive timestepper function with support for events. This timestepper will *not*
// propagate the state of the system. Instead, its output will be the jet of derivatives
// of all state variables and event equations, and the deduced timestep value(s).
template <typename T, typename U>
auto taylor_add_adaptive_step_with_events(llvm_state &s, const std::string &name, const U &sys, T tol,
                                          std::uint32_t batch_size, bool, bool compact_mode,
                                          const std::vector<expression> &evs, bool high_accuracy)
{
    using std::isfinite;

    assert(!s.is_compiled());
    assert(batch_size != 0u);
    assert(isfinite(tol) && tol > 0);

    // Determine the order from the tolerance.
    const auto order = taylor_order_from_tol(tol);

    // Record the number of equations/variables.
    const auto n_eq = boost::numeric_cast<std::uint32_t>(sys.size());

    // Decompose the system of equations.
    auto [dc, ev_dc] = taylor_decompose(sys, evs);

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = boost::numeric_cast<std::uint32_t>(dc.size() - n_eq);

    auto &builder = s.builder();
    auto &context = s.context();

    // Prepare the function prototype. The arguments are:
    // - pointer to the output jet of derivative (write only),
    // - pointer to the current state vector (read only),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the max_abs_state output variable (write only).
    // These pointers cannot overlap.
    std::vector<llvm::Type *> fargs(6, llvm::PointerType::getUnqual(to_llvm_type<T>(context)));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument(
            "Unable to create a function for an adaptive Taylor stepper with name '{}'"_format(name));
    }
    // LCOV_EXCL_STOP

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
    auto diff_variant = taylor_compute_jet<T>(s, state_ptr, par_ptr, time_ptr, dc, ev_dc, n_eq, n_uvars, order,
                                              batch_size, compact_mode, high_accuracy);

    // Determine the integration timestep.
    auto h = taylor_determine_h<T>(s, diff_variant, ev_dc, svf_ptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                   max_abs_state_ptr);

    // Store h to memory.
    store_vector_to_memory(builder, h_ptr, h);

    // Copy the jet of derivatives to jet_ptr.
    taylor_write_tc(s, diff_variant, ev_dc, svf_ptr, jet_ptr, n_eq, n_uvars, order, batch_size);

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    // NOTE: this does nothing currently, as the optimisation
    // level is set to zero from the outside.
    s.optimise();

    return std::tuple{std::move(dc), order};
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

// NOTE: in compact mode, care must be taken when adding multiple stepper functions to the same llvm state
// with the same floating-point type, batch size and number of u variables. The potential issue there
// is that when the first stepper is added, the compact mode AD functions are created and then optimised.
// The optimisation pass might alter the functions in a way that makes them incompatible with subsequent
// uses in the second stepper (e.g., an argument might be removed from the signature because it is a
// compile-time constant). A workaround to avoid issues is to set the optimisation level to zero
// in the state, add the 2 steppers and then run a single optimisation pass. This is what we do
// in the integrators' ctors.
// NOTE: document this eventually.
template <typename T, typename U>
auto taylor_add_adaptive_step(llvm_state &s, const std::string &name, const U &sys, T tol, std::uint32_t batch_size,
                              bool high_accuracy, bool compact_mode)
{
    using std::isfinite;

    assert(!s.is_compiled());
    assert(batch_size > 0u);
    assert(isfinite(tol) && tol > 0);

    // Determine the order from the tolerance.
    const auto order = taylor_order_from_tol(tol);

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

    // Prepare the function prototype. The arguments are:
    // - pointer to the current state vector (read & write),
    // - pointer to the parameters (read only),
    // - pointer to the time value(s) (read only),
    // - pointer to the array of max timesteps (read & write),
    // - pointer to the Taylor coefficients output (write only).
    // These pointers cannot overlap.
    auto *fp_t = to_llvm_type<T>(context);
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);
    std::vector<llvm::Type *> fargs(5, llvm::PointerType::getUnqual(fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument(
            "Unable to create a function for an adaptive Taylor stepper with name '{}'"_format(name));
    }

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
    auto diff_variant = taylor_compute_jet<T>(s, state_ptr, par_ptr, time_ptr, dc, {}, n_eq, n_uvars, order, batch_size,
                                              compact_mode, high_accuracy);

    // Determine the integration timestep.
    auto h = taylor_determine_h<T>(s, diff_variant, sv_funcs_dc, nullptr, h_ptr, n_eq, n_uvars, order, batch_size,
                                   nullptr);

    // Evaluate the Taylor polynomials, producing the updated state of the system.
    auto new_state_var
        = high_accuracy ? taylor_run_ceval(s, diff_variant, h, n_eq, n_uvars, order, high_accuracy, compact_mode)
                        : taylor_run_multihorner(s, diff_variant, h, n_eq, n_uvars, order, batch_size, compact_mode);

    // Store the new state.
    // NOTE: no need to perform overflow check on n_eq * batch_size,
    // as in taylor_compute_jet() we already checked.
    if (compact_mode) {
        auto new_state = std::get<llvm::Value *>(new_state_var);

        llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n_eq), [&](llvm::Value *cur_var_idx) {
            assert(llvm_depr_GEP_type_check(new_state, fp_vec_t)); // LCOV_EXCL_LINE
            auto val = builder.CreateLoad(fp_vec_t, builder.CreateInBoundsGEP(fp_vec_t, new_state, cur_var_idx));
            assert(llvm_depr_GEP_type_check(state_ptr, fp_t)); // LCOV_EXCL_LINE
            store_vector_to_memory(builder,
                                   builder.CreateInBoundsGEP(
                                       fp_t, state_ptr, builder.CreateMul(cur_var_idx, builder.getInt32(batch_size))),
                                   val);
        });
    } else {
        const auto &new_state = std::get<std::vector<llvm::Value *>>(new_state_var);

        assert(new_state.size() == n_eq);

        for (std::uint32_t var_idx = 0; var_idx < n_eq; ++var_idx) {
            assert(llvm_depr_GEP_type_check(state_ptr, fp_t)); // LCOV_EXCL_LINE
            store_vector_to_memory(builder,
                                   builder.CreateInBoundsGEP(fp_t, state_ptr, builder.getInt32(var_idx * batch_size)),
                                   new_state[var_idx]);
        }
    }

    // Store the timesteps that were used.
    store_vector_to_memory(builder, h_ptr, h);

    // Write the Taylor coefficients, if requested.
    auto nptr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(to_llvm_type<T>(s.context())));
    llvm_if_then_else(
        s, builder.CreateICmpNE(tc_ptr, nptr),
        [&]() {
            // tc_ptr is not null: copy the Taylor coefficients
            // for the state variables.
            taylor_write_tc(s, diff_variant, {}, nullptr, tc_ptr, n_eq, n_uvars, order, batch_size);
        },
        [&]() {
            // Taylor coefficients were not requested,
            // don't do anything in this branch.
        });

    // Create the return value.
    builder.CreateRetVoid();

    // Verify the function.
    s.verify_function(f);

    // Run the optimisation pass.
    // NOTE: this does nothing currently, as the optimisation
    // level is set to zero from the outside.
    s.optimise();

    return std::tuple{std::move(dc), order};
}

} // namespace

template <typename T>
template <typename U>
void taylor_adaptive_impl<T>::finalise_ctor_impl(const U &sys, std::vector<T> state, T time, T tol, bool high_accuracy,
                                                 bool compact_mode, std::vector<T> pars, std::vector<t_event_t> tes,
                                                 std::vector<nt_event_t> ntes)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    using std::isfinite;

    // Assign the data members.
    m_state = std::move(state);
    m_time = dfloat<T>(time);
    m_pars = std::move(pars);
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check input params.
    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() != sys.size()) {
        throw std::invalid_argument(
            "Inconsistent sizes detected in the initialization of an adaptive Taylor "
            "integrator: the state vector has a dimension of {}, while the number of equations is {}"_format(
                m_state.size(), sys.size()));
    }

    if (!isfinite(m_time)) {
        throw std::invalid_argument(
            "Cannot initialise an adaptive Taylor integrator with a non-finite initial time of {}"_format(
                static_cast<T>(m_time)));
    }

    if (!isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead"_format(
                tol));
    }

    // Store the tolerance.
    m_tol = tol;

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Temporarily disable optimisations in s, so that
    // we don't optimise twice when adding the step
    // and then the d_out.
    std::optional<opt_disabler> od(m_llvm);

    // Add the stepper function.
    if (with_events) {
        std::vector<expression> ee;
        // NOTE: no need for deep copies of the expressions: ee is never mutated
        // and we will be deep-copying it anyway when we do the decomposition.
        for (const auto &ev : tes) {
            ee.push_back(ev.get_expression());
        }
        for (const auto &ev : ntes) {
            ee.push_back(ev.get_expression());
        }

        std::tie(m_dc, m_order) = taylor_add_adaptive_step_with_events<T>(m_llvm, "step_e", sys, tol, 1, high_accuracy,
                                                                          compact_mode, ee, high_accuracy);
    } else {
        std::tie(m_dc, m_order) = taylor_add_adaptive_step<T>(m_llvm, "step", sys, tol, 1, high_accuracy, compact_mode);
    }

    // Fix m_pars' size, if necessary.
    const auto npars = n_pars_in_dc(m_dc);
    if (m_pars.size() < npars) {
        m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(npars));
    } else if (m_pars.size() > npars) {
        throw std::invalid_argument(
            "Excessive number of parameter values passed to the constructor of an adaptive "
            "Taylor integrator: {} parameter values were passed, but the ODE system contains only {} parameters"_format(
                m_pars.size(), npars));
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of
    // the dense output.
    taylor_add_d_out_function<T>(m_llvm, m_dim, m_order, 1, high_accuracy);

    get_logger()->trace("Taylor dense output runtime: {}", sw);
    sw.reset();

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass manually.
    m_llvm.optimise();

    get_logger()->trace("Taylor global opt pass runtime: {}", sw);
    sw.reset();

    // Run the jit.
    m_llvm.compile();

    get_logger()->trace("Taylor LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    if (with_events) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }

    // Fetch the function to compute the dense output.
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));

    // Setup the vector for the Taylor coefficients.
    // LCOV_EXCL_START
    if (m_order == std::numeric_limits<std::uint32_t>::max()
        || m_state.size() > std::numeric_limits<decltype(m_tc.size())>::max() / (m_order + 1u)) {
        throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator: the order "
                                  "or the state size is too large");
    }
    // LCOV_EXCL_STOP

    m_tc.resize(m_state.size() * (m_order + 1u));

    // Setup the vector for the dense output.
    m_d_out.resize(m_state.size());

    // Init the event data structure if needed.
    // NOTE: this can be done in parallel with the rest of the constructor,
    // once we have m_order/m_dim and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<ed_data>(std::move(tes), std::move(ntes), m_order, m_dim);
    }
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl()
    : taylor_adaptive_impl({prime("x"_var) = 0_dbl}, {T(0)}, kw::tol = T(1e-1))
{
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(const taylor_adaptive_impl &other)
    : m_state(other.m_state), m_time(other.m_time), m_llvm(other.m_llvm), m_dim(other.m_dim), m_order(other.m_order),
      m_tol(other.m_tol), m_high_accuracy(other.m_high_accuracy), m_compact_mode(other.m_compact_mode),
      m_pars(other.m_pars), m_tc(other.m_tc), m_last_h(other.m_last_h), m_d_out(other.m_d_out),
      m_ed_data(other.m_ed_data ? std::make_unique<ed_data>(*other.m_ed_data) : nullptr)
{
    // NOTE: make explicit deep copy of the decomposition.
    m_dc.reserve(other.m_dc.size());
    for (const auto &[ex, deps] : other.m_dc) {
        m_dc.emplace_back(copy(ex), deps);
    }

    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T> &taylor_adaptive_impl<T>::operator=(const taylor_adaptive_impl &other)
{
    if (this != &other) {
        *this = taylor_adaptive_impl(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive_impl<T> &taylor_adaptive_impl<T>::operator=(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T>::~taylor_adaptive_impl() = default;

// NOTE: the save/load patterns mimic the copy constructor logic.
template <typename T>
template <typename Archive>
void taylor_adaptive_impl<T>::save_impl(Archive &ar, unsigned) const
{
    ar << m_state;
    ar << m_time;
    ar << m_llvm;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
    ar << m_ed_data;
}

template <typename T>
template <typename Archive>
void taylor_adaptive_impl<T>::load_impl(Archive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<taylor_adaptive_impl<T>>::type::value)) {
        throw std::invalid_argument("Unable to load a taylor_adaptive integrator: "
                                    "the archive version ({}) is too old"_format(version));
    }
    // LCOV_EXCL_STOP

    ar >> m_state;
    ar >> m_time;
    ar >> m_llvm;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_ed_data;

    // Recover the function pointers.
    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
}

template <typename T>
void taylor_adaptive_impl<T>::save(boost::archive::binary_oarchive &ar, unsigned v) const
{
    save_impl(ar, v);
}

template <typename T>
void taylor_adaptive_impl<T>::load(boost::archive::binary_iarchive &ar, unsigned v)
{
    load_impl(ar, v);
}

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced, but it will
// always be not greater than abs(max_delta_t). The propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will return a pair, containing
// a flag describing the outcome of the integration,
// and the integration timestep that was used.
//
// NOTE: for the docs:
// - outcome:
//   - if nf state is detected, err_nf_state, else
//   - if terminal events trigger, return the index
//     of the first event triggering, else
//   - either time_limit or success, depending on whether
//     max_delta_t was used as a timestep or not;
// - event detection happens in the [0, h) half-open range (that is,
//   all detected events are guaranteed to trigger within
//   the [0, h) range). Thus, if the timestep ends up being zero
//   (either because max_delta_t == 0
//   or the inferred timestep is zero), then event detection is skipped
//   altogether;
// - the execution of the events' callbacks is guaranteed to proceed in
//   chronological order;
// - a timestep h == 0 will still result in m_last_h being updated (to zero)
//   and the Taylor coefficients being recorded in the internal array
//   (if wtc == true). That is, h == 0 is not treated in any special way.
template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step_impl(T max_delta_t, bool wtc)
{
    using std::abs;
    using std::isfinite;

#if !defined(NDEBUG)
    // NOTE: this is the only precondition on max_delta_t.
    using std::isnan;
    assert(!isnan(max_delta_t)); // LCOV_EXCL_LINE
#endif

    auto h = max_delta_t;

    if (m_step_f.index() == 0u) {
        assert(!m_ed_data); // LCOV_EXCL_LINE

        // Invoke the vanilla stepper.
        std::get<0>(m_step_f)(m_state.data(), m_pars.data(), &m_time.hi, &h, wtc ? m_tc.data() : nullptr);

        // Update the time.
        m_time += h;

        // Store the last timestep.
        m_last_h = h;

        // Check if the time or the state vector are non-finite at the
        // end of the timestep.
        if (!isfinite(m_time)
            || std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !isfinite(x); })) {
            return std::tuple{taylor_outcome::err_nf_state, h};
        }

        return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, h};
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &ed_data = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        T max_abs_state;
        std::get<1>(m_step_f)(ed_data.m_ev_jet.data(), m_state.data(), m_pars.data(), &m_time.hi, &h, &max_abs_state);

        // Compute the maximum absolute error on the Taylor series of the event equations, which we will use for
        // automatic cooldown deduction. If max_abs_state is not finite, set it to inf so that
        // in ed_data.detect_events() we skip event detection altogether.
        T g_eps;
        if (isfinite(max_abs_state)) {
            // Are we in absolute or relative error control mode?
            const auto abs_or_rel = max_abs_state < 1;

            // Estimate the size of the largest remainder in the Taylor
            // series of both the dynamical equations and the events.
            const auto max_r_size = abs_or_rel ? m_tol : (m_tol * max_abs_state);

            // NOTE: depending on m_tol, max_r_size is arbitrarily small, but the real
            // integration error cannot be too small due to floating-point truncation.
            // This is the case for instance if we use sub-epsilon integration tolerances
            // to achieve Brouwer's law. In such a case, we cap the value of g_eps,
            // using eps * max_abs_state as an estimation of the smallest number
            // that can be resolved with the current floating-point type.
            // NOTE: the if condition in the next line is equivalent, in relative
            // error control mode, to:
            // if (m_tol < std::numeric_limits<T>::epsilon())
            if (max_r_size < std::numeric_limits<T>::epsilon() * max_abs_state) {
                g_eps = std::numeric_limits<T>::epsilon() * max_abs_state;
            } else {
                g_eps = max_r_size;
            }
        } else {
            g_eps = std::numeric_limits<T>::infinity();
        }

        // Write unconditionally the tcs.
        std::copy(ed_data.m_ev_jet.data(), ed_data.m_ev_jet.data() + m_dim * (m_order + 1u), m_tc.data());

        // Do the event detection.
        ed_data.detect_events(h, m_order, m_dim, g_eps);

        // NOTE: before this point, we did not alter
        // any user-visible data in the integrator (just
        // temporary memory). From here until we start invoking
        // the callbacks, everything is noexcept, so we don't
        // risk leaving the integrator in a half-baked state.

        // Sort the events by time.
        // NOTE: the time coordinates in m_d_(n)tes is relative
        // to the beginning of the timestep. It will be negative
        // for backward integration, thus we compare using
        // abs() so that the first events are those which
        // happen closer to the beginning of the timestep.
        // NOTE: the checks inside ed_data.detect_events() ensure
        // that we can safely sort the events' times.
        auto cmp = [](const auto &ev0, const auto &ev1) { return abs(std::get<1>(ev0)) < abs(std::get<1>(ev1)); };
        std::sort(ed_data.m_d_tes.begin(), ed_data.m_d_tes.end(), cmp);
        std::sort(ed_data.m_d_ntes.begin(), ed_data.m_d_ntes.end(), cmp);

        // If we have terminal events we need
        // to update the value of h.
        if (!ed_data.m_d_tes.empty()) {
            h = std::get<1>(ed_data.m_d_tes[0]);
        }

        // Update the state.
        m_d_out_f(m_state.data(), ed_data.m_ev_jet.data(), &h);

        // Update the time.
        m_time += h;

        // Store the last timestep.
        m_last_h = h;

        // Check if the time or the state vector are non-finite at the
        // end of the timestep.
        if (!isfinite(m_time)
            || std::any_of(m_state.cbegin(), m_state.cend(), [](const auto &x) { return !isfinite(x); })) {
            // Let's also reset the cooldown values, as at this point
            // they have become useless.
            reset_cooldowns();

            return std::tuple{taylor_outcome::err_nf_state, h};
        }

        // Update the cooldowns.
        for (auto &cd : ed_data.m_te_cooldowns) {
            if (cd) {
                // Check if the timestep we just took
                // brought this event outside the cooldown.
                auto tmp = cd->first + h;

                if (abs(tmp) >= cd->second) {
                    // We are now outside the cooldown period
                    // for this event, reset cd.
                    cd.reset();
                } else {
                    // Still in cooldown, update the
                    // time spent in cooldown.
                    cd->first = tmp;
                }
            }
        }

        // If we don't have terminal events, we will invoke the callbacks
        // of *all* the non-terminal events. Otherwise, we need to figure
        // out which non-terminal events do not happen because their time
        // coordinate is past the the first terminal event.
        const auto ntes_end_it
            = ed_data.m_d_tes.empty()
                  ? ed_data.m_d_ntes.end()
                  : std::lower_bound(ed_data.m_d_ntes.begin(), ed_data.m_d_ntes.end(), h,
                                     [](const auto &ev, const auto &t) { return abs(std::get<1>(ev)) < abs(t); });

        // Invoke the callbacks of the non-terminal events, which are guaranteed
        // to happen before the first terminal event.
        for (auto it = ed_data.m_d_ntes.begin(); it != ntes_end_it; ++it) {
            const auto &t = *it;
            const auto &cb = ed_data.m_ntes[std::get<0>(t)].get_callback();
            assert(cb); // LCOV_EXCL_LINE
            cb(*this, static_cast<T>(m_time - m_last_h + std::get<1>(t)), std::get<2>(t));
        }

        // The return value of the first
        // terminal event's callback. It will be
        // unused if there are no terminal events.
        bool te_cb_ret = false;

        if (!ed_data.m_d_tes.empty()) {
            // Fetch the first terminal event.
            const auto te_idx = std::get<0>(ed_data.m_d_tes[0]);
            assert(te_idx < ed_data.m_tes.size()); // LCOV_EXCL_LINE
            const auto &te = ed_data.m_tes[te_idx];

            // Set the corresponding cooldown.
            if (te.get_cooldown() >= 0) {
                // Cooldown explicitly provided by the user, use it.
                ed_data.m_te_cooldowns[te_idx].emplace(0, te.get_cooldown());
            } else {
                // Deduce the cooldown automatically.
                // NOTE: if g_eps is not finite, we skipped event detection
                // altogether and thus we never end up here. If the derivative
                // of the event equation is not finite, the event is also skipped.
                ed_data.m_te_cooldowns[te_idx].emplace(0,
                                                       taylor_deduce_cooldown(g_eps, std::get<4>(ed_data.m_d_tes[0])));
            }

            // Invoke the callback of the first terminal event, if it has one.
            if (te.get_callback()) {
                te_cb_ret = te.get_callback()(*this, std::get<2>(ed_data.m_d_tes[0]), std::get<3>(ed_data.m_d_tes[0]));
            }
        }

        if (ed_data.m_d_tes.empty()) {
            // No terminal events detected, return success or time limit.
            return std::tuple{h == max_delta_t ? taylor_outcome::time_limit : taylor_outcome::success, h};
        } else {
            // Terminal event detected. Fetch its index.
            const auto ev_idx = static_cast<std::int64_t>(std::get<0>(ed_data.m_d_tes[0]));

            // NOTE: if te_cb_ret is true, it means that the terminal event has
            // a callback and its invocation returned true (meaning that the
            // integration should continue). Otherwise, either the terminal event
            // has no callback or its callback returned false, meaning that the
            // integration must stop.
            return std::tuple{taylor_outcome{te_cb_ret ? ev_idx : (-ev_idx - 1)}, h};
        }
    }
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step(bool wtc)
{
    // NOTE: time limit +inf means integration forward in time
    // and no time limit.
    return step_impl(std::numeric_limits<T>::infinity(), wtc);
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step_backward(bool wtc)
{
    return step_impl(-std::numeric_limits<T>::infinity(), wtc);
}

template <typename T>
std::tuple<taylor_outcome, T> taylor_adaptive_impl<T>::step(T max_delta_t, bool wtc)
{
    using std::isnan;

    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A NaN max_delta_t was passed to the step() function of an adaptive Taylor integrator");
    }

    return step_impl(max_delta_t, wtc);
}

// Reset all cooldowns for the terminal events.
template <typename T>
void taylor_adaptive_impl<T>::reset_cooldowns()
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    for (auto &cd : m_ed_data->m_te_cooldowns) {
        cd.reset();
    }
}

// NOTE: possible outcomes:
// - time_limit iff the propagation was performed
//   up to the final time, else
// - cb_interrupt if the propagation was interrupted
//   via callback, else
// - err_nf_state if a non-finite state was generated, else
// - an event index if a stopping terminal event was encountered.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected.
// The continuous output is always updated at the end of each timestep,
// unless a non-finite state was detected.
template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t, std::optional<continuous_output<T>>>
taylor_adaptive_impl<T>::propagate_until_impl(const dfloat<T> &t, std::size_t max_steps, T max_delta_t,
                                              const std::function<bool(taylor_adaptive_impl &)> &cb, bool wtc,
                                              bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    // Check the current time.
    if (!isfinite(m_time)) {
        throw std::invalid_argument("Cannot invoke the propagate_until() function of an adaptive Taylor integrator if "
                                    "the current time is not finite");
    }

    // Check the final time.
    if (!isfinite(t)) {
        throw std::invalid_argument(
            "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // Check max_delta_t.
    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A nan max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator");
    }
    if (max_delta_t <= 0) {
        throw std::invalid_argument(
            "A non-positive max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // If with_c_out is true, we always need to write the Taylor coefficients.
    wtc = wtc || with_c_out;

    // These vectors are used in the construction of the continuous output.
    // If continuous output is not requested, they will remain empty.
    std::vector<T> c_out_tcs, c_out_times_hi, c_out_times_lo;
    if (with_c_out) {
        // Push in the starting time.
        c_out_times_hi.push_back(m_time.hi);
        c_out_times_lo.push_back(m_time.lo);
    }

    // Initial values for the counters
    // and the min/max abs of the integration
    // timesteps.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a nonzero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    // Init the remaining time.
    auto rem_time = t - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= T(0));

    // Helper to create the continuous output object.
    auto make_c_out = [&]() -> std::optional<continuous_output<T>> {
        if (with_c_out) {
            if (c_out_times_hi.size() < 2u) {
                // NOTE: this means that no successful steps
                // were taken.
                return {};
            }

            // Construct the return value.
            continuous_output<T> ret(m_llvm.make_similar());

            // Fill in the data.
            ret.m_tcs = std::move(c_out_tcs);
            ret.m_times_hi = std::move(c_out_times_hi);
            ret.m_times_lo = std::move(c_out_times_lo);

            // Prepare the output vector.
            ret.m_output.resize(boost::numeric_cast<decltype(ret.m_output.size())>(m_dim));

            // Add the continuous output function.
            ret.add_c_out_function(m_order, m_dim, m_high_accuracy);

            return std::optional{std::move(ret)};
        } else {
            return {};
        }
    };

    // Helper to update the continuous output data after a timestep.
    auto update_c_out = [&]() {
        if (with_c_out) {
#if !defined(NDEBUG)
            const dfloat<T> prev_time(c_out_times_hi.back(), c_out_times_lo.back());
#endif

            c_out_times_hi.push_back(m_time.hi);
            c_out_times_lo.push_back(m_time.lo);

#if !defined(NDEBUG)
            const dfloat<T> new_time(c_out_times_hi.back(), c_out_times_lo.back());
            assert(isfinite(new_time));
            if (t_dir) {
                assert(!(new_time < prev_time));
            } else {
                assert(!(new_time > prev_time));
            }
#endif

            c_out_tcs.insert(c_out_tcs.end(), m_tc.begin(), m_tc.end());
        }
    };

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        assert((rem_time >= T(0)) == t_dir); // LCOV_EXCL_LINE
        const auto dt_limit
            = t_dir ? std::min(dfloat<T>(max_delta_t), rem_time) : std::max(dfloat<T>(-max_delta_t), rem_time);
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        const auto [oc, h] = step_impl(static_cast<T>(dt_limit), wtc);

        if (oc == taylor_outcome::err_nf_state) {
            // If a non-finite state is detected, we do *not* want
            // to execute the propagate() callback and we do *not* want
            // to update the continuous output. Just exit.
            return std::tuple{oc, min_h, max_h, step_counter, make_c_out()};
        }

        // Update the number of steps.
        step_counter += static_cast<std::size_t>(h != 0);

        // Update min_h/max_h, but only if the outcome is success (otherwise
        // the step was artificially clamped either by a time limit or
        // by a terminal event).
        if (oc == taylor_outcome::success) {
            const auto abs_h = abs(h);
            min_h = std::min(min_h, abs_h);
            max_h = std::max(max_h, abs_h);
        }

        // Update the continuous output data.
        update_c_out();

        // Update the number of iterations.
        ++iter_counter;

        // Execute the propagate() callback, if applicable.
        if (cb && !cb(*this)) {
            // Interruption via callback.
            return std::tuple{taylor_outcome::cb_stop, min_h, max_h, step_counter, make_c_out()};
        }

        // The breakout conditions:
        // - a step of rem_time was used, or
        // - a stopping terminal event was detected.
        // NOTE: we check h == rem_time, instead of just
        // oc == time_limit, because clamping via max_delta_t
        // could also result in time_limit.
        const bool ste_detected = oc > taylor_outcome::success && oc < taylor_outcome{0};
        if (h == static_cast<T>(rem_time) || ste_detected) {
#if !defined(NDEBUG)
            if (h == static_cast<T>(rem_time)) {
                assert(oc == taylor_outcome::time_limit);
            }
#endif
            return std::tuple{oc, min_h, max_h, step_counter, make_c_out()};
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger (modulo wraparound)
        // as by this point we are sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter, make_c_out()};
        }

        // Update the remaining time.
        // NOTE: at this point, we are sure
        // that abs(h) < abs(static_cast<T>(rem_time)). This implies
        // that t - m_time cannot undergo a sign
        // flip and invert the integration direction.
        assert(abs(h) < abs(static_cast<T>(rem_time)));
        rem_time = t - m_time;
    }
}

template <typename T>
std::tuple<taylor_outcome, T, T, std::size_t, std::vector<T>>
taylor_adaptive_impl<T>::propagate_grid_impl(const std::vector<T> &grid, std::size_t max_steps, T max_delta_t,
                                             const std::function<bool(taylor_adaptive_impl &)> &cb)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    if (!isfinite(m_time)) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator if the current time is not finite");
    }

    // Check max_delta_t.
    if (isnan(max_delta_t)) {
        throw std::invalid_argument(
            "A nan max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator");
    }
    if (max_delta_t <= 0) {
        throw std::invalid_argument(
            "A non-positive max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator");
    }

    // Check the grid.
    if (grid.empty()) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator if the time grid is empty");
    }

    constexpr auto nf_err_msg
        = "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator";
    constexpr auto ig_err_msg
        = "A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator";
    // Check the first point.
    if (!isfinite(grid[0])) {
        throw std::invalid_argument(nf_err_msg);
    }
    if (grid.size() > 1u) {
        // Establish the direction of the grid from
        // the first two points.
        if (!isfinite(grid[1])) {
            throw std::invalid_argument(nf_err_msg);
        }
        if (grid[1] == grid[0]) {
            throw std::invalid_argument(ig_err_msg);
        }
        const auto grid_direction = grid[1] > grid[0];

        // Check that the remaining points are finite and that
        // they are ordered monotonically.
        for (decltype(grid.size()) i = 2; i < grid.size(); ++i) {
            if (!isfinite(grid[i])) {
                throw std::invalid_argument(nf_err_msg);
            }

            if ((grid[i] > grid[i - 1u]) != grid_direction) {
                throw std::invalid_argument(ig_err_msg);
            }
        }
    }

    // Pre-allocate the return value.
    std::vector<T> retval;
    // LCOV_EXCL_START
    if (get_dim() > std::numeric_limits<decltype(retval.size())>::max() / grid.size()) {
        throw std::overflow_error("Overflow detected in the creation of the return value of propagate_grid() in an "
                                  "adaptive Taylor integrator");
    }
    // LCOV_EXCL_STOP
    retval.reserve(grid.size() * get_dim());

    // Initial values for the counters
    // and the min/max abs of the integration
    // timesteps.
    // NOTE: iter_counter is for keeping track of the max_steps
    // limits, step_counter counts the number of timesteps performed
    // with a nonzero h. Most of the time these two quantities
    // will be identical, apart from corner cases.
    std::size_t iter_counter = 0, step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h(0);

    // Propagate the system up to the first grid point.
    // NOTE: we pass write_tc = true because some grid
    // points after the first one might end up being
    // calculated via dense output *before*
    // taking additional steps, and, in such case, we
    // must ensure the TCs are up to date.
    // NOTE: if the integrator time is already at grid[0],
    // then propagate_until() will take a zero timestep,
    // resulting in m_last_h also going to zero and no event
    // detection being performed.
    // NOTE: use the same max_steps for the initial propagation,
    // and don't pass the callback.
    const auto oc = std::get<0>(
        propagate_until(grid[0], kw::max_delta_t = max_delta_t, kw::max_steps = max_steps, kw::write_tc = true));

    if (oc != taylor_outcome::time_limit) {
        // The outcome is not time_limit, exit now.
        return std::tuple{oc, min_h, max_h, step_counter, std::move(retval)};
    }

    // Add the first result to retval.
    retval.insert(retval.end(), m_state.begin(), m_state.end());

    // Init the remaining time.
    auto rem_time = grid.back() - m_time;

    // Check it.
    if (!isfinite(rem_time)) {
        throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                    "integrator results in an overflow condition");
    }

    // Cache the integration direction.
    const auto t_dir = (rem_time >= T(0));

    // Iterate over the remaining grid points.
    for (decltype(grid.size()) cur_grid_idx = 1; cur_grid_idx < grid.size();) {
        // Establish the time range of the last
        // taken timestep.
        // NOTE: this is computed so that t0 < t1,
        // regardless of the integration direction.
        const auto t0 = std::min(m_time, m_time - m_last_h);
        const auto t1 = std::max(m_time, m_time - m_last_h);

        // Compute the state of the system via dense output for as many grid
        // points as possible, i.e., as long as the grid times
        // fall within the validity range for the dense output.
        while (true) {
            // Fetch the current time target.
            const auto cur_tt = grid[cur_grid_idx];

            // NOTE: we force processing of all remaining grid points
            // if we are at the last timestep. We do this in order to avoid
            // numerical issues when deciding if the last grid point
            // falls within the range of validity of the dense output.
            if ((cur_tt >= t0 && cur_tt <= t1) || (rem_time == dfloat<T>(T(0)))) {
                // The current time target falls within the range of
                // validity of the dense output. Compute the dense
                // output in cur_tt.
                update_d_output(cur_tt);

                // Add the result to retval.
                retval.insert(retval.end(), m_d_out.begin(), m_d_out.end());
            } else {
                // Cannot use dense output on the current time target,
                // need to take another step.
                break;
            }

            // Move to the next time target, or break out
            // if we have no more.
            if (++cur_grid_idx == grid.size()) {
                break;
            }
        }

        if (cur_grid_idx == grid.size()) {
            // No more grid points, exit.
            break;
        }

        // Take the next step, making sure to write the Taylor coefficients
        // and to cap the timestep size so that we don't go past the
        // last grid point and we don't use a timestep exceeding max_delta_t.
        // NOTE: rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        assert((rem_time >= T(0)) == t_dir); // LCOV_EXCL_LINE
        const auto dt_limit
            = t_dir ? std::min(dfloat<T>(max_delta_t), rem_time) : std::max(dfloat<T>(-max_delta_t), rem_time);
        const auto [oc, h] = step_impl(static_cast<T>(dt_limit), true);

        if (oc != taylor_outcome::success && oc != taylor_outcome::time_limit && oc < taylor_outcome{0}) {
            // Something went wrong in the propagation of the timestep, or we reached
            // a stopping terminal event.

            if (oc > taylor_outcome::success) {
                // In case of a stopping terminal event, we still want to update
                // the step counter.
                step_counter += static_cast<std::size_t>(h != 0);
            }

            return std::tuple{oc, min_h, max_h, step_counter, std::move(retval)};
        }

        // Update the number of iterations.
        ++iter_counter;

        // Update the number of steps.
        step_counter += static_cast<std::size_t>(h != 0);

        // Update min_h/max_h, but only if the outcome is success (otherwise
        // the step was artificially clamped either by a time limit or
        // by a terminal event).
        if (oc == taylor_outcome::success) {
            const auto abs_h = abs(h);
            min_h = std::min(min_h, abs_h);
            max_h = std::max(max_h, abs_h);
        }

        // Step successful: invoke the callback, if needed.
        if (cb && !cb(*this)) {
            // Interruption via callback.
            return std::tuple{taylor_outcome::cb_stop, min_h, max_h, step_counter, std::move(retval)};
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger as by this point we are
        // sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            return std::tuple{taylor_outcome::step_limit, min_h, max_h, step_counter, std::move(retval)};
        }

        // Update the remaining time.
        // NOTE: if static_cast<T>(rem_time) was used as a timestep,
        // it means that we hit the time limit. Force rem_time to zero
        // to signal this, avoiding inconsistencies with grid.back() - m_time
        // not going exactly to zero due to numerical issues. A zero rem_time
        // will also force the processing of all remaining grid points.
        if (h == static_cast<T>(rem_time)) {
            assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE
            rem_time = dfloat<T>(T(0));
        } else {
            rem_time = grid.back() - m_time;
        }
    }

    // Everything went well, return time_limit.
    return std::tuple{taylor_outcome::time_limit, min_h, max_h, step_counter, std::move(retval)};
}

template <typename T>
const llvm_state &taylor_adaptive_impl<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const taylor_dc_t &taylor_adaptive_impl<T>::get_decomposition() const
{
    return m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive_impl<T>::get_order() const
{
    return m_order;
}

template <typename T>
T taylor_adaptive_impl<T>::get_tol() const
{
    return m_tol;
}

template <typename T>
bool taylor_adaptive_impl<T>::get_high_accuracy() const
{
    return m_high_accuracy;
}

template <typename T>
bool taylor_adaptive_impl<T>::get_compact_mode() const
{
    return m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive_impl<T>::get_dim() const
{
    return m_dim;
}

template <typename T>
void taylor_adaptive_impl<T>::set_dtime(T hi, T lo)
{
    m_time = normalise(dfloat<T>{hi, lo});
}

template <typename T>
const std::vector<T> &taylor_adaptive_impl<T>::update_d_output(T time, bool rel_time)
{
    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        const auto h = m_last_h + time;

        m_d_out_f(m_d_out.data(), m_tc.data(), &h);
    } else {
        // Absolute time coordinate.
        const auto h = time - (m_time - m_last_h);

        m_d_out_f(m_d_out.data(), m_tc.data(), &h.hi);
    }

    return m_d_out;
}

// Explicit instantiation of the implementation classes/functions.
// NOTE: on Windows apparently it is necessary to declare that
// these instantiations are meant to be dll-exported.
template class taylor_adaptive_impl<double>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<double>::finalise_ctor_impl(const std::vector<expression> &,
                                                                                 std::vector<double>, double, double,
                                                                                 bool, bool, std::vector<double>,
                                                                                 std::vector<t_event_t>,
                                                                                 std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void
taylor_adaptive_impl<double>::finalise_ctor_impl(const std::vector<std::pair<expression, expression>> &,
                                                 std::vector<double>, double, double, bool, bool, std::vector<double>,
                                                 std::vector<t_event_t>, std::vector<nt_event_t>);

template class taylor_adaptive_impl<long double>;

template HEYOKA_DLL_PUBLIC void
taylor_adaptive_impl<long double>::finalise_ctor_impl(const std::vector<expression> &, std::vector<long double>,
                                                      long double, long double, bool, bool, std::vector<long double>,
                                                      std::vector<t_event_t>, std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<long double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<long double>, long double, long double, bool,
    bool, std::vector<long double>, std::vector<t_event_t>, std::vector<nt_event_t>);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_impl<mppp::real128>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<mppp::real128>, mppp::real128, mppp::real128, bool, bool,
    std::vector<mppp::real128>, std::vector<t_event_t>, std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_impl<mppp::real128>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<mppp::real128>, mppp::real128, mppp::real128,
    bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>, std::vector<nt_event_t>);

#endif

} // namespace detail

namespace detail
{

template <typename T>
template <typename U>
void taylor_adaptive_batch_impl<T>::finalise_ctor_impl(const U &sys, std::vector<T> state, std::uint32_t batch_size,
                                                       std::vector<T> time, T tol, bool high_accuracy,
                                                       bool compact_mode, std::vector<T> pars,
                                                       std::vector<t_event_t> tes, std::vector<nt_event_t> ntes)
{
#if defined(HEYOKA_ARCH_PPC)
    if constexpr (std::is_same_v<T, long double>) {
        throw not_implemented_error("'long double' computations are not supported on PowerPC");
    }
#endif

    using std::isfinite;

    // Init the data members.
    m_batch_size = batch_size;
    m_state = std::move(state);
    m_time_hi = std::move(time);
    m_time_lo.resize(m_time_hi.size());
    m_pars = std::move(pars);
    m_high_accuracy = high_accuracy;
    m_compact_mode = compact_mode;

    // Check input params.
    if (m_batch_size == 0u) {
        throw std::invalid_argument("The batch size in an adaptive Taylor integrator cannot be zero");
    }

    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() % m_batch_size != 0u) {
        throw std::invalid_argument(
            "Invalid size detected in the initialization of an adaptive Taylor "
            "integrator: the state vector has a size of {}, which is not a multiple of the batch size ({})"_format(
                m_state.size(), m_batch_size));
    }

    if (m_state.size() / m_batch_size != sys.size()) {
        throw std::invalid_argument(
            "Inconsistent sizes detected in the initialization of an adaptive Taylor "
            "integrator: the state vector has a dimension of {} and a batch size of {}, "
            "while the number of equations is {}"_format(m_state.size() / m_batch_size, m_batch_size, sys.size()));
    }

    if (m_time_hi.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid size detected in the initialization of an adaptive Taylor "
            "integrator: the time vector has a size of {}, which is not equal to the batch size ({})"_format(
                m_time_hi.size(), m_batch_size));
    }
    // NOTE: no need to check m_time_lo for finiteness, as it
    // was inited to zero already.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), [](const auto &x) { return !isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite initial time was detected in the initialisation of an adaptive Taylor integrator");
    }

    if (!isfinite(tol) || tol <= 0) {
        throw std::invalid_argument(
            "The tolerance in an adaptive Taylor integrator must be finite and positive, but it is {} instead"_format(
                tol));
    }

    // Store the tolerance.
    m_tol = tol;

    // Store the dimension of the system.
    m_dim = boost::numeric_cast<std::uint32_t>(sys.size());

    // Do we have events?
    const auto with_events = !tes.empty() || !ntes.empty();

    // Temporarily disable optimisations in s, so that
    // we don't optimise twice when adding the step
    // and then the d_out.
    std::optional<opt_disabler> od(m_llvm);

    // Add the stepper function.
    if (with_events) {
        std::vector<expression> ee;
        // NOTE: no need for deep copies of the expressions: ee is never mutated
        // and we will be deep-copying it anyway when we do the decomposition.
        for (const auto &ev : tes) {
            ee.push_back(ev.get_expression());
        }
        for (const auto &ev : ntes) {
            ee.push_back(ev.get_expression());
        }

        std::tie(m_dc, m_order) = taylor_add_adaptive_step_with_events<T>(
            m_llvm, "step_e", sys, tol, batch_size, high_accuracy, compact_mode, ee, high_accuracy);
    } else {
        std::tie(m_dc, m_order)
            = taylor_add_adaptive_step<T>(m_llvm, "step", sys, tol, batch_size, high_accuracy, compact_mode);
    }

    // Fix m_pars' size, if necessary.
    const auto npars = n_pars_in_dc(m_dc);
    // LCOV_EXCL_START
    if (npars > std::numeric_limits<std::uint32_t>::max() / m_batch_size) {
        throw std::overflow_error("Overflow detected when computing the size of the parameter array in an adaptive "
                                  "Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP
    if (m_pars.size() < npars * m_batch_size) {
        m_pars.resize(boost::numeric_cast<decltype(m_pars.size())>(npars * m_batch_size));
    } else if (m_pars.size() > npars * m_batch_size) {
        throw std::invalid_argument("Excessive number of parameter values passed to the constructor of an adaptive "
                                    "Taylor integrator in batch mode: {} parameter values were passed, but the ODE "
                                    "system contains only {} parameters "
                                    "(in batches of {})"_format(m_pars.size(), npars, m_batch_size));
    }

    // Log runtimes in trace mode.
    spdlog::stopwatch sw;

    // Add the function for the computation of
    // the dense output.
    taylor_add_d_out_function<T>(m_llvm, m_dim, m_order, m_batch_size, high_accuracy);

    get_logger()->trace("Taylor batch dense output runtime: {}", sw);
    sw.reset();

    // Restore the original optimisation level in s.
    od.reset();

    // Run the optimisation pass manually.
    m_llvm.optimise();

    get_logger()->trace("Taylor batch global opt pass runtime: {}", sw);
    sw.reset();

    // Run the jit.
    m_llvm.compile();

    get_logger()->trace("Taylor batch LLVM compilation runtime: {}", sw);

    // Fetch the stepper.
    if (with_events) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }

    // Fetch the function to compute the dense output.
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));

    // Setup the vector for the Taylor coefficients.
    // LCOV_EXCL_START
    if (m_order == std::numeric_limits<std::uint32_t>::max()
        || m_state.size() > std::numeric_limits<decltype(m_tc.size())>::max() / (m_order + 1u)) {
        throw std::overflow_error(
            "Overflow detected in the initialisation of an adaptive Taylor integrator in batch mode: the order "
            "or the state size is too large");
    }
    // LCOV_EXCL_STOP

    // NOTE: the size of m_state.size() already takes
    // into account the batch size.
    m_tc.resize(m_state.size() * (m_order + 1u));

    // Setup m_last_h.
    m_last_h.resize(boost::numeric_cast<decltype(m_last_h.size())>(batch_size));

    // Setup the vector for the dense output.
    // NOTE: the size of m_state.size() already takes
    // into account the batch size.
    m_d_out.resize(m_state.size());

    // Prepare the temp vectors.
    m_pinf.resize(m_batch_size, std::numeric_limits<T>::infinity());
    m_minf.resize(m_batch_size, -std::numeric_limits<T>::infinity());
    // LCOV_EXCL_START
    if (m_batch_size > std::numeric_limits<std::uint32_t>::max() / 2u) {
        throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator in batch "
                                  "mode: the batch size is too large");
    }
    // LCOV_EXCL_STOP
    // NOTE: make it twice as big because we need twice the storage in step_impl().
    m_delta_ts.resize(boost::numeric_cast<decltype(m_delta_ts.size())>(m_batch_size * 2u));

    // NOTE: init the outcome to success, the rest to zero.
    m_step_res.resize(boost::numeric_cast<decltype(m_step_res.size())>(m_batch_size),
                      std::tuple{taylor_outcome::success, T(0)});
    m_prop_res.resize(boost::numeric_cast<decltype(m_prop_res.size())>(m_batch_size),
                      std::tuple{taylor_outcome::success, T(0), T(0), std::size_t(0)});

    m_ts_count.resize(boost::numeric_cast<decltype(m_ts_count.size())>(m_batch_size));
    m_min_abs_h.resize(m_batch_size);
    m_max_abs_h.resize(m_batch_size);
    m_cur_max_delta_ts.resize(m_batch_size);
    m_pfor_ts.resize(boost::numeric_cast<decltype(m_pfor_ts.size())>(m_batch_size));
    m_t_dir.resize(boost::numeric_cast<decltype(m_t_dir.size())>(m_batch_size));
    m_rem_time.resize(m_batch_size);

    m_d_out_time.resize(m_batch_size);

    // Init the event data structure if needed.
    // NOTE: this can be done in parallel with the rest of the constructor,
    // once we have m_order/m_dim/m_batch_size and we are done using tes/ntes.
    if (with_events) {
        m_ed_data = std::make_unique<ed_data>(std::move(tes), std::move(ntes), m_order, m_dim, m_batch_size);
    }
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl()
    : taylor_adaptive_batch_impl({prime("x"_var) = 0_dbl}, {T(0)}, 1u, kw::tol = T(1e-1))
{
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(const taylor_adaptive_batch_impl &other)
    // NOTE: make a manual copy of all members, apart from the function pointers.
    : m_batch_size(other.m_batch_size), m_state(other.m_state), m_time_hi(other.m_time_hi), m_time_lo(other.m_time_lo),
      m_llvm(other.m_llvm), m_dim(other.m_dim), m_order(other.m_order), m_tol(other.m_tol),
      m_high_accuracy(other.m_high_accuracy), m_compact_mode(other.m_compact_mode), m_pars(other.m_pars),
      m_tc(other.m_tc), m_last_h(other.m_last_h), m_d_out(other.m_d_out), m_pinf(other.m_pinf), m_minf(other.m_minf),
      m_delta_ts(other.m_delta_ts), m_step_res(other.m_step_res), m_prop_res(other.m_prop_res),
      m_ts_count(other.m_ts_count), m_min_abs_h(other.m_min_abs_h), m_max_abs_h(other.m_max_abs_h),
      m_cur_max_delta_ts(other.m_cur_max_delta_ts), m_pfor_ts(other.m_pfor_ts), m_t_dir(other.m_t_dir),
      m_rem_time(other.m_rem_time), m_d_out_time(other.m_d_out_time),
      m_ed_data(other.m_ed_data ? std::make_unique<ed_data>(*other.m_ed_data) : nullptr)
{
    // NOTE: make explicit deep copy of the decomposition.
    m_dc.reserve(other.m_dc.size());
    for (const auto &[ex, deps] : other.m_dc) {
        m_dc.emplace_back(copy(ex), deps);
    }

    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
}

template <typename T>
taylor_adaptive_batch_impl<T>::taylor_adaptive_batch_impl(taylor_adaptive_batch_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_batch_impl<T> &taylor_adaptive_batch_impl<T>::operator=(const taylor_adaptive_batch_impl &other)
{
    if (this != &other) {
        *this = taylor_adaptive_batch_impl(other);
    }

    return *this;
}

template <typename T>
taylor_adaptive_batch_impl<T> &
taylor_adaptive_batch_impl<T>::operator=(taylor_adaptive_batch_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_batch_impl<T>::~taylor_adaptive_batch_impl() = default;

// NOTE: the save/load patterns mimic the copy constructor logic.
template <typename T>
template <typename Archive>
void taylor_adaptive_batch_impl<T>::save_impl(Archive &ar, unsigned) const
{
    // NOTE: save all members, apart from the function pointers.
    ar << m_batch_size;
    ar << m_state;
    ar << m_time_hi;
    ar << m_time_lo;
    ar << m_llvm;
    ar << m_dim;
    ar << m_dc;
    ar << m_order;
    ar << m_tol;
    ar << m_high_accuracy;
    ar << m_compact_mode;
    ar << m_pars;
    ar << m_tc;
    ar << m_last_h;
    ar << m_d_out;
    ar << m_pinf;
    ar << m_minf;
    ar << m_delta_ts;
    ar << m_step_res;
    ar << m_prop_res;
    ar << m_ts_count;
    ar << m_min_abs_h;
    ar << m_max_abs_h;
    ar << m_cur_max_delta_ts;
    ar << m_pfor_ts;
    ar << m_t_dir;
    ar << m_rem_time;
    ar << m_d_out_time;
    ar << m_ed_data;
}

template <typename T>
template <typename Archive>
void taylor_adaptive_batch_impl<T>::load_impl(Archive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<taylor_adaptive_batch_impl<T>>::type::value)) {
        throw std::invalid_argument("Unable to load a taylor_adaptive_batch integrator: "
                                    "the archive version ({}) is too old"_format(version));
    }
    // LCOV_EXCL_STOP

    ar >> m_batch_size;
    ar >> m_state;
    ar >> m_time_hi;
    ar >> m_time_lo;
    ar >> m_llvm;
    ar >> m_dim;
    ar >> m_dc;
    ar >> m_order;
    ar >> m_tol;
    ar >> m_high_accuracy;
    ar >> m_compact_mode;
    ar >> m_pars;
    ar >> m_tc;
    ar >> m_last_h;
    ar >> m_d_out;
    ar >> m_pinf;
    ar >> m_minf;
    ar >> m_delta_ts;
    ar >> m_step_res;
    ar >> m_prop_res;
    ar >> m_ts_count;
    ar >> m_min_abs_h;
    ar >> m_max_abs_h;
    ar >> m_cur_max_delta_ts;
    ar >> m_pfor_ts;
    ar >> m_t_dir;
    ar >> m_rem_time;
    ar >> m_d_out_time;
    ar >> m_ed_data;

    // Recover the function pointers.
    if (m_ed_data) {
        m_step_f = reinterpret_cast<step_f_e_t>(m_llvm.jit_lookup("step_e"));
    } else {
        m_step_f = reinterpret_cast<step_f_t>(m_llvm.jit_lookup("step"));
    }
    m_d_out_f = reinterpret_cast<d_out_f_t>(m_llvm.jit_lookup("d_out_f"));
}

template <typename T>
void taylor_adaptive_batch_impl<T>::save(boost::archive::binary_oarchive &ar, unsigned v) const
{
    save_impl(ar, v);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::load(boost::archive::binary_iarchive &ar, unsigned v)
{
    load_impl(ar, v);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_time(const std::vector<T> &new_time)
{
    // Check the dimensionality of new_time.
    if (new_time.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified times is {}"_format(m_batch_size, new_time.size()));
    }

    // Copy over the new times.
    // NOTE: do not use std::copy(), as it gives UB if new_time
    // and m_time_hi are the same object.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_time_hi[i] = new_time[i];
    }
    // Reset the lo part.
    std::fill(m_time_lo.begin(), m_time_lo.end(), T(0));
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_time(T new_time)
{
    // Set the hi part.
    std::fill(m_time_hi.begin(), m_time_hi.end(), new_time);
    // Reset the lo part.
    std::fill(m_time_lo.begin(), m_time_lo.end(), T(0));
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_dtime(const std::vector<T> &hi, const std::vector<T> &lo)
{
    // Check the dimensionalities.
    if (hi.size() != m_batch_size || lo.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified times is ({}, {})"_format(m_batch_size, hi.size(), lo.size()));
    }

    // Copy over the new times, ensuring proper
    // normalisation.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        const auto tmp = normalise(dfloat<T>{hi[i], lo[i]});

        m_time_hi[i] = tmp.hi;
        m_time_lo[i] = tmp.lo;
    }
}

template <typename T>
void taylor_adaptive_batch_impl<T>::set_dtime(T hi, T lo)
{
    // Copy over the new time, ensuring proper
    // normalisation.
    const auto tmp = normalise(dfloat<T>{hi, lo});
    std::fill(m_time_hi.begin(), m_time_hi.end(), tmp.hi);
    std::fill(m_time_lo.begin(), m_time_lo.end(), tmp.lo);
}

// Implementation detail to make a single integration timestep.
// The magnitude of the timestep is automatically deduced for each
// state vector, but it will always be not greater than
// the absolute value of the corresponding element in max_delta_ts.
// For each state vector, the propagation
// is done forward in time if max_delta_t >= 0, backwards in
// time otherwise.
//
// The function will write to res a pair for each state
// vector, containing a flag describing the outcome of the integration
// and the integration timestep that was used.
//
// NOTE: for the docs:
// - document exception rethrowing behaviour when an event
//   callback throws;
// - outcome for each batch element:
//   - if nf state is detected, err_nf_state, else
//   - if terminal events trigger, return the index
//     of the first event triggering, else
//   - either time_limit or success, depending on whether
//     max_delta_t was used as a timestep or not;
// - the docs for the scalar step function are applicable to
//   the batch version too.
template <typename T>
void taylor_adaptive_batch_impl<T>::step_impl(const std::vector<T> &max_delta_ts, bool wtc)
{
    using std::abs;
    using std::isfinite;

    // LCOV_EXCL_START
    // Check preconditions.
    assert(max_delta_ts.size() == m_batch_size);
    assert(std::none_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
        using std::isnan;
        return isnan(x);
    }));

    // Sanity check.
    assert(m_step_res.size() == m_batch_size);
    // LCOV_EXCL_STOP

    // Copy max_delta_ts to the tmp buffer, twice.
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.data());
    std::copy(max_delta_ts.begin(), max_delta_ts.end(), m_delta_ts.data() + m_batch_size);

    // Helper to check if the state vector of a batch element
    // contains a non-finite value.
    auto check_nf_batch = [this](std::uint32_t batch_idx) {
        for (std::uint32_t i = 0; i < m_dim; ++i) {
            if (!isfinite(m_state[i * m_batch_size + batch_idx])) {
                return true;
            }
        }
        return false;
    };

    if (m_step_f.index() == 0u) {
        assert(!m_ed_data); // LCOV_EXCL_LINE

        std::get<0>(m_step_f)(m_state.data(), m_pars.data(), m_time_hi.data(), m_delta_ts.data(),
                              wtc ? m_tc.data() : nullptr);

        // Update the times and the last timesteps, and write out the result.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // The timestep that was actually used for
            // this batch element.
            const auto h = m_delta_ts[i];

            // Compute the new time in double-length arithmetic.
            const auto new_time = dfloat<T>(m_time_hi[i], m_time_lo[i]) + h;
            m_time_hi[i] = new_time.hi;
            m_time_lo[i] = new_time.lo;

            // Update the size of the last timestep.
            m_last_h[i] = h;

            if (!isfinite(new_time) || check_nf_batch(i)) {
                // Either the new time or state contain non-finite values,
                // return an error condition.
                m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};
            } else {
                m_step_res[i] = std::tuple{
                    // NOTE: use here the original value of
                    // max_delta_ts, stored at the end of m_delta_ts,
                    // in case max_delta_ts aliases integrator data
                    // which was modified during the step.
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
            }
        }
    } else {
        assert(m_ed_data); // LCOV_EXCL_LINE

        auto &ed_data = *m_ed_data;

        // Invoke the stepper for event handling. We will record the norm infinity of the state vector +
        // event equations at the beginning of the timestep for later use.
        std::get<1>(m_step_f)(ed_data.m_ev_jet.data(), m_state.data(), m_pars.data(), m_time_hi.data(),
                              m_delta_ts.data(), ed_data.m_max_abs_state.data());

        // Compute the maximum absolute error on the Taylor series of the event equations, which we will use for
        // automatic cooldown deduction. If max_abs_state is not finite, set it to inf so that
        // in ed_data.detect_events() we skip event detection altogether.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto max_abs_state = ed_data.m_max_abs_state[i];

            if (isfinite(max_abs_state)) {
                // Are we in absolute or relative error control mode?
                const auto abs_or_rel = max_abs_state < 1;

                // Estimate the size of the largest remainder in the Taylor
                // series of both the dynamical equations and the events.
                const auto max_r_size = abs_or_rel ? m_tol : (m_tol * max_abs_state);

                // NOTE: depending on m_tol, max_r_size is arbitrarily small, but the real
                // integration error cannot be too small due to floating-point truncation.
                // This is the case for instance if we use sub-epsilon integration tolerances
                // to achieve Brouwer's law. In such a case, we cap the value of g_eps,
                // using eps * max_abs_state as an estimation of the smallest number
                // that can be resolved with the current floating-point type.
                // NOTE: the if condition in the next line is equivalent, in relative
                // error control mode, to:
                // if (m_tol < std::numeric_limits<T>::epsilon())
                if (max_r_size < std::numeric_limits<T>::epsilon() * max_abs_state) {
                    ed_data.m_g_eps[i] = std::numeric_limits<T>::epsilon() * max_abs_state;
                } else {
                    ed_data.m_g_eps[i] = max_r_size;
                }
            } else {
                ed_data.m_g_eps[i] = std::numeric_limits<T>::infinity();
            }
        }

        // Write unconditionally the tcs.
        std::copy(ed_data.m_ev_jet.data(), ed_data.m_ev_jet.data() + m_dim * (m_order + 1u) * m_batch_size,
                  m_tc.data());

        // Do the event detection.
        ed_data.detect_events(m_delta_ts.data(), m_order, m_dim, m_batch_size);

        // NOTE: before this point, we did not alter
        // any user-visible data in the integrator (just
        // temporary memory). From here until we start invoking
        // the callbacks, everything is noexcept, so we don't
        // risk leaving the integrator in a half-baked state.

        // Sort the events by time.
        // NOTE: the time coordinates in m_d_(n)tes is relative
        // to the beginning of the timestep. It will be negative
        // for backward integration, thus we compare using
        // abs() so that the first events are those which
        // happen closer to the beginning of the timestep.
        // NOTE: the checks inside ed_data.detect_events() ensure
        // that we can safely sort the events' times.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            auto cmp = [](const auto &ev0, const auto &ev1) { return abs(std::get<1>(ev0)) < abs(std::get<1>(ev1)); };
            std::sort(ed_data.m_d_tes[i].begin(), ed_data.m_d_tes[i].end(), cmp);
            std::sort(ed_data.m_d_ntes[i].begin(), ed_data.m_d_ntes[i].end(), cmp);

            // If we have terminal events we need
            // to update the value of h.
            if (!ed_data.m_d_tes[i].empty()) {
                m_delta_ts[i] = std::get<1>(ed_data.m_d_tes[i][0]);
            }
        }

        // Update the state.
        m_d_out_f(m_state.data(), ed_data.m_ev_jet.data(), m_delta_ts.data());

        // We will use this to capture the first exception thrown
        // by a callback, if any.
        std::exception_ptr eptr;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto h = m_delta_ts[i];

            // Compute the new time in double-length arithmetic.
            const auto new_time = dfloat<T>(m_time_hi[i], m_time_lo[i]) + h;
            m_time_hi[i] = new_time.hi;
            m_time_lo[i] = new_time.lo;

            // Store the last timestep.
            m_last_h[i] = h;

            // Check if the time or the state vector are non-finite at the
            // end of the timestep.
            if (!isfinite(new_time) || check_nf_batch(i)) {
                // Let's also reset the cooldown values for this batch index,
                // as at this point they have become useless.
                reset_cooldowns(i);

                m_step_res[i] = std::tuple{taylor_outcome::err_nf_state, h};

                // Move to the next batch element.
                continue;
            }

            // Update the cooldowns.
            for (auto &cd : ed_data.m_te_cooldowns[i]) {
                if (cd) {
                    // Check if the timestep we just took
                    // brought this event outside the cooldown.
                    auto tmp = cd->first + h;

                    if (abs(tmp) >= cd->second) {
                        // We are now outside the cooldown period
                        // for this event, reset cd.
                        cd.reset();
                    } else {
                        // Still in cooldown, update the
                        // time spent in cooldown.
                        cd->first = tmp;
                    }
                }
            }

            // If we don't have terminal events, we will invoke the callbacks
            // of *all* the non-terminal events. Otherwise, we need to figure
            // out which non-terminal events do not happen because their time
            // coordinate is past the the first terminal event.
            const auto ntes_end_it
                = ed_data.m_d_tes[i].empty()
                      ? ed_data.m_d_ntes[i].end()
                      : std::lower_bound(ed_data.m_d_ntes[i].begin(), ed_data.m_d_ntes[i].end(), h,
                                         [](const auto &ev, const auto &t) { return abs(std::get<1>(ev)) < abs(t); });

            // Invoke the callbacks of the non-terminal events, which are guaranteed
            // to happen before the first terminal event.
            for (auto it = ed_data.m_d_ntes[i].begin(); it != ntes_end_it; ++it) {
                const auto &t = *it;
                const auto &cb = ed_data.m_ntes[std::get<0>(t)].get_callback();
                assert(cb); // LCOV_EXCL_LINE
                try {
                    cb(*this, static_cast<T>(new_time - m_last_h[i] + std::get<1>(t)), std::get<2>(t), i);
                } catch (...) {
                    if (!eptr) {
                        eptr = std::current_exception();
                    }
                }
            }

            // The return value of the first
            // terminal event's callback. It will be
            // unused if there are no terminal events.
            bool te_cb_ret = false;

            if (!ed_data.m_d_tes[i].empty()) {
                // Fetch the first terminal event.
                const auto te_idx = std::get<0>(ed_data.m_d_tes[i][0]);
                assert(te_idx < ed_data.m_tes.size()); // LCOV_EXCL_LINE
                const auto &te = ed_data.m_tes[te_idx];

                // Set the corresponding cooldown.
                if (te.get_cooldown() >= 0) {
                    // Cooldown explicitly provided by the user, use it.
                    ed_data.m_te_cooldowns[i][te_idx].emplace(0, te.get_cooldown());
                } else {
                    // Deduce the cooldown automatically.
                    // NOTE: if m_g_eps[i] is not finite, we skipped event detection
                    // altogether and thus we never end up here. If the derivative
                    // of the event equation is not finite, the event is also skipped.
                    ed_data.m_te_cooldowns[i][te_idx].emplace(
                        0, taylor_deduce_cooldown(ed_data.m_g_eps[i], std::get<4>(ed_data.m_d_tes[i][0])));
                }

                // Invoke the callback of the first terminal event, if it has one.
                if (te.get_callback()) {
                    try {
                        te_cb_ret = te.get_callback()(*this, std::get<2>(ed_data.m_d_tes[i][0]),
                                                      std::get<3>(ed_data.m_d_tes[i][0]), i);
                    } catch (...) {
                        if (!eptr) {
                            eptr = std::current_exception();
                        }
                    }
                }
            }

            if (ed_data.m_d_tes[i].empty()) {
                // No terminal events detected, return success or time limit.
                m_step_res[i] = std::tuple{
                    // NOTE: use here the original value of
                    // max_delta_ts, stored at the end of m_delta_ts,
                    // in case max_delta_ts aliases integrator data
                    // which was modified during the step.
                    h == m_delta_ts[m_batch_size + i] ? taylor_outcome::time_limit : taylor_outcome::success, h};
            } else {
                // Terminal event detected. Fetch its index.
                const auto ev_idx = static_cast<std::int64_t>(std::get<0>(ed_data.m_d_tes[i][0]));

                // NOTE: if te_cb_ret is true, it means that the terminal event has
                // a callback and its invocation returned true (meaning that the
                // integration should continue). Otherwise, either the terminal event
                // has no callback or its callback returned false, meaning that the
                // integration must stop.
                m_step_res[i] = std::tuple{taylor_outcome{te_cb_ret ? ev_idx : (-ev_idx - 1)}, h};
            }
        }

        // Check if any callback threw an exception, and re-throw
        // it in case.
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step(bool wtc)
{
    step_impl(m_pinf, wtc);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step_backward(bool wtc)
{
    step_impl(m_minf, wtc);
}

template <typename T>
void taylor_adaptive_batch_impl<T>::step(const std::vector<T> &max_delta_ts, bool wtc)
{
    // Check the dimensionality of max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}"_format(m_batch_size, max_delta_ts.size()));
    }

    // Make sure no values in max_delta_ts are nan.
    if (std::any_of(max_delta_ts.begin(), max_delta_ts.end(), [](const auto &x) {
            using std::isnan;
            return isnan(x);
        })) {
        throw std::invalid_argument(
            "Cannot invoke the step() function of an adaptive Taylor integrator in batch mode if "
            "one of the max timesteps is nan");
    }

    step_impl(max_delta_ts, wtc);
}

template <typename T>
std::optional<continuous_output_batch<T>> taylor_adaptive_batch_impl<T>::propagate_for_impl(
    const std::vector<T> &delta_ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch_impl &)> &cb, bool wtc, bool with_c_out)
{
    // Check the dimensionality of delta_ts.
    if (delta_ts.size() != m_batch_size) {
        throw std::invalid_argument("Invalid number of time intervals specified in a Taylor integrator in batch mode: "
                                    "the batch size is {}, but the number of specified time intervals is {}"_format(
                                        m_batch_size, delta_ts.size()));
    }

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = dfloat<T>(m_time_hi[i], m_time_lo[i]) + delta_ts[i];
    }

    // NOTE: max_delta_ts is checked in propagate_until_impl().
    return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts, std::move(cb), wtc, with_c_out);
}

// NOTE: possible outcomes:
// - all time_limit iff all batch elements
//   were successfully propagated up to the final times; else,
// - all cb_interrupt if the integration was interrupted via
//   a callback; else,
// - 1 or more err_nf_state if 1 or more batch elements generated
//   a non-finite state. The other elements will have their outcome
//   set by the last step taken; else,
// - 1 or more event indices if 1 or more batch elements generated
//   a stopping terminal event. The other elements will have their outcome
//   set by the last step taken.
// The callback is always executed at the end of each timestep, unless
// a non-finite state was detected in any batch element.
// The continuous output is always updated at the end of each timestep,
// unless a non-finite state was detected in any batch element.
template <typename T>
std::optional<continuous_output_batch<T>> taylor_adaptive_batch_impl<T>::propagate_until_impl(
    const std::vector<dfloat<T>> &ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch_impl &)> &cb, bool wtc, bool with_c_out)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    // NOTE: this function is called from either the other propagate_until() overload,
    // or propagate_for(). In both cases, we have already set up correctly the dimension of ts.
    assert(ts.size() == m_batch_size); // LCOV_EXCL_LINE

    // Check the current times.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), [](const auto &t) { return !isfinite(t); })
        || std::any_of(m_time_lo.begin(), m_time_lo.end(), [](const auto &t) { return !isfinite(t); })) {
        throw std::invalid_argument(
            "Cannot invoke the propagate_until() function of an adaptive Taylor integrator in batch mode if "
            "one of the current times is not finite");
    }

    // Check the final times.
    if (std::any_of(ts.begin(), ts.end(), [](const auto &t) { return !isfinite(t); })) {
        throw std::invalid_argument("A non-finite time was passed to the propagate_until() function of an adaptive "
                                    "Taylor integrator in batch mode");
    }

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}"_format(m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_until() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_until() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // If with_c_out is true, we always need to write the Taylor coefficients.
    wtc = wtc || with_c_out;

    // These vectors are used in the construction of the continuous output.
    // If continuous output is not requested, they will remain empty.
    std::vector<T> c_out_tcs, c_out_times_hi, c_out_times_lo;
    if (with_c_out) {
        // Push in the starting time.
        c_out_times_hi.insert(c_out_times_hi.end(), m_time_hi.begin(), m_time_hi.end());
        c_out_times_lo.insert(c_out_times_lo.end(), m_time_lo.begin(), m_time_lo.end());
    }

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    // Compute the integration directions and init
    // the remaining times.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i] = ts[i] - dfloat<T>(m_time_hi[i], m_time_lo[i]);
        if (!isfinite(m_rem_time[i])) {
            throw std::invalid_argument("The final time passed to the propagate_until() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = (m_rem_time[i] >= T(0));
    }

    // Helper to create the continuous output object.
    auto make_c_out = [&]() -> std::optional<continuous_output_batch<T>> {
        if (with_c_out) {
            if (c_out_times_hi.size() / m_batch_size < 2u) {
                // NOTE: this means that no successful steps
                // were taken.
                return {};
            }

            // Construct the return value.
            continuous_output_batch<T> ret(m_llvm.make_similar());

            // Fill in the data.
            ret.m_batch_size = m_batch_size;
            ret.m_tcs = std::move(c_out_tcs);
            ret.m_times_hi = std::move(c_out_times_hi);
            ret.m_times_lo = std::move(c_out_times_lo);

            // Add padding to the times vectors to make the
            // vectorised upper_bound implementation well defined.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                ret.m_times_hi.push_back(m_t_dir[i] ? std::numeric_limits<T>::infinity()
                                                    : -std::numeric_limits<T>::infinity());
                ret.m_times_lo.push_back(T(0));
            }

            // Prepare the output vector.
            ret.m_output.resize(boost::numeric_cast<decltype(ret.m_output.size())>(m_dim * m_batch_size));

            // Prepare the temp time vector.
            ret.m_tmp_tm.resize(boost::numeric_cast<decltype(ret.m_tmp_tm.size())>(m_batch_size));

            // Add the continuous output function.
            ret.add_c_out_function(m_order, m_dim, m_high_accuracy);

            return std::optional{std::move(ret)};
        } else {
            return {};
        }
    };

    // Helper to update the continuous output data after a timestep.
    auto update_c_out = [&]() {
        if (with_c_out) {
#if !defined(NDEBUG)
            std::vector<dfloat<T>> prev_times;
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                prev_times.emplace_back(c_out_times_hi[c_out_times_hi.size() - m_batch_size + i],
                                        c_out_times_lo[c_out_times_lo.size() - m_batch_size + i]);
            }
#endif

            c_out_times_hi.insert(c_out_times_hi.end(), m_time_hi.begin(), m_time_hi.end());
            c_out_times_lo.insert(c_out_times_lo.end(), m_time_lo.begin(), m_time_lo.end());

#if !defined(NDEBUG)
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                const dfloat<T> new_time(c_out_times_hi[c_out_times_hi.size() - m_batch_size + i],
                                         c_out_times_lo[c_out_times_lo.size() - m_batch_size + i]);
                assert(isfinite(new_time));
                if (m_t_dir[i]) {
                    assert(!(new_time < prev_times[i]));
                } else {
                    assert(!(new_time > prev_times[i]));
                }
            }
#endif

            c_out_tcs.insert(c_out_tcs.end(), m_tc.begin(), m_tc.end());
        }
    };

    while (true) {
        // Compute the max integration times for this timestep.
        // NOTE: m_rem_time[i] is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            assert((m_rem_time[i] >= T(0)) == m_t_dir[i] || m_rem_time[i] == T(0)); // LCOV_EXCL_LINE

            // Compute the time limit.
            const auto dt_limit = m_t_dir[i] ? std::min(dfloat<T>(max_delta_ts[i]), m_rem_time[i])
                                             : std::max(dfloat<T>(-max_delta_ts[i]), m_rem_time[i]);

            // Store it.
            m_cur_max_delta_ts[i] = static_cast<T>(dt_limit);
        }

        // Run the integration timestep.
        // NOTE: if dt_limit is zero, step_impl() will always return time_limit.
        step_impl(m_cur_max_delta_ts, wtc);

        // Check the outcomes of the step for each batch element,
        // update the step counters, min_h/max_h and the remaining times
        // (if meaningful), and keep track of:
        // - the number of batch elements which reached the time limit,
        // - whether or not any non-finite state was detected,
        // - whether or not any stopping terminal event triggered.
        std::uint32_t n_done = 0;
        bool nfs_detected = false, ste_detected = false;

        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto [oc, h] = m_step_res[i];

            if (oc == taylor_outcome::err_nf_state) {
                // Non-finite state: flag it and do nothing else.
                nfs_detected = true;
            } else {
                // Step outcome is one of:
                // - success,
                // - time_limit,
                // - terminal event.

                // Update the local step counters.
                // NOTE: the local step counters increase only if we integrated
                // for a nonzero time.
                m_ts_count[i] += static_cast<std::size_t>(h != 0);

                // Update min_h/max_h only if the outcome is success (otherwise
                // the step was artificially clamped either by a time limit or
                // by a terminal event).
                if (oc == taylor_outcome::success) {
                    const auto abs_h = abs(h);
                    m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                    m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
                }

                // Flag if we encountered a terminal event.
                ste_detected = ste_detected || (oc > taylor_outcome::success && oc < taylor_outcome{0});

                // Check if this batch element is done.
                // NOTE: we check h == rem_time, instead of just
                // oc == time_limit, because clamping via max_delta_t
                // could also result in time_limit.
                const auto cur_done = (h == static_cast<T>(m_rem_time[i]));
                n_done += cur_done;

                // Update the remaining times.
                if (cur_done) {
                    assert(oc == taylor_outcome::time_limit);

                    // Force m_rem_time[i] to zero so that
                    // zero-length steps will be taken
                    // for all remaining iterations.
                    // NOTE: if m_rem_time[i] was previously set to zero, it
                    // will end up being repeatedly set to zero here. This
                    // should be harmless.
                    m_rem_time[i] = dfloat<T>(T(0));
                } else {
                    // NOTE: this should never flip the time direction of the
                    // integration for the same reasons as explained in the
                    // scalar implementation.
                    assert(abs(h) < abs(static_cast<T>(m_rem_time[i])));
                    m_rem_time[i] = ts[i] - dfloat<T>(m_time_hi[i], m_time_lo[i]);
                }
            }

            // Write into m_prop_res.
            m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
        }

        if (nfs_detected) {
            // At least 1 batch element generated a non-finite state. In this situation,
            // we do *not* want to execute the propagate() callback and we do *not* want
            // to update the continuous output. Just exit.
            return make_c_out();
        }

        // Update the continuous output.
        update_c_out();

        // Update the iteration counter.
        ++iter_counter;

        // Execute the propagate() callback, if applicable.
        if (cb && !cb(*this)) {
            // Change m_prop_res before exiting by setting all outcomes
            // to cb_stop regardless of the timestep outcome.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                std::get<0>(m_prop_res[i]) = taylor_outcome::cb_stop;
            }

            return make_c_out();
        }

        // We need to break out if either we reached the final time
        // for all batch elements, or we encountered at least 1 stopping
        // terminal event. In either case, m_prop_res was already set up
        // in the loop where we checked the outcomes.
        if (n_done == m_batch_size || ste_detected) {
            return make_c_out();
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger (modulo wraparound) as by this point we are
        // sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            // We reached the max_steps limit: set the outcome for all batch elements
            // to step_limit.
            // NOTE: this is the same logic adopted when the integration is stopped
            // by the callback (see above).
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                std::get<0>(m_prop_res[i]) = taylor_outcome::step_limit;
            }

            return make_c_out();
        }
    }

    // LCOV_EXCL_START
    assert(false);

    return {};
    // LCOV_EXCL_STOP
}

template <typename T>
std::optional<continuous_output_batch<T>> taylor_adaptive_batch_impl<T>::propagate_until_impl(
    const std::vector<T> &ts, std::size_t max_steps, const std::vector<T> &max_delta_ts,
    const std::function<bool(taylor_adaptive_batch_impl &)> &cb, bool wtc, bool with_c_out)
{
    // Check the dimensionality of ts.
    if (ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time limits specified in a Taylor integrator in batch mode: the "
            "batch size is {}, but the number of specified time limits is {}"_format(m_batch_size, ts.size()));
    }

    // NOTE: re-use m_pfor_ts as tmp storage.
    assert(m_pfor_ts.size() == m_batch_size); // LCOV_EXCL_LINE
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_pfor_ts[i] = dfloat<T>(ts[i]);
    }

    // NOTE: max_delta_ts is checked in the other propagate_until_impl() overload.
    return propagate_until_impl(m_pfor_ts, max_steps, max_delta_ts, std::move(cb), wtc, with_c_out);
}

template <typename T>
std::vector<T>
taylor_adaptive_batch_impl<T>::propagate_grid_impl(const std::vector<T> &grid, std::size_t max_steps,
                                                   const std::vector<T> &max_delta_ts,
                                                   const std::function<bool(taylor_adaptive_batch_impl &)> &cb)
{
    using std::abs;
    using std::isnan;

    // Helper to detect if an input value is nonfinite.
    auto is_nf = [](const T &t) {
        using std::isfinite;
        return !isfinite(t);
    };

    if (grid.empty()) {
        throw std::invalid_argument(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if the time grid is empty");
    }

    // Check that the grid size is a multiple of m_batch_size.
    if (grid.size() % m_batch_size != 0u) {
        throw std::invalid_argument(
            "Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
            "the grid has a size of {}, which is not a multiple of the batch size ({})"_format(grid.size(),
                                                                                               m_batch_size));
    }

    // Check the current time coordinates.
    if (std::any_of(m_time_hi.begin(), m_time_hi.end(), is_nf)
        || std::any_of(m_time_lo.begin(), m_time_lo.end(), is_nf)) {
        throw std::invalid_argument("Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if "
                                    "the current time is not finite");
    }

    // Check max_delta_ts.
    if (max_delta_ts.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is {}, "
            "but the number of specified timesteps is {}"_format(m_batch_size, max_delta_ts.size()));
    }
    for (const auto &dt : max_delta_ts) {
        if (isnan(dt)) {
            throw std::invalid_argument("A nan max_delta_t was passed to the propagate_grid() function of an adaptive "
                                        "Taylor integrator in batch mode");
        }
        if (dt <= 0) {
            throw std::invalid_argument("A non-positive max_delta_t was passed to the propagate_grid() function of an "
                                        "adaptive Taylor integrator in batch mode");
        }
    }

    // The number of grid points.
    const auto n_grid_points = grid.size() / m_batch_size;

    // Pointer to the grid data.
    const auto *const grid_ptr = grid.data();

    // Check the input grid points.
    constexpr auto nf_err_msg
        = "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode";
    constexpr auto ig_err_msg = "A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                "Taylor integrator in batch mode";

    // Check the first point.
    if (std::any_of(grid_ptr, grid_ptr + m_batch_size, is_nf)) {
        throw std::invalid_argument(nf_err_msg);
    }
    if (n_grid_points > 1u) {
        // Establish the direction of the grid from
        // the first two batches of points.
        if (std::any_of(grid_ptr + m_batch_size, grid_ptr + m_batch_size + m_batch_size, is_nf)) {
            throw std::invalid_argument(nf_err_msg);
        }
        if (grid_ptr[m_batch_size] == grid_ptr[0]) {
            throw std::invalid_argument(ig_err_msg);
        }

        const auto grid_direction = grid_ptr[m_batch_size] > grid_ptr[0];
        for (std::uint32_t i = 1; i < m_batch_size; ++i) {
            if ((grid_ptr[m_batch_size + i] > grid_ptr[i]) != grid_direction) {
                throw std::invalid_argument(ig_err_msg);
            }
        }

        // Check that the remaining points are finite and that
        // they are ordered monotonically.
        for (decltype(grid.size()) i = 2; i < n_grid_points; ++i) {
            if (std::any_of(grid_ptr + i * m_batch_size, grid_ptr + (i + 1u) * m_batch_size, is_nf)) {
                throw std::invalid_argument(nf_err_msg);
            }

            if (std::any_of(
                    grid_ptr + i * m_batch_size, grid_ptr + (i + 1u) * m_batch_size,
                    [this, grid_direction](const T &t) { return (t > *(&t - m_batch_size)) != grid_direction; })) {
                throw std::invalid_argument(ig_err_msg);
            }
        }
    }

    // Pre-allocate the return value.
    std::vector<T> retval;
    // LCOV_EXCL_START
    if (get_dim() > std::numeric_limits<decltype(retval.size())>::max() / grid.size()) {
        throw std::overflow_error("Overflow detected in the creation of the return value of propagate_grid() in an "
                                  "adaptive Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP
    // NOTE: fill with NaNs, so that the missing entries
    // are signalled with NaN if we exit early.
    retval.resize(grid.size() * get_dim(), std::numeric_limits<T>::quiet_NaN());

    // Propagate the system up to the first batch of grid points.
    std::vector<T> pgrid_tmp;
    pgrid_tmp.resize(boost::numeric_cast<decltype(pgrid_tmp.size())>(m_batch_size));
    std::copy(grid_ptr, grid_ptr + m_batch_size, pgrid_tmp.begin());
    // NOTE: we pass write_tc = true because some grid
    // points after the first one might end up being
    // calculated via dense output *before*
    // taking additional steps, and, in such case, we
    // must ensure the TCs are up to date.
    // NOTE: if the integrator time is already at grid[0],
    // then propagate_until() will take a zero timestep,
    // resulting in m_last_h also going to zero and no event
    // detection being performed.
    // NOTE: use the same max_steps for the initial propagation,
    // and don't pass the callback.
    propagate_until(pgrid_tmp, kw::max_delta_t = max_delta_ts, kw::max_steps = max_steps, kw::write_tc = true);

    // Check the result of the integration.
    if (std::any_of(m_prop_res.begin(), m_prop_res.end(), [](const auto &t) {
            // Check if any outcome is not time_limit.
            return std::get<0>(t) != taylor_outcome::time_limit;
        })) {
        // NOTE: for consistency with the scalar implementation,
        // keep the outcomes from propagate_until() but we reset
        // min/max h and the step counter.
        for (auto &[_, min_h, max_h, ts_count] : m_prop_res) {
            min_h = std::numeric_limits<T>::infinity();
            max_h = 0;
            ts_count = 0;
        }

        return retval;
    }

    // Add the first result to retval.
    std::copy(m_state.begin(), m_state.end(), retval.begin());

    // Init the remaining times and directions.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_rem_time[i] = grid_ptr[(n_grid_points - 1u) * m_batch_size + i] - dfloat<T>(m_time_hi[i], m_time_lo[i]);

        // Check it.
        if (!isfinite(m_rem_time[i])) {
            throw std::invalid_argument("The final time passed to the propagate_grid() function of an adaptive Taylor "
                                        "integrator in batch mode results in an overflow condition");
        }

        m_t_dir[i] = (m_rem_time[i] >= T(0));
    }

    // Reset the counters and the min/max abs(h) vectors.
    std::size_t iter_counter = 0;
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_ts_count[i] = 0;
        m_min_abs_h[i] = std::numeric_limits<T>::infinity();
        m_max_abs_h[i] = 0;
    }

    // NOTE: in general, an integration timestep will cover a different number
    // of grid points for each batch element. We thus need to track the grid
    // index separately for each batch element. We will start with index
    // 1 for all batch elements, since all batch elements have been propagated to
    // index 0 already.
    std::vector<decltype(grid.size())> cur_grid_idx(
        boost::numeric_cast<typename std::vector<decltype(grid.size())>::size_type>(m_batch_size), 1);

    // Vectors to keep track of the time range of the last taken timestep.
    std::vector<dfloat<T>> t0(boost::numeric_cast<typename std::vector<dfloat<T>>::size_type>(m_batch_size)), t1(t0);

    // Vector of flags to keep track of the batch elements
    // we can compute dense output for.
    std::vector<unsigned> dflags(boost::numeric_cast<std::vector<unsigned>::size_type>(m_batch_size));

    // NOTE: loop until we have processed all grid points
    // for all batch elements.
    auto cont_cond = [n_grid_points, &cur_grid_idx]() {
        return std::any_of(cur_grid_idx.begin(), cur_grid_idx.end(),
                           [n_grid_points](auto idx) { return idx < n_grid_points; });
    };

    while (cont_cond()) {
        // Establish the time ranges of the last
        // taken timestep.
        // NOTE: t0 < t1.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const dfloat<T> cur_time(m_time_hi[i], m_time_lo[i]), cmp = cur_time - m_last_h[i];

            t0[i] = std::min(cur_time, cmp);
            t1[i] = std::max(cur_time, cmp);
        }

        // Reset dflags.
        std::fill(dflags.begin(), dflags.end(), 1u);

        // Compute the state of the system via dense output for as many grid
        // points as possible, i.e., as long as the grid times
        // fall within the validity range for the dense output of at least
        // one batch element.
        while (true) {
            // Establish and count for which batch elements we
            // can still compute dense output.
            std::uint32_t counter = 0;
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                // Fetch the grid index for the current batch element.
                const auto gidx = cur_grid_idx[i];

                if (dflags[i] && gidx < n_grid_points) {
                    // The current batch element has not been eliminated
                    // yet from the candidate list and it still has grid
                    // points available. Determine if the current grid point
                    // falls within the validity domain for the dense output.
                    // NOTE: if we are at the last timestep for this batch
                    // element, force processing of all remaining grid points.
                    // We do this to avoid numerical issues when deciding if
                    // he last grid point falls within the range of validity
                    // of the dense output.
                    const auto idx = gidx * m_batch_size + i;
                    const auto d_avail
                        = (grid_ptr[idx] >= t0[i] && grid_ptr[idx] <= t1[i]) || (m_rem_time[i] == dfloat<T>(T(0)));
                    dflags[i] = d_avail;
                    counter += d_avail;

                    // Copy over the grid point to pgrid_tmp regardless
                    // of whether d_avail is true or false.
                    pgrid_tmp[i] = grid_ptr[idx];
                } else {
                    // Either the batch element had already been eliminated
                    // previously, or there are no more grid points available.
                    // Make sure the batch element is marked as eliminated.
                    dflags[i] = 0;
                }
            }

            if (counter == 0u) {
                // Cannot use dense output on any of the batch elements,
                // need to take another step.
                break;
            }

            // Compute the dense output.
            update_d_output(pgrid_tmp);

            // Add the results to retval and bump up the values in cur_grid_idx.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                if (dflags[i] != 0u) {
                    const auto gidx = cur_grid_idx[i];

                    for (std::uint32_t j = 0; j < m_dim; ++j) {
                        retval[gidx * m_batch_size * m_dim + j * m_batch_size + i] = m_d_out[j * m_batch_size + i];
                    }

                    assert(cur_grid_idx[i] < n_grid_points); // LCOV_EXCL_LINE
                    ++cur_grid_idx[i];
                }
            }

            // Check if we exhausted all grid points for all batch elements.
            if (!cont_cond()) {
                break;
            }
        }

        // Check if we exhausted all grid points for all batch elements.
        if (!cont_cond()) {
            break;
        }

        // Take the next step, making sure to write the Taylor coefficients
        // and to cap the timestep size so that we don't go past the
        // last grid point and we don't use a timestep exceeding max_delta_t.
        // NOTE: m_rem_time is guaranteed to be finite: we check it explicitly above
        // and we keep on decreasing its magnitude at the following iterations.
        // If some non-finite state/time is generated in
        // the step function, the integration will be stopped.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            // Max delta_t for the current batch element.
            const auto max_delta_t = max_delta_ts[i];

            // Compute the step limit for the current batch element.
            assert((m_rem_time[i] >= T(0)) == m_t_dir[i] || m_rem_time[i] == T(0)); // LCOV_EXCL_LINE
            const auto dt_limit = m_t_dir[i] ? std::min(dfloat<T>(max_delta_t), m_rem_time[i])
                                             : std::max(dfloat<T>(-max_delta_t), m_rem_time[i]);

            pgrid_tmp[i] = static_cast<T>(dt_limit);
        }
        step_impl(pgrid_tmp, true);

        // Check the result of the integration.
        if (std::any_of(m_step_res.begin(), m_step_res.end(), [](const auto &t) {
                // Something went wrong in the propagation of the timestep, or we reached
                // a stopping terminal event.
                const auto oc = std::get<0>(t);
                return oc != taylor_outcome::success && oc != taylor_outcome::time_limit && oc < taylor_outcome{0};
            })) {
            // Setup m_prop_res before exiting.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                const auto [oc, h] = m_step_res[i];
                if (oc > taylor_outcome::success) {
                    // In case of a stopping terminal event, we still want to update
                    // the step counter.
                    m_ts_count[i] += static_cast<std::size_t>(h != 0);
                }

                m_prop_res[i] = std::tuple{oc, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
            }

            return retval;
        }

        // Update the number of iterations.
        ++iter_counter;

        // Update m_rem_time, the local step counters, min_h/max_h.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const auto [oc, h] = m_step_res[i];

            // NOTE: the local step counters increase only if we integrated
            // for a nonzero time.
            m_ts_count[i] += static_cast<std::size_t>(h != 0);

            // Update the remaining time.
            // NOTE: if static_cast<T>(m_rem_time[i]) was used as a timestep,
            // it means that we hit the time limit. Force rem_time to zero
            // to signal this, so that zero-length steps will be taken
            // for all remaining iterations, thus always triggering the
            // time_limit outcome. A zero m_rem_time[i]
            // will also force the processing of all remaining grid points.
            // NOTE: if m_rem_time[i] was previously set to zero, it
            // will end up being repeatedly set to zero here. This
            // should be harmless.
            if (h == static_cast<T>(m_rem_time[i])) {
                assert(oc == taylor_outcome::time_limit); // LCOV_EXCL_LINE
                m_rem_time[i] = dfloat<T>(T(0));
            } else {
                m_rem_time[i]
                    = grid_ptr[(n_grid_points - 1u) * m_batch_size + i] - dfloat<T>(m_time_hi[i], m_time_lo[i]);
            }

            // Update min_h/max_h, but only if the outcome is success (otherwise
            // the step was artificially clamped either by a time limit or
            // by a continuing terminal event).
            if (oc == taylor_outcome::success) {
                const auto abs_h = abs(h);
                m_min_abs_h[i] = std::min(m_min_abs_h[i], abs_h);
                m_max_abs_h[i] = std::max(m_max_abs_h[i], abs_h);
            }
        }

        // Step successful: invoke the callback, if needed.
        if (cb && !cb(*this)) {
            // Setup m_prop_res before exiting. We set all outcomes
            // to cb_stop regardless of the timestep outcome.
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                m_prop_res[i] = std::tuple{taylor_outcome::cb_stop, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
            }

            return retval;
        }

        // Check the iteration limit.
        // NOTE: if max_steps is 0 (i.e., no limit on the number of steps),
        // then this condition will never trigger as by this point we are
        // sure iter_counter is at least 1.
        if (iter_counter == max_steps) {
            // We reached the max_steps limit: set the outcome for all batch elements
            // to step_limit.
            // NOTE: this is the same logic adopted when the integration is stopped
            // by the callback (see above).
            for (std::uint32_t i = 0; i < m_batch_size; ++i) {
                m_prop_res[i] = std::tuple{taylor_outcome::step_limit, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
            }

            return retval;
        }
    }

    // Everything went fine, set all outcomes to time_limit.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        m_prop_res[i] = std::tuple{taylor_outcome::time_limit, m_min_abs_h[i], m_max_abs_h[i], m_ts_count[i]};
    }

    return retval;
}

template <typename T>
const llvm_state &taylor_adaptive_batch_impl<T>::get_llvm_state() const
{
    return m_llvm;
}

template <typename T>
const taylor_dc_t &taylor_adaptive_batch_impl<T>::get_decomposition() const
{
    return m_dc;
}

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_order() const
{
    return m_order;
}

template <typename T>
T taylor_adaptive_batch_impl<T>::get_tol() const
{
    return m_tol;
}

template <typename T>
bool taylor_adaptive_batch_impl<T>::get_high_accuracy() const
{
    return m_high_accuracy;
}

template <typename T>
bool taylor_adaptive_batch_impl<T>::get_compact_mode() const
{
    return m_compact_mode;
}

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_batch_size() const
{
    return m_batch_size;
}

template <typename T>
std::uint32_t taylor_adaptive_batch_impl<T>::get_dim() const
{
    return m_dim;
}

template <typename T>
const std::vector<T> &taylor_adaptive_batch_impl<T>::update_d_output(const std::vector<T> &time, bool rel_time)
{
    // Check the dimensionality of time.
    if (time.size() != m_batch_size) {
        throw std::invalid_argument(
            "Invalid number of time coordinates specified for the dense output in a Taylor integrator in batch "
            "mode: the batch size is {}, but the number of time coordinates is {}"_format(m_batch_size, time.size()));
    }

    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = m_last_h[i] + time[i];
        }
    } else {
        // Absolute time coordinate.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = static_cast<T>(time[i] - (dfloat<T>(m_time_hi[i], m_time_lo[i]) - m_last_h[i]));
        }
    }

    m_d_out_f(m_d_out.data(), m_tc.data(), m_d_out_time.data());

    return m_d_out;
}

// NOTE: there's some overlap with the code from the other overload here.
template <typename T>
const std::vector<T> &taylor_adaptive_batch_impl<T>::update_d_output(T time, bool rel_time)
{
    // NOTE: "time" needs to be translated
    // because m_d_out_f expects a time coordinate
    // with respect to the starting time t0 of
    // the *previous* timestep.
    if (rel_time) {
        // Time coordinate relative to the current time.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = m_last_h[i] + time;
        }
    } else {
        // Absolute time coordinate.
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            m_d_out_time[i] = static_cast<T>(time - (dfloat<T>(m_time_hi[i], m_time_lo[i]) - m_last_h[i]));
        }
    }

    m_d_out_f(m_d_out.data(), m_tc.data(), m_d_out_time.data());

    return m_d_out;
}

template <typename T>
void taylor_adaptive_batch_impl<T>::reset_cooldowns()
{
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        reset_cooldowns(i);
    }
}

template <typename T>
void taylor_adaptive_batch_impl<T>::reset_cooldowns(std::uint32_t i)
{
    if (!m_ed_data) {
        throw std::invalid_argument("No events were defined for this integrator");
    }

    if (i >= m_batch_size) {
        throw std::invalid_argument(
            "Cannot reset the cooldowns at batch index {}: the batch size for this integrator is only {}"_format(
                i, m_batch_size));
    }

    for (auto &cd : m_ed_data->m_te_cooldowns[i]) {
        cd.reset();
    }
}

// Explicit instantiation of the batch implementation classes.
template class taylor_adaptive_batch_impl<double>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<double>, std::uint32_t, std::vector<double>, double, bool, bool,
    std::vector<double>, std::vector<t_event_t>, std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<double>, std::uint32_t, std::vector<double>,
    double, bool, bool, std::vector<double>, std::vector<t_event_t>, std::vector<nt_event_t>);

template class taylor_adaptive_batch_impl<long double>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<long double>, std::uint32_t, std::vector<long double>, long double,
    bool, bool, std::vector<long double>, std::vector<t_event_t>, std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<long double>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<long double>, std::uint32_t,
    std::vector<long double>, long double, bool, bool, std::vector<long double>, std::vector<t_event_t>,
    std::vector<nt_event_t>);

#if defined(HEYOKA_HAVE_REAL128)

template class taylor_adaptive_batch_impl<mppp::real128>;

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(
    const std::vector<expression> &, std::vector<mppp::real128>, std::uint32_t, std::vector<mppp::real128>,
    mppp::real128, bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>, std::vector<nt_event_t>);

template HEYOKA_DLL_PUBLIC void taylor_adaptive_batch_impl<mppp::real128>::finalise_ctor_impl(
    const std::vector<std::pair<expression, expression>> &, std::vector<mppp::real128>, std::uint32_t,
    std::vector<mppp::real128>, mppp::real128, bool, bool, std::vector<mppp::real128>, std::vector<t_event_t>,
    std::vector<nt_event_t>);

#endif

} // namespace detail

namespace detail
{

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
                         const std::vector<expression> &sv_funcs)
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

    // Prepare the function prototype. The first argument is a float pointer to the in/out array,
    // the second argument a const float pointer to the pars, the third argument
    // a float pointer to the time. These arrays cannot overlap.
    auto *fp_t = to_llvm_type<T>(s.context());
    std::vector<llvm::Type *> fargs(3, llvm::PointerType::getUnqual(fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &s.module());
    if (f == nullptr) {
        throw std::invalid_argument(
            "Unable to create a function for the computation of the jet of Taylor derivatives with name '{}'"_format(
                name));
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
    auto *bb = llvm::BasicBlock::Create(s.context(), "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Compute the jet of derivatives.
    auto diff_variant = taylor_compute_jet<T>(s, in_out, par_ptr, time_ptr, dc, sv_funcs_dc, n_eq, n_uvars, order,
                                              batch_size, compact_mode, high_accuracy);

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

    // Run the optimisation pass.
    s.optimise();

    return dc;
}

} // namespace

} // namespace detail

taylor_dc_t taylor_add_jet_dbl(llvm_state &s, const std::string &name, const std::vector<expression> &sys,
                               std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                               const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<double>(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
}

taylor_dc_t taylor_add_jet_ldbl(llvm_state &s, const std::string &name, const std::vector<expression> &sys,
                                std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                                const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<long double>(s, name, sys, order, batch_size, high_accuracy, compact_mode,
                                                    sv_funcs);
}

#if defined(HEYOKA_HAVE_REAL128)

taylor_dc_t taylor_add_jet_f128(llvm_state &s, const std::string &name, const std::vector<expression> &sys,
                                std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                                const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, sys, order, batch_size, high_accuracy, compact_mode,
                                                      sv_funcs);
}

#endif

taylor_dc_t taylor_add_jet_dbl(llvm_state &s, const std::string &name,
                               const std::vector<std::pair<expression, expression>> &sys, std::uint32_t order,
                               std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                               const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<double>(s, name, sys, order, batch_size, high_accuracy, compact_mode, sv_funcs);
}

taylor_dc_t taylor_add_jet_ldbl(llvm_state &s, const std::string &name,
                                const std::vector<std::pair<expression, expression>> &sys, std::uint32_t order,
                                std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                                const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<long double>(s, name, sys, order, batch_size, high_accuracy, compact_mode,
                                                    sv_funcs);
}

#if defined(HEYOKA_HAVE_REAL128)

taylor_dc_t taylor_add_jet_f128(llvm_state &s, const std::string &name,
                                const std::vector<std::pair<expression, expression>> &sys, std::uint32_t order,
                                std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
                                const std::vector<expression> &sv_funcs)
{
    return detail::taylor_add_jet_impl<mppp::real128>(s, name, sys, order, batch_size, high_accuracy, compact_mode,
                                                      sv_funcs);
}

#endif

} // namespace heyoka
