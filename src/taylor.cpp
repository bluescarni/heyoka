// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <heyoka/binary_operator.hpp>
#include <heyoka/detail/assert_nonnull_ret.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

// Transform in-place ex by decomposition, appending the
// result of the decomposition to u_vars_defs.
// The return value is the index, in u_vars_defs,
// which corresponds to the decomposed version of ex.
// If the return value is zero, ex was not decomposed.
// NOTE: this will render ex unusable.
std::vector<expression>::size_type taylor_decompose_in_place(expression &&ex, std::vector<expression> &u_vars_defs)
{
    return std::visit(
        [&u_vars_defs](auto &&v) { return taylor_decompose_in_place(std::forward<decltype(v)>(v), u_vars_defs); },
        std::move(ex.value()));
}

namespace detail
{

namespace
{

// Simplify a Taylor decomposition by removing
// common subexpressions.
std::vector<expression> taylor_decompose_cse(std::vector<expression> &v_ex, std::vector<expression>::size_type n_eq)
{
    // A Taylor decomposition is supposed
    // to have n_eq variables at the beginning,
    // n_eq variables at the end and possibly
    // extra variables in the middle.
    assert(v_ex.size() >= n_eq * 2u);

    using idx_t = std::vector<expression>::size_type;

    std::vector<expression> retval;

    // expression -> idx map. This will end up containing
    // all the unique expressions from v_ex, and it will
    // map them to their indices in retval (which will
    // in general differ from their indices in v_ex).
    std::unordered_map<expression, idx_t> ex_map;

    // Map for the renaming of u variables
    // in the expressions.
    std::unordered_map<std::string, std::string> uvars_rename;

    // Add the definitions of the first n_eq
    // variables in terms of u variables.
    // No need to modify anything here.
    for (idx_t i = 0; i < n_eq; ++i) {
        retval.emplace_back(std::move(v_ex[i]));
    }

    for (auto i = n_eq; i < v_ex.size() - n_eq; ++i) {
        auto &ex = v_ex[i];

        // Rename the u variables in ex.
        rename_variables(ex, uvars_rename);

        if (auto it = ex_map.find(ex); it == ex_map.end()) {
            // This is the first occurrence of ex in the
            // decomposition. Add it to retval.
            retval.emplace_back(ex);

            // Add ex to ex_map, mapping it to
            // the index it corresponds to in retval
            // (let's call it j).
            ex_map.emplace(std::move(ex), retval.size() - 1u);

            // Update uvars_rename. This will ensure that
            // occurrences of the variable 'u_i' in the next
            // elements of v_ex will be renamed to 'u_j'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace("u_" + li_to_string(i), "u_" + li_to_string(retval.size() - 1u));
            assert(res.second);
        } else {
            // ex is a redundant expression. This means
            // that it already appears in retval at index
            // it->second. Don't add anything to retval,
            // and remap the variable name 'u_i' to
            // 'u_{it->second}'.
            [[maybe_unused]] const auto res
                = uvars_rename.emplace("u_" + li_to_string(i), "u_" + li_to_string(it->second));
            assert(res.second);
        }
    }

    // Handle the derivatives of the state variables at the
    // end of the decomposition. We just need to ensure that
    // the u variables in their definitions are renamed with
    // the new indices.
    for (auto i = v_ex.size() - n_eq; i < v_ex.size(); ++i) {
        auto &ex = v_ex[i];

        rename_variables(ex, uvars_rename);

        retval.emplace_back(std::move(ex));
    }

    return retval;
}

#if !defined(NDEBUG)

// Helper to verify a Taylor decomposition.
void verify_taylor_dec(const std::vector<expression> &orig, const std::vector<expression> &dc)
{
    using idx_t = std::vector<expression>::size_type;

    const auto n_eq = orig.size();

    assert(dc.size() >= n_eq * 2u);

    // The first n_eq expressions of u variables
    // must be just variables.
    for (idx_t i = 0; i < n_eq; ++i) {
        assert(std::holds_alternative<variable>(dc[i].value()));
    }

    // From n_eq to dc.size() - n_eq, the expressions
    // must contain variables only in the u_n form,
    // where n < i.
    for (auto i = n_eq; i < dc.size() - n_eq; ++i) {
        for (const auto &var : get_variables(dc[i])) {
            assert(var.rfind("u_", 0) == 0);
            assert(uname_to_index(var) < i);
        }
    }

    // From dc.size() - n_eq to dc.size(), the expressions
    // must be variables in the u_n form,
    // where n < i.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        std::visit(
            [i](const auto &v) {
                if constexpr (std::is_same_v<detail::uncvref_t<decltype(v)>, variable>) {
                    assert(v.name().rfind("u_", 0) == 0);
                    assert(uname_to_index(v.name()) < i);
                } else {
                    assert(false);
                }
            },
            dc[i].value());
    }

    std::unordered_map<std::string, expression> subs_map;

    // For each u variable, expand its definition
    // in terms of state variables or other u variables,
    // and store it in subs_map.
    for (idx_t i = 0; i < dc.size() - n_eq; ++i) {
        subs_map.emplace("u_" + li_to_string(i), subs(dc[i], subs_map));
    }

    // Reconstruct the right-hand sides of the system
    // and compare them to the original ones.
    for (auto i = dc.size() - n_eq; i < dc.size(); ++i) {
        assert(subs(dc[i], subs_map) == orig[i - (dc.size() - n_eq)]);
    }
}

#endif

} // namespace

} // namespace detail

std::vector<expression> taylor_decompose(std::vector<expression> v_ex)
{
    if (v_ex.empty()) {
        throw std::invalid_argument("Cannot decompose a system of zero equations");
    }

    // Determine the variables in the system of equations.
    std::vector<std::string> vars;
    for (const auto &ex : v_ex) {
        auto ex_vars = get_variables(ex);
        vars.insert(vars.end(), std::make_move_iterator(ex_vars.begin()), std::make_move_iterator(ex_vars.end()));
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    }

    if (vars.size() != v_ex.size()) {
        throw std::invalid_argument("The number of variables (" + std::to_string(vars.size())
                                    + ") differs from the number of equations (" + std::to_string(v_ex.size()) + ")");
    }

    // Cache the number of equations/variables
    // for later use.
    const auto n_eq = v_ex.size();

    // Create the map for renaming the variables to u_i.
    // The renaming will be done in alphabetical order.
    std::unordered_map<std::string, std::string> repl_map;
    for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
        [[maybe_unused]] const auto eres = repl_map.emplace(vars[i], "u_" + detail::li_to_string(i));
        assert(eres.second);
    }

#if !defined(NDEBUG)
    // Store a copy of the original system for checking later.
    const auto orig_v_ex = v_ex;
#endif

    // Rename the variables in the original equations.
    for (auto &ex : v_ex) {
        rename_variables(ex, repl_map);
    }

    // Init the vector containing the definitions
    // of the u variables. It begins with a list
    // of the original variables of the system.
    std::vector<expression> u_vars_defs;
    for (const auto &var : vars) {
        u_vars_defs.emplace_back(variable{var});
    }

    // Create a copy of the original equations in terms of u variables.
    // We will be reusing this below.
    auto v_ex_copy = v_ex;

    // Run the decomposition on each equation.
    for (decltype(v_ex.size()) i = 0; i < v_ex.size(); ++i) {
        // Decompose the current equation.
        if (const auto dres = taylor_decompose_in_place(std::move(v_ex[i]), u_vars_defs)) {
            // NOTE: if the equation was decomposed
            // (that is, it is not constant or a single variable),
            // we have to update the original definition
            // of the equation in v_ex_copy
            // so that it points to the u variable
            // that now represents it.
            v_ex_copy[i] = expression{variable{"u_" + detail::li_to_string(dres)}};
        }
    }

    // Append the (possibly updated) definitions of the diff equations
    // in terms of u variables.
    for (auto &ex : v_ex_copy) {
        u_vars_defs.emplace_back(std::move(ex));
    }

#if !defined(NDEBUG)
    // Verify the decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
#endif

    // Simplify the decomposition.
    u_vars_defs = detail::taylor_decompose_cse(u_vars_defs, n_eq);

#if !defined(NDEBUG)
    // Verify the simplified decomposition.
    detail::verify_taylor_dec(orig_v_ex, u_vars_defs);
#endif

    return u_vars_defs;
}

llvm::Value *taylor_init_dbl(llvm_state &s, const expression &e, llvm::Value *arr)
{
    heyoka_assert_nonnull_ret(
        std::visit([&s, arr](const auto &arg) { return taylor_init_dbl(s, arg, arr); }, e.value()));
}

llvm::Value *taylor_init_ldbl(llvm_state &s, const expression &e, llvm::Value *arr)
{
    heyoka_assert_nonnull_ret(
        std::visit([&s, arr](const auto &arg) { return taylor_init_ldbl(s, arg, arr); }, e.value()));
}

llvm::Function *taylor_diff_dbl(llvm_state &s, const expression &e, std::uint32_t idx, const std::string &name,
                                std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    auto visitor = [&s, idx, &name, n_uvars, &cd_uvars](const auto &v) -> llvm::Function * {
        using type = detail::uncvref_t<decltype(v)>;

        if constexpr (std::is_same_v<type, binary_operator> || std::is_same_v<type, function>) {
            return taylor_diff_dbl(s, v, idx, name, n_uvars, cd_uvars);
        } else {
            throw std::invalid_argument("Taylor derivatives can be computed only for binary operators or functions");
        }
    };

    heyoka_assert_nonnull_ret(std::visit(visitor, e.value()));
}

llvm::Function *taylor_diff_ldbl(llvm_state &s, const expression &e, std::uint32_t idx, const std::string &name,
                                 std::uint32_t n_uvars, const std::unordered_map<std::uint32_t, number> &cd_uvars)
{
    auto visitor = [&s, idx, &name, n_uvars, &cd_uvars](const auto &v) -> llvm::Function * {
        using type = detail::uncvref_t<decltype(v)>;

        if constexpr (std::is_same_v<type, binary_operator> || std::is_same_v<type, function>) {
            return taylor_diff_ldbl(s, v, idx, name, n_uvars, cd_uvars);
        } else {
            throw std::invalid_argument("Taylor derivatives can be computed only for binary operators or functions");
        }
    };

    heyoka_assert_nonnull_ret(std::visit(visitor, e.value()));
}

namespace detail
{

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(std::vector<expression> sys, std::vector<T> state, T time, T rtol, T atol,
                                              unsigned opt_level)
    : m_state(std::move(state)), m_time(time), m_rtol(rtol),
      m_atol(atol), m_llvm{"adaptive taylor integrator", opt_level}
{
    // Check input params.
    if (std::any_of(m_state.begin(), m_state.end(), [](const auto &x) { return !std::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite value was detected in the initial state of an adaptive Taylor integrator");
    }

    if (m_state.size() != sys.size()) {
        throw std::invalid_argument("Inconsistent sizes detected in the initialization of an adaptive Taylor "
                                    "integrator: the state vector has a size of "
                                    + std::to_string(m_state.size()) + ", while the number of equations is "
                                    + std::to_string(sys.size()));
    }

    if (!std::isfinite(m_time)) {
        throw std::invalid_argument("Cannot initialise an adaptive Taylor integrator with a non-finite initial time");
    }

    if (!std::isfinite(m_rtol) || m_rtol <= 0) {
        throw std::invalid_argument(
            "The relative tolerance in an adaptive Taylor integrator must be finite and positive, but it is "
            + std::to_string(m_rtol) + " instead");
    }

    if (!std::isfinite(m_atol) || m_atol <= 0) {
        throw std::invalid_argument(
            "The absolute tolerance in an adaptive Taylor integrator must be finite and positive, but it is "
            + std::to_string(m_atol) + " instead");
    }

    // Compute the max order for the integration.
    const auto mo_r = std::ceil(-std::log(m_rtol) / 2 + 1);
    const auto mo_a = std::ceil(-std::log(m_atol) / 2 + 1);
    if (!std::isfinite(mo_r) || !std::isfinite(mo_a)) {
        throw std::invalid_argument(
            "The computation of the max Taylor order in an adaptive Taylor integrator produced non-finite values");
    }
    // NOTE: make sure the max order is at least 2.
    const auto mo_f = std::max(T(2), std::max(mo_r, mo_a));
    // NOTE: static cast is safe because we now that T is at least
    // a double-precision IEEE type.
    if (mo_f > static_cast<T>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "The computation of the max Taylor order in an adaptive Taylor integrator produced the value "
            + std::to_string(mo_f) + ", which results in an overflow condition");
    }
    m_max_order = static_cast<std::uint32_t>(mo_f);

    // Record the number of variables
    // before consuming sys.
    const auto n_vars = sys.size();

    // Add the function for computing the jet
    // of derivatives.
    if constexpr (std::is_same_v<T, double>) {
        m_dc = m_llvm.add_taylor_jet_dbl("jet", std::move(sys), m_max_order);
    } else if constexpr (std::is_same_v<T, long double>) {
        m_dc = m_llvm.add_taylor_jet_ldbl("jet", std::move(sys), m_max_order);
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }

    // Store the IR before compiling.
    m_ir = m_llvm.dump();

    // Run the jit.
    m_llvm.compile();

    // Fetch the compiled function for computing
    // the jet of derivatives.
    if constexpr (std::is_same_v<T, double>) {
        m_jet_f = m_llvm.fetch_taylor_jet_dbl("jet");
    } else if constexpr (std::is_same_v<T, long double>) {
        m_jet_f = m_llvm.fetch_taylor_jet_ldbl("jet");
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }

    // Init the jet vector. Its maximum size is n_vars * (max_order + 1).
    // NOTE: n_vars must be nonzero because we successfully
    // created a Taylor jet function from sys.
    using jet_size_t = decltype(m_jet.size());
    if (m_max_order >= std::numeric_limits<jet_size_t>::max()
        || (static_cast<jet_size_t>(m_max_order) + 1u) > std::numeric_limits<jet_size_t>::max() / n_vars) {
        throw std::overflow_error("The computation of the size of the jet of derivatives in an adaptive Taylor "
                                  "integrator resulted in an overflow condition");
    }
    m_jet.resize((static_cast<jet_size_t>(m_max_order) + 1u) * n_vars);

    // Check the values of the derivatives
    // for the initial state.

    // Copy the current state to the order zero
    // of the jet of derivatives.
    std::copy(m_state.begin(), m_state.end(), m_jet.begin());

    // Compute the jet of derivatives at max order.
    auto jet_ptr = m_jet.data();
    m_jet_f(jet_ptr, m_max_order);

    // Check the computed derivatives, starting from order 1.
    if (std::any_of(jet_ptr + n_vars, jet_ptr + m_jet.size(), [](const T &x) { return !std::isfinite(x); })) {
        throw std::invalid_argument(
            "Non-finite value(s) detected in the jet of derivatives corresponding to the initial "
            "state of an adaptive Taylor integrator");
    }
}

template <typename T>
taylor_adaptive_impl<T>::taylor_adaptive_impl(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T> &taylor_adaptive_impl<T>::operator=(taylor_adaptive_impl &&) noexcept = default;

template <typename T>
taylor_adaptive_impl<T>::~taylor_adaptive_impl() = default;

// Implementation detail to make a single integration timestep.
// The size of the timestep is automatically deduced. If
// LimitTimestep is true, then the integration timestep will be
// limited not to be greater than max_delta_t in absolute value.
// If Direction is true then the propagation is done forward
// in time, otherwise backwards. In any case max_delta_t can never
// be negative.
// The function will return a triple, containing
// a flag describing the outcome of the integration,
// the integration timestep that was used and the
// Taylor order that was used.
// NOTE: perhaps there's performance to be gained
// by moving the timestep deduction logic and the
// actual propagation in LLVM (e.g., unrolling
// over the number of variables).
// NOTE: the safer adaptive timestep from
// Jorba still needs to be implemented.
template <typename T>
template <bool LimitTimestep, bool Direction>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, std::uint32_t>
taylor_adaptive_impl<T>::step_impl([[maybe_unused]] T max_delta_t)
{
    assert(std::isfinite(max_delta_t));
    if constexpr (LimitTimestep) {
        assert(max_delta_t >= 0);
    } else {
        assert(max_delta_t == 0);
    }

    // Store the number of variables in the system.
    const auto nvars = static_cast<std::uint32_t>(m_state.size());

    // The tolerance for the timestep is either m_rtol or m_atol,
    // depending on the current values in the state vector.

    // Compute the norm infinity in the state vector.
    T max_abs_state = 0;
    for (const auto &x : m_state) {
        // NOTE: can we do this check just once at the end of the loop?
        // Need to reason about NaNs.
        if (!std::isfinite(x)) {
            return std::tuple{outcome::nf_state, T(0), std::uint32_t(0)};
        }

        max_abs_state = std::max(max_abs_state, std::abs(x));
    }

    // Compute the tolerance for this timestep.
    const auto use_abs = m_rtol * max_abs_state <= m_atol;
    const auto tol = use_abs ? m_atol : m_rtol;

    // Compute the current Taylor order, ensuring it is at least 2.
    // NOTE: the static cast here is ok, as we checked in the constructor
    // that m_max_order is representable.
    const auto order = std::max(std::uint32_t(2), static_cast<std::uint32_t>(std::ceil(-std::log(tol) / 2 + 1)));
    assert(order <= m_max_order);

    // Copy the current state to the order zero
    // of the jet of derivatives.
    auto jet_ptr = m_jet.data();
    std::copy(m_state.begin(), m_state.end(), jet_ptr);

    // Compute the jet of derivatives at the given order.
    m_jet_f(jet_ptr, order);

    // Check the computed derivatives, starting from order 1.
    if (std::any_of(jet_ptr + nvars, jet_ptr + (order + 1u) * nvars, [](const T &x) { return !std::isfinite(x); })) {
        return std::tuple{outcome::nf_derivative, T(0), std::uint32_t(0)};
    }

    // Now we compute an estimation of the radius of convergence of the Taylor
    // series at orders order and order - 1.

    // First step is to determine the norm infinity of the derivatives
    // at orders order and order - 1.
    T max_abs_diff_o = 0, max_abs_diff_om1 = 0;
    for (std::uint32_t i = 0; i < nvars; ++i) {
        max_abs_diff_om1 = std::max(max_abs_diff_om1, std::abs(jet_ptr[(order - 1u) * nvars + i]));
        max_abs_diff_o = std::max(max_abs_diff_o, std::abs(jet_ptr[order * nvars + i]));
    }

    // Estimate rho at orders order and order - 1.
    const auto rho_om1 = use_abs ? std::pow(1 / max_abs_diff_om1, 1 / static_cast<T>(order - 1u))
                                 : std::pow(max_abs_state / max_abs_diff_om1, 1 / static_cast<T>(order - 1u));
    const auto rho_o = use_abs ? std::pow(1 / max_abs_diff_o, 1 / static_cast<T>(order))
                               : std::pow(max_abs_state / max_abs_diff_o, 1 / static_cast<T>(order));
    if (std::isnan(rho_om1) || std::isnan(rho_o)) {
        return std::tuple{outcome::nan_rho, T(0), std::uint32_t(0)};
    }

    // Take the minimum.
    const auto rho_m = std::min(rho_o, rho_om1);

    // Now determine the step size using the formula with safety factors.
    auto h = rho_m / (std::exp(T(1)) * std::exp(T(1))) * std::exp((T(-7) / T(10)) / (order - 1u));
    if constexpr (LimitTimestep) {
        // Make sure h does not exceed max_delta_t.
        h = std::min(h, max_delta_t);
    }
    if constexpr (!Direction) {
        // When propagating backwards, invert the sign of the timestep.
        h = -h;
    }

    // Update the state.
    auto cur_h = h;
    for (std::uint32_t o = 1; o < order + 1u; ++o, cur_h *= h) {
        const auto d_ptr = jet_ptr + o * nvars;

        for (std::uint32_t i = 0; i < nvars; ++i) {
            // NOTE: use FMA wrappers here?
            m_state[i] += cur_h * d_ptr[i];
        }
    }

    // Update the time.
    m_time += h;

    return std::tuple{outcome::success, h, order};
}

template <typename T>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, std::uint32_t> taylor_adaptive_impl<T>::step()
{
    return step_impl<false, true>(0);
}

template <typename T>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, std::uint32_t> taylor_adaptive_impl<T>::step_backward()
{
    return step_impl<false, false>(0);
}

template <typename T>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, std::uint32_t> taylor_adaptive_impl<T>::step(T max_delta_t)
{
    if (!std::isfinite(max_delta_t)) {
        throw std::invalid_argument(
            "A non-finite max_delta_t was passed to the step() function of an adaptive Taylor integrator");
    }

    if (max_delta_t >= 0) {
        return step_impl<true, true>(max_delta_t);
    } else {
        return step_impl<true, false>(-max_delta_t);
    }
}

template <typename T>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, T, std::uint32_t, std::uint32_t, std::size_t>
taylor_adaptive_impl<T>::propagate_for(T delta_t, std::size_t max_steps)
{
    return propagate_until(m_time + delta_t, max_steps);
}

template <typename T>
std::tuple<typename taylor_adaptive_impl<T>::outcome, T, T, std::uint32_t, std::uint32_t, std::size_t>
taylor_adaptive_impl<T>::propagate_until(T t, std::size_t max_steps)
{
    if (!std::isfinite(t)) {
        throw std::invalid_argument(
            "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator");
    }

    // Initial values for the counter,
    // the min/max abs of the integration
    // timesteps, and min/max Taylor orders.
    std::size_t step_counter = 0;
    T min_h = std::numeric_limits<T>::infinity(), max_h = 0;
    std::uint32_t min_order = std::numeric_limits<std::uint32_t>::max(), max_order = 0;

    if (t == m_time) {
        return std::tuple{outcome::success, min_h, max_h, min_order, max_order, step_counter};
    }

    if ((t > m_time && !std::isfinite(t - m_time)) || (t < m_time && !std::isfinite(m_time - t))) {
        throw std::overflow_error("The time limit passed to the propagate_until() function is too large and it "
                                  "results in an overflow condition");
    }

    if (t > m_time) {
        while (true) {
            const auto [res, h, t_order] = step_impl<true, true>(t - m_time);

            if (res != outcome::success) {
                return std::tuple{res, min_h, max_h, min_order, max_order, step_counter};
            }

            // Update the number of steps
            // completed successfully.
            ++step_counter;

            // Update min/max Taylor orders.
            min_order = std::min(min_order, t_order);
            max_order = std::max(max_order, t_order);

            // Break out if the time limit is reached,
            // *before* updating the min_h/max_h values.
            if (t <= m_time) {
                break;
            }

            // Update min_h/max_h.
            assert(h >= 0);
            min_h = std::min(min_h, h);
            max_h = std::max(max_h, h);

            // Check the max number of steps stopping criterion.
            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{outcome::step_limit, min_h, max_h, min_order, max_order, step_counter};
            }
        }
    } else {
        while (true) {
            const auto [res, h, t_order] = step_impl<true, false>(m_time - t);

            if (res != outcome::success) {
                return std::tuple{res, min_h, max_h, min_order, max_order, step_counter};
            }

            ++step_counter;

            min_order = std::min(min_order, t_order);
            max_order = std::max(max_order, t_order);

            if (t >= m_time) {
                break;
            }

            min_h = std::min(min_h, std::abs(h));
            max_h = std::max(max_h, std::abs(h));

            if (max_steps != 0u && step_counter == max_steps) {
                return std::tuple{outcome::step_limit, min_h, max_h, min_order, max_order, step_counter};
            }
        }
    }

    return std::tuple{outcome::success, min_h, max_h, min_order, max_order, step_counter};
}

template <typename T>
void taylor_adaptive_impl<T>::set_time(T t)
{
    if (!std::isfinite(t)) {
        throw std::invalid_argument("Non-finite time " + std::to_string(t)
                                    + " passed to the set_time() function of an adaptive Taylor integrator");
    }

    m_time = t;
}

template <typename T>
void taylor_adaptive_impl<T>::set_state(const std::vector<T> &state)
{
    if (&state == &m_state) {
        // Check that state and m_state are not the same object,
        // otherwise std::copy() cannot be used.
        return;
    }

    if (state.size() != m_state.size()) {
        throw std::invalid_argument(
            "The state vector passed to the set_state() function of an adaptive Taylor integrator has a size of "
            + std::to_string(state.size()) + ", which is inconsistent with the size of the current state vector ("
            + std::to_string(m_state.size()) + ")");
    }

    if (std::any_of(state.begin(), state.end(), [](const T &x) { return !std::isfinite(x); })) {
        throw std::invalid_argument(
            "A non-finite state vector was passed to the set_state() function of an adaptive Taylor integrator");
    }

    // Do the copy.
    std::copy(state.begin(), state.end(), m_state.begin());
}

template <typename T>
const std::string &taylor_adaptive_impl<T>::get_ir() const
{
    return m_ir;
}

template <typename T>
const std::vector<expression> &taylor_adaptive_impl<T>::get_decomposition() const
{
    return m_dc;
}

// Explicit instantiation of the implementation classes.
template class taylor_adaptive_impl<double>;
template class taylor_adaptive_impl<long double>;

} // namespace detail

} // namespace heyoka
