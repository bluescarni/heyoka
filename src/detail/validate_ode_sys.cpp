// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fast_unordered.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

void validate_ode_sys_impl(const std::vector<std::pair<expression, expression>> &sys,
                           const std::vector<expression> &t_funcs, const std::vector<expression> &nt_funcs)
{
    if (sys.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot integrate a system of zero equations");
    }

    // Store in a separate vector the rhs.
    std::vector<expression> sys_rhs;
    sys_rhs.reserve(sys.size());
    std::ranges::transform(sys, std::back_inserter(sys_rhs), &std::pair<expression, expression>::second);

    // Determine the variables in the system of equations
    // from the lhs of the equations. We need to ensure that:
    // - all the lhs expressions are variables
    //   and there are no duplicates,
    // - no lhs variable begins with "__",
    // - all the variables in the rhs expressions
    //   appear in the lhs.
    // Note that not all variables in the lhs
    // need to appear in the rhs: that is, not all variables
    // need to appear in the ODEs.

    // This will eventually contain the list
    // of all variables in the system.
    std::vector<std::string> lhs_vars;
    // Maintain a set as well to check for duplicates.
    fast_uset<std::string> lhs_vars_set;

    for (const auto &[lhs, rhs] : sys) {
        // Infer the variable from the current lhs.
        std::visit(
            [&lhs, &lhs_vars, &lhs_vars_set]<typename T>(const T &v) {
                if constexpr (std::same_as<T, variable>) {
                    // Check if it begins with "__".
                    if (v.name().starts_with("__")) [[unlikely]] {
                        throw std::invalid_argument(
                            fmt::format("Invalid system of differential equations detected: the variable '{}' "
                                        "appears in the left-hand side, but variables beginning with '__' are reserved "
                                        "for internal use",
                                        v.name()));
                    }

                    // Check if this is a duplicate variable.
                    if (const auto res = lhs_vars_set.emplace(v.name()); res.second) [[likely]] {
                        // Not a duplicate, add it to lhs_vars.
                        lhs_vars.push_back(v.name());
                    } else {
                        // Duplicate, error out.
                        throw std::invalid_argument(
                            fmt::format("Invalid system of differential equations detected: the variable '{}' "
                                        "appears in the left-hand side twice",
                                        v.name()));
                    }
                } else {
                    throw std::invalid_argument(
                        fmt::format("Invalid system of differential equations detected: the "
                                    "left-hand side contains the expression '{}', which is not a variable",
                                    lhs));
                }
            },
            lhs.value());
    }

    // Fetch the set of variables in the rhs.
    const auto rhs_vars_set = get_variables(sys_rhs);

    // Check that all variables in the rhs appear in the lhs.
    for (const auto &var : rhs_vars_set) {
        if (!lhs_vars_set.contains(var)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid system of differential equations detected: the variable '{}' "
                            "appears in the right-hand side but not in the left-hand side",
                            var));
        }
    }

    // Store all event functions in a single vector.
    std::vector<expression> ev_funcs(t_funcs);
    ev_funcs.insert(ev_funcs.end(), nt_funcs.begin(), nt_funcs.end());

    // Check that the expressions in ev_funcs contain only
    // state variables.
    const auto ev_funcs_vars = get_variables(ev_funcs);
    for (const auto &ev_func_var : ev_funcs_vars) {
        if (!lhs_vars_set.contains(ev_func_var)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid system of differential equations detected: an event function contains the variable '{}', "
                "which is not a state variable",
                ev_func_var));
        }
    }
}

} // namespace

// Helper to validate a system of ODEs, including the event functions (if any).
template <typename TEvent, typename NTEvent>
void validate_ode_sys(const std::vector<std::pair<expression, expression>> &sys, const std::vector<TEvent> &t_events,
                      const std::vector<NTEvent> &nt_events)
{
    std::vector<expression> t_funcs;
    t_funcs.reserve(t_events.size());
    std::ranges::transform(t_events, std::back_inserter(t_funcs), &TEvent::get_expression);

    std::vector<expression> nt_funcs;
    nt_funcs.reserve(nt_events.size());
    std::ranges::transform(nt_events, std::back_inserter(nt_funcs), &NTEvent::get_expression);

    validate_ode_sys_impl(sys, t_funcs, nt_funcs);
}

// Explicit instantiations.
#define HEYOKA_VALIDATE_ODE_SYS_INST(T)                                                                                \
    template void validate_ode_sys<t_event_impl<T, false>, nt_event_impl<T, false>>(                                   \
        const std::vector<std::pair<expression, expression>> &, const std::vector<t_event_impl<T, false>> &,           \
        const std::vector<nt_event_impl<T, false>> &);

#define HEYOKA_VALIDATE_ODE_SYS_INST_BATCH(T)                                                                          \
    template void validate_ode_sys<t_event_impl<T, true>, nt_event_impl<T, true>>(                                     \
        const std::vector<std::pair<expression, expression>> &, const std::vector<t_event_impl<T, true>> &,            \
        const std::vector<nt_event_impl<T, true>> &);

HEYOKA_VALIDATE_ODE_SYS_INST(float)
HEYOKA_VALIDATE_ODE_SYS_INST_BATCH(float)

HEYOKA_VALIDATE_ODE_SYS_INST(double)
HEYOKA_VALIDATE_ODE_SYS_INST_BATCH(double)

HEYOKA_VALIDATE_ODE_SYS_INST(long double)
HEYOKA_VALIDATE_ODE_SYS_INST_BATCH(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_VALIDATE_ODE_SYS_INST(mppp::real128)
HEYOKA_VALIDATE_ODE_SYS_INST_BATCH(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_VALIDATE_ODE_SYS_INST(mppp::real)

#endif

#undef HEYOKA_VALIDATE_ODE_SYS_INST
#undef HEYOKA_VALIDATE_ODE_SYS_INST_BATCH

} // namespace detail

HEYOKA_END_NAMESPACE
