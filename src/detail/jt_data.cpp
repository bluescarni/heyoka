// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/i_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

void jt_data::save(boost::archive::binary_oarchive &, unsigned) const {}

void jt_data::load(boost::archive::binary_iarchive &, unsigned) {}

jt_data::jt_data() = default;

namespace
{

auto jt_initial_checks(const std::vector<std::pair<expression, expression>> &sys,
                       const std::variant<diff_args, std::vector<expression>> &jt_args)
{
    // Initial checks on sys:
    // - sys must not be empty,
    // - the lhs must consist of variables with unique names,
    // - all the variables in the rhs must appear in the lhs,
    // - no variable in the lhs or rhs can begin with "__".
    // NOTE: these checks are redundant with the checks performed by taylor_decompose()
    // later. It seems however excessively complicated to abstract these checks away
    // to avoid repetition, so for now let us just eat the cost.
    if (sys.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot integrate a system of zero equations");
    }

    // Store in a separate vector the rhs.
    std::vector<expression> sys_rhs;
    sys_rhs.reserve(sys.size());
    std::ranges::copy(sys | std::views::transform([](const auto &p) { return p.second; }), std::back_inserter(sys_rhs));

    // This will eventually contain the list
    // of all variables in the system.
    std::vector<std::string> lhs_vars;
    // Maintain a set as well to check for duplicates.
    std::unordered_set<std::string> lhs_vars_set;

    for (const auto &[lhs, _] : sys) {
        // Infer the variable from the current lhs.
        std::visit(
            [&lhs, &lhs_vars, &lhs_vars_set](const auto &v) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(v)>, variable>) {
                    // Check if the lhs variable begins with "__".
                    if (v.name().starts_with("__")) [[unlikely]] {
                        throw std::invalid_argument(
                            fmt::format("Invalid variable name '{}' detected in a system of differential equations: "
                                        "names beginning with '__' are reserved for internal use",
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

    // Now we need to validate jt_args. Specifically, if jt_args is a list of expressions,
    // we check that:
    // - all expressions are either vars or pars,
    // - all vars/pars appear in the ODE sys.
}

} // namespace

jt_data::jt_data(const std::vector<std::pair<expression, expression>> &sys,
                 std::variant<diff_args, std::vector<expression>> jt_args)
{
    jt_initial_checks(sys, jt_args);
}

jt_data::jt_data(const jt_data &) = default;

jt_data::jt_data(jt_data &&) noexcept = default;

jt_data &jt_data::operator=(const jt_data &) = default;

jt_data &jt_data::operator=(jt_data &&) noexcept = default;

jt_data::~jt_data() = default;

} // namespace detail

HEYOKA_END_NAMESPACE
