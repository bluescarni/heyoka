// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

void lagrangian_impl_sanity_checks(const expression &L, const std::vector<expression> &qs,
                                   const std::vector<expression> &qdots, const expression &D)
{
    // Sanity checks on qs and qdots.
    if (qs.size() != qdots.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of generalised coordinates ({}) must be equal to the number of generalised velocities ({})",
            qs.size(), qdots.size()));
    }

    if (qs.empty()) {
        throw std::invalid_argument("Cannot define a Lagrangian without state variables");
    }

    for (const auto &q : qs) {
        if (!std::holds_alternative<variable>(q.value())) {
            throw std::invalid_argument(fmt::format(
                "The list of generalised coordinates contains the expression '{}' which is not a variable", q));
        }

        if (std::get<variable>(q.value()).name().starts_with("__")) {
            throw std::invalid_argument(
                fmt::format("The list of generalised coordinates contains a variable with the invalid name '{}': names "
                            "starting with '__' are reserved for internal use",
                            std::get<variable>(q.value()).name()));
        }
    }

    for (const auto &qdot : qdots) {
        if (!std::holds_alternative<variable>(qdot.value())) {
            throw std::invalid_argument(fmt::format(
                "The list of generalised velocities contains the expression '{}' which is not a variable", qdot));
        }

        if (std::get<variable>(qdot.value()).name().starts_with("__")) {
            throw std::invalid_argument(
                fmt::format("The list of generalised velocities contains a variable with the invalid name '{}': names "
                            "starting with '__' are reserved for internal use",
                            std::get<variable>(qdot.value()).name()));
        }
    }

    // Check for duplicates.
    const std::unordered_set<expression> qs_set{qs.begin(), qs.end()};
    const std::unordered_set<expression> qdots_set{qdots.begin(), qdots.end()};

    if (qs_set.size() != qs.size()) {
        throw std::invalid_argument("The list of generalised coordinates contains duplicates");
    }

    if (qdots_set.size() != qdots.size()) {
        throw std::invalid_argument("The list of generalised velocities contains duplicates");
    }

    for (const auto &q : qs) {
        if (qdots_set.contains(q)) {
            throw std::invalid_argument(fmt::format("The list of generalised coordinates contains the expression '{}' "
                                                    "which also appears as a generalised velocity",
                                                    q));
        }
    }

    // Sanity checks on L.
    for (const auto &v : get_variables(L)) {
        if (!qs_set.contains(expression{v}) && !qdots_set.contains(expression{v})) {
            throw std::invalid_argument(fmt::format(
                "The Lagrangian contains the variable '{}' which is not a generalised position or velocity", v));
        }
    }

    // Sanity checks on D.
    for (const auto &v : get_variables(D)) {
        if (!qdots_set.contains(expression{v})) {
            throw std::invalid_argument(fmt::format(
                "The dissipation function contains the variable '{}' which is not a generalised velocity", v));
        }
    }
}

} // namespace

} // namespace detail

std::vector<std::pair<expression, expression>> lagrangian(const expression &L_, const std::vector<expression> &qs,
                                                          const std::vector<expression> &qdots, const expression &D)
{
    using size_type = boost::safe_numerics::safe<decltype(qs.size())>;

    // Sanity checks.
    detail::lagrangian_impl_sanity_checks(L_, qs, qdots, D);

    // Cache the number of generalised coordinates/velocities.
    const auto n_qs = size_type(qs.size());

    // Replace the time expression with a time variable.
    const auto tm_var = "__tm"_var;
    const auto L = subs(L_, {{heyoka::time, tm_var}});

    // Assemble the diff arguments.
    auto diff_args = qs;
    diff_args.insert(diff_args.end(), qdots.begin(), qdots.end());
    diff_args.push_back(tm_var);

    // NOTE: these next two bits can be run in parallel if needed.
    // Compute the tensor of derivatives of L up to order 2 wrt
    // qs, qdots and time.
    const auto L_dt = diff_tensors({L}, kw::diff_args = diff_args, kw::diff_order = 2);

    // Compute the tensor of derivatives of D up to order 1 wrt qdots.
    const auto D_dt = diff_tensors({D}, kw::diff_args = qdots, kw::diff_order = 1);

    // Start assembling the RHS of the return value.
    std::vector<expression> rhs_ret;
    rhs_ret.reserve(n_qs * 2);

    // dq/dt = qdot.
    for (size_type i = 0; i < n_qs; ++i) {
        rhs_ret.push_back(qdots[i]);
    }

    // Prepare vectors of indices for indexing into L_dt and D_dt.
    std::vector<std::uint32_t> vidx_L;
    vidx_L.resize(1 + n_qs * 2 + 1);

    std::vector<std::uint32_t> vidx_D;
    vidx_D.resize(1 + n_qs);

    // dqdot/dt.
    for (size_type i = 0; i < n_qs; ++i) {
        // Reset the vectors of indices.
        std::ranges::fill(vidx_L, 0);
        std::ranges::fill(vidx_D, 0);

        // dL/dq.
        vidx_L[1 + i] = 1;
        const auto dL_dq = fix_nn(L_dt[vidx_L]);

        // d2L/dqdqdot.
        vidx_L[1 + n_qs + i] = 1;
        const auto d2L_dqdqdot = fix_nn(L_dt[vidx_L]);

        // d2L/dqdot2.
        vidx_L[1 + i] = 0;
        vidx_L[1 + n_qs + i] = 2;
        const auto d2L_dqdot2 = fix_nn(L_dt[vidx_L]);

        // d2L/dtdqdot.
        vidx_L[1 + n_qs + i] = 1;
        vidx_L.back() = 1;
        const auto d2L_dtdqdot = fix_nn(L_dt[vidx_L]);

        // dD/dqdot.
        vidx_D[1 + i] = 1;
        const auto dD_dqdot = fix_nn(D_dt[vidx_D]);

        // Assemble.
        rhs_ret.push_back(sum({dL_dq, -d2L_dqdqdot * qdots[i], -d2L_dtdqdot, -dD_dqdot}) / d2L_dqdot2);
    }

    // Restore the time expression.
    rhs_ret = subs(rhs_ret, {{tm_var, heyoka::time}});

    // Assemble the result.
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(n_qs * 2);

    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(qs[i], std::move(rhs_ret[i]));
    }

    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(qdots[i], std::move(rhs_ret[n_qs + i]));
    }

    return ret;
}

HEYOKA_END_NAMESPACE
