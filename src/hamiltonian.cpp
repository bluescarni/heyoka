// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/hamiltonian.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

void hamiltonian_impl_sanity_checks(const expression &H, const std::vector<expression> &qs,
                                    const std::vector<expression> &ps)
{
    // Sanity checks on qs and ps.
    if (qs.size() != ps.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of generalised coordinates ({}) must be equal to the number of generalised momenta ({})",
            qs.size(), ps.size()));
    }

    if (qs.empty()) {
        throw std::invalid_argument("Cannot define a Hamiltonian without state variables");
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

    for (const auto &p : ps) {
        if (!std::holds_alternative<variable>(p.value())) {
            throw std::invalid_argument(
                fmt::format("The list of generalised momenta contains the expression '{}' which is not a variable", p));
        }

        if (std::get<variable>(p.value()).name().starts_with("__")) {
            throw std::invalid_argument(
                fmt::format("The list of generalised momenta contains a variable with the invalid name '{}': names "
                            "starting with '__' are reserved for internal use",
                            std::get<variable>(p.value()).name()));
        }
    }

    // Check for duplicates.
    const std::unordered_set<expression> qs_set{qs.begin(), qs.end()};
    const std::unordered_set<expression> ps_set{ps.begin(), ps.end()};

    if (qs_set.size() != qs.size()) {
        throw std::invalid_argument("The list of generalised coordinates contains duplicates");
    }

    if (ps_set.size() != ps.size()) {
        throw std::invalid_argument("The list of generalised momenta contains duplicates");
    }

    for (const auto &q : qs) {
        if (ps_set.contains(q)) {
            throw std::invalid_argument(fmt::format("The list of generalised coordinates contains the expression '{}' "
                                                    "which also appears as a generalised momentum",
                                                    q));
        }
    }

    // Sanity checks on H.
    for (const auto &v : get_variables(H)) {
        if (!qs_set.contains(expression{v}) && !ps_set.contains(expression{v})) {
            throw std::invalid_argument(fmt::format(
                "The Hamiltonian contains the variable '{}' which is not a generalised position or momentum", v));
        }
    }
}

} // namespace

} // namespace detail

std::vector<std::pair<expression, expression>> hamiltonian(const expression &H, const std::vector<expression> &qs,
                                                           const std::vector<expression> &ps)
{
    using size_type = boost::safe_numerics::safe<decltype(qs.size())>;

    // Sanity checks.
    detail::hamiltonian_impl_sanity_checks(H, qs, ps);

    // Cache the number of generalised coordinates/momenta.
    const auto n_qs = size_type(qs.size());

    // Assemble the diff arguments.
    auto diff_args = qs;
    diff_args.insert(diff_args.end(), ps.begin(), ps.end());

    // Compute the tensor of derivatives of H up to order 1 wrt
    // qs and ps.
    const auto H_dt = diff_tensors({H}, diff_args, kw::diff_order = 1);

    // Fetch the gradient.
    auto grad = H_dt.get_gradient();

    // Assemble the return value.
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(n_qs * 2);

    // dq/dt.
    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(qs[i], std::move(grad[n_qs + i]));
    }

    // dp/dt.
    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(ps[i], -grad[i]);
    }

    return ret;
}

HEYOKA_END_NAMESPACE
