// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/dfun.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

dfun_impl::dfun_impl() : dfun_impl("x"_var, {}, {}) {}

namespace
{

// Helper to validate the construction arguments of a dfun and assemble
// the function name.
auto make_dfun_name(const expression &v, std::vector<expression> args,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &didx)
{
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;

    // v must be a variable.
    if (!std::holds_alternative<variable>(v.value())) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "dfuns can be created only from a variable, but the input expression '{}' is not a variable", v));
    }

    // All expressions in args must be either variables or parameters.
    if (std::ranges::any_of(args, [](const auto &e) {
            return !std::holds_alternative<variable>(e.value()) && !std::holds_alternative<param>(e.value());
        })) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("All arguments to a dfun must be either variables or parameters, but one or more non-variable "
                        "and non-parameter were detected in the arguments list '{}'",
                        args));
    }

    // Check for duplicates in args.
    if (std::unordered_set(args.begin(), args.end()).size() != args.size()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Duplicate expressions where detected when constructing a dfun from the arguments '{}'", args));
    }

    // args must not contain v.
    if (std::ranges::any_of(args, [&v](const auto &e) { return e == v; })) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "A dfun constructed from the variable '{}' cannot contain the same variable within the arguments", v));
    }

    // Validate didx.
    if (!didx.empty()) {
        // Make sure we can compute the total diff order via std::uint32_t.
        su32_t tot_diff_order = 0;

        // Helper to validate an element in didx.
        auto validate_p = [&args, &tot_diff_order](const auto &p) {
            const auto [idx, order] = p;

            if (idx >= args.size()) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Invalid index {} detected in the indices vector passed to the constructor of a dfun: "
                                "the index must be less than the number of arguments ({})",
                                idx, args.size()));
            }

            if (order == 0u) [[unlikely]] {
                throw std::invalid_argument("Invalid zero derivative order detected in the indices vector passed to "
                                            "the constructor of a dfun: all derivative orders must be positive");
            }

            // Update tot_diff_order.
            tot_diff_order += order;
        };

        // Validate the first element.
        validate_p(didx[0]);

        // Validate the remaining elements
        for (auto it = didx.begin() + 1; it != didx.end(); ++it) {
            validate_p(*it);

            // Check that didx is sorted according to the index.
            if (!(it->first > (it - 1)->first)) [[unlikely]] {
                throw std::invalid_argument("The indices in the indices vector passed to "
                                            "the constructor of a dfun must be sorted in strictly ascending order");
            }
        }
    }

    // Create the name,
    std::string retval = "dfun_";

    for (auto it = didx.begin(); it != didx.end(); ++it) {
        const auto [idx, order] = *it;
        retval += fmt::format("{},{}", idx, order);

        if (it + 1 != didx.end()) {
            retval += " ";
        }
    }

    retval += "_";
    retval += std::get<variable>(v.value()).name();

    return std::make_tuple(std::move(retval), std::move(args));
}

} // namespace

dfun_impl::dfun_impl(const expression &v, std::vector<expression> args,
                     std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
    : func_base(make_dfun_name(v, std::move(args), didx)), m_v_name(std::get<variable>(v.value()).name()),
      m_didx(std::move(didx))
{
}

void dfun_impl::to_stream(std::ostringstream &oss) const
{
    // Compute the total diff order.
    const auto diff_order = std::accumulate(m_didx.begin(), m_didx.end(), static_cast<std::uint32_t>(0),
                                            [](auto acc, const auto &p) { return acc + p.second; });

    if (diff_order == 1u) {
        oss << fmt::format("(d{})", m_v_name);
    } else {
        oss << fmt::format("(d^{} {})", diff_order, m_v_name);
    }

    assert(diff_order > 0u || m_didx.empty());

    if (!m_didx.empty()) {
        oss << "/(";

        for (auto it = m_didx.begin(); it != m_didx.end(); ++it) {
            const auto [idx, order] = *it;

            oss << "d";
            stream_expression(oss, args()[idx]);
            if (order > 1u) {
                oss << fmt::format("^{}", order);
            }

            if (it + 1 != m_didx.end()) {
                oss << " ";
            }
        }

        oss << ")";
    }
}

} // namespace detail

expression dfun(const expression &v, std::vector<expression> args,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
{
    return expression{func{detail::dfun_impl{v, std::move(args), std::move(didx)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::dfun_impl)
