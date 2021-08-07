// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/taylor.hpp>

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

atan2_impl::atan2_impl(expression y, expression x) : func_base("atan2", std::vector{std::move(y), std::move(x)}) {}

atan2_impl::atan2_impl() : atan2_impl(0_dbl, 1_dbl) {}

expression atan2_impl::diff(const std::string &s) const
{
    assert(args().size() == 2u);

    const auto &y = args()[0];
    const auto &x = args()[1];

    auto den = square(x) + square(y);

    return (x * heyoka::diff(y, s) - y * heyoka::diff(x, s)) / std::move(den);
}

taylor_dc_t::size_type atan2_impl::taylor_decompose(taylor_dc_t &u_vars_defs) &&
{
    assert(args().size() == 2u);

    // Decompose the arguments.
    auto &y = *get_mutable_args_it().first;
    if (const auto dres = taylor_decompose_in_place(std::move(y), u_vars_defs)) {
        y = expression{"u_{}"_format(dres)};
    }
    auto &x = *(get_mutable_args_it().first + 1);
    if (const auto dres = taylor_decompose_in_place(std::move(x), u_vars_defs)) {
        x = expression{"u_{}"_format(dres)};
    }

    // Append x * x and y * y.
    u_vars_defs.emplace_back(square(x), std::vector<std::uint32_t>{});
    u_vars_defs.emplace_back(square(y), std::vector<std::uint32_t>{});

    // Append x*x + y*y.
    u_vars_defs.emplace_back(expression{"u_{}"_format(u_vars_defs.size() - 2u)}
                                 + expression{"u_{}"_format(u_vars_defs.size() - 1u)},
                             std::vector<std::uint32_t>{});

    // Append the atan2 decomposition.
    u_vars_defs.emplace_back(func{std::move(*this)}, std::vector<std::uint32_t>{});

    // Add the hidden dep.
    (u_vars_defs.end() - 1)->second.push_back(boost::numeric_cast<std::uint32_t>(u_vars_defs.size() - 2u));

    // Compute the return value (pointing to the
    // decomposed atan2).
    return u_vars_defs.size() - 1u;
}

} // namespace detail

expression atan2(expression y, expression x)
{
    return expression{func{detail::atan2_impl(std::move(y), std::move(x))}};
}

} // namespace heyoka
