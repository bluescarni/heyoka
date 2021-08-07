// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/square.hpp>

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

} // namespace detail

expression atan2(expression y, expression x)
{
    return expression{func{detail::atan2_impl(std::move(y), std::move(x))}};
}

} // namespace heyoka
