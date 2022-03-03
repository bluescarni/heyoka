// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_BINOMIAL_HPP
#define HEYOKA_DETAIL_BINOMIAL_HPP

#include <cassert>
#include <cmath>
#include <cstddef>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace heyoka::detail
{

template <typename T>
inline auto binomial(std::uint32_t i, std::uint32_t j)
{
    // NOTE: the Boost implementation requires j <= i.
    assert(j <= i); // LCOV_EXCL_LINE
    return boost::math::binomial_coefficient<T>(boost::numeric_cast<unsigned>(i), boost::numeric_cast<unsigned>(j));
}

// NOTE: if Boost does not support long double, we provide
// our own implementation.
#if defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)

// LCOV_EXCL_START

template <>
inline auto binomial<long double>(std::uint32_t i, std::uint32_t j)
{
    assert(j <= i);

    const auto a = std::lgamma(static_cast<long double>(i) + 1);
    const auto b = std::lgamma(static_cast<long double>(j) + 1);
    const auto c = std::lgamma(static_cast<long double>(i) - j + 1);

    return std::exp(a - b - c);
}

// LCOV_EXCL_STOP

#endif

} // namespace heyoka::detail

#endif
