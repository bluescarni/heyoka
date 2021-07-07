// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <sstream>

#include <boost/math/constants/constants.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

TEST_CASE("erf")
{
    using std::erf;
    auto [x] = make_vars("x");
    // Test the textual output
    std::ostringstream stream;
    stream << erf(x);
    REQUIRE(stream.str() == "erf(x)");
    // Test the expression evaluation
    REQUIRE(eval<double>(erf(x), {{"x", 0.}}) == erf(0.));
    REQUIRE(eval<double>(erf(x), {{"x", 1.}}) == erf(1.));
}

// Variable template for the constant sqrt(pi) / 2 at different levels of precision.
template <typename T>
const auto sqrt_pi_2 = std::sqrt(boost::math::constants::pi<T>()) / 2;

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 sqrt_pi_2<mppp::real128> = mppp::sqrt(mppp::pi_128) / 2;

#endif

TEST_CASE("erf diff")
{
    auto [x, y] = make_vars("x", "y");
#if defined(HEYOKA_HAVE_REAL128)
    auto coeff = 1. / sqrt_pi_2<mppp::real128>;
#else
    auto coeff = 1. / sqrt_pi_2<long double>;
#endif
    REQUIRE(diff(erf(x * x - y), x) == (coeff * exp((-(x * x - y) * (x * x - y)))) * (2. * x));
    REQUIRE(diff(erf(x * x + y), y) == (coeff * exp((-(x * x + y) * (x * x + y)))));
}

TEST_CASE("erf s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = erf(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == erf(x));
}
