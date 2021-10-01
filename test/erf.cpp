// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/erf.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/s11n.hpp>

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

TEST_CASE("erf diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(erf(x * x - y), x) == (2_dbl / sqrt(pi) * exp((-(x * x - y) * (x * x - y)))) * (2. * x));
    REQUIRE(diff(erf(x * x + y), y) == (2_dbl / sqrt(pi) * exp((-(x * x + y) * (x * x + y)))));

    REQUIRE(diff(erf(par[0] * par[0] - y), par[0])
            == (2_dbl / sqrt(pi) * exp((-(par[0] * par[0] - y) * (par[0] * par[0] - y)))) * (2. * par[0]));
    REQUIRE(diff(erf(x * x + par[1]), par[1]) == (2_dbl / sqrt(pi) * exp((-(x * x + par[1]) * (x * x + par[1])))));
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
