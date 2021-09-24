// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/asinh.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("asinh diff var")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(asinh(x * x - y), x) == pow(square(square(x) - y) + 1., -.5) * (2. * x));
    REQUIRE(diff(asinh(x * x + y), y) == pow(square(square(x) + y) + 1., -.5));
}

TEST_CASE("asinh diff par")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(asinh(par[0] * par[0] - y), par[0]) == pow(square(square(par[0]) - y) + 1., -.5) * (2. * par[0]));
    REQUIRE(diff(asinh(x * x + par[1]), par[1]) == pow(square(square(x) + par[1]) + 1., -.5));
}

TEST_CASE("asinh s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = asinh(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == asinh(x));
}
