// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("square diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(square(x * x - y), "x") == 2_dbl * (x * x - y) * (2_dbl * x));
    REQUIRE(diff(square(x * x - y), "y") == -(2_dbl * (x * x - y)));

    REQUIRE(diff(square(par[0] * par[0] - y), par[0]) == 2_dbl * (par[0] * par[0] - y) * (2_dbl * par[0]));
    REQUIRE(diff(square(x * x - par[1]), par[1]) == -(2_dbl * (x * x - par[1])));
}

TEST_CASE("square s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = square(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == square(x));
}

TEST_CASE("square stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;
        oss << square(x);

        REQUIRE(oss.str() == "x**2");
    }

    {
        std::ostringstream oss;
        oss << square(x + y);

        REQUIRE(oss.str() == "(x + y)**2");
    }

    {
        std::ostringstream oss;
        oss << square(2_dbl);

        REQUIRE(oss.str() == "2.0000000000000000**2");
    }

    {
        std::ostringstream oss;
        oss << square(par[0]);

        REQUIRE(oss.str() == "p0**2");
    }

    {
        std::ostringstream oss;
        oss << square(x + par[0]);

        REQUIRE(oss.str() == "(x + p0)**2");
    }

    {
        std::ostringstream oss;
        oss << square(cos(x + par[0]));

        REQUIRE(oss.str() == "cos((x + p0))**2");
    }
}
