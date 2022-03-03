// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/neg.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("neg ostream")
{
    auto [x, y] = make_vars("x", "y");

    std::ostringstream oss;
    oss << neg(x + y);

    REQUIRE(oss.str() == "-(x + y)");

    oss.str("");
    oss << -x;

    REQUIRE(oss.str() == "-x");
}

TEST_CASE("neg diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(neg(x + y), "x") == -1_dbl);
    REQUIRE(diff(-(x + y), "x") == -1_dbl);
    REQUIRE(diff(-(x * x + y * x), "x") == -(2. * x + y));

    REQUIRE(diff(neg(par[0] + y), par[0]) == -1_dbl);
    REQUIRE(diff(-(par[0] + y), par[0]) == -1_dbl);
    REQUIRE(diff(-(x * x + par[1] * x), par[1]) == -x);
}

TEST_CASE("unary minus simpl")
{
    REQUIRE(-1_dbl == expression{-1.});
    REQUIRE(-1.1_ldbl == expression{-1.1l});

    auto [x] = make_vars("x");

    REQUIRE(-x == neg(x));
}

TEST_CASE("unary minus minus simpl")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(-(-(x + y)) == x + y);
    REQUIRE(-(-sin(x + y)) == sin(x + y));
    REQUIRE(-sin(x + y) == neg(sin(x + y)));
}

TEST_CASE("neg s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = -x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == -x);
}
