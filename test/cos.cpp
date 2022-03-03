// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <sstream>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/neg.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("cos neg simpl")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(cos(-(x + y)) == cos(x + y));
    REQUIRE(cos(neg(x + y)) == cos(x + y));
    REQUIRE(cos(neg(neg(x + y))) == cos(x + y));
    REQUIRE(cos(neg(neg(par[0]))) == cos(par[0]));
}

TEST_CASE("cos diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(cos(x * x - y), x) == -sin(x * x - y) * (2. * x));
    REQUIRE(diff(cos(x * x - y), y) == sin(x * x - y));

    REQUIRE(diff(cos(par[0] * par[0] - y), par[0]) == -sin(par[0] * par[0] - y) * (2. * par[0]));
    REQUIRE(diff(cos(x * x - par[1]), par[1]) == sin(x * x - par[1]));
}

TEST_CASE("cos number simpl")
{
    using std::cos;

    auto [x] = make_vars("x");

    REQUIRE(cos(x * 0.) == 1_dbl);
    REQUIRE(cos(0.123_dbl) == expression{cos(0.123)});
    // REQUIRE(cos(-0.123_ldbl) == expression{cos(-0.123l)});

#if defined(HEYOKA_HAVE_REAL128)
    using namespace mppp::literals;

    REQUIRE(cos(-0.123_f128) == expression{cos(0.123_rq)});
#endif
}

TEST_CASE("cos s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = cos(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == cos(x));
}
