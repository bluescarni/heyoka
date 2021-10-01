// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math/sin.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("sin diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(sin(x * x - y), x) == cos(x * x - y) * (2. * x));
    REQUIRE(diff(sin(x * x - y), y) == -cos(x * x - y));

    REQUIRE(diff(sin(par[0] * par[0] - y), par[0]) == cos(par[0] * par[0] - y) * (2. * par[0]));
    REQUIRE(diff(sin(x * x - par[1]), par[1]) == -cos(x * x - par[1]));
}

TEST_CASE("sin s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = sin(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sin(x));
}

TEST_CASE("sin number simpl")
{
    using std::sin;

    auto [x] = make_vars("x");

    REQUIRE(sin(x * 0.) == 0_dbl);
    REQUIRE(sin(0.123_dbl) == expression{sin(0.123)});
    REQUIRE(sin(-0.123_ldbl) == expression{sin(-0.123l)});

#if defined(HEYOKA_HAVE_REAL128)
    using namespace mppp::literals;

    REQUIRE(sin(-0.123_f128) == expression{sin(-0.123_rq)});
#endif
}
