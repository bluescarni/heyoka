// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/square.hpp>

#include "catch.hpp"

using namespace heyoka;

#if defined(HEYOKA_HAVE_REAL128)

using namespace mppp::literals;

#endif

TEST_CASE("pow expo 0")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 0.) == 1_dbl);
    REQUIRE(heyoka::pow(x, 0.l) == 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 0._rq) == 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, 1.) != 1_dbl);
    REQUIRE(heyoka::pow(x, 1.l) != 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1._rq) != 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, expression{0.}) == 1_dbl);
    REQUIRE(heyoka::pow(x, expression{0.l}) == 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, expression{0._rq}) == 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, expression{1.}) != 1_dbl);
    REQUIRE(heyoka::pow(x, expression{1.l}) != 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, expression{1._rq}) != 1_dbl);

#endif
}

TEST_CASE("pow expo 1")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 1.) == x);
    REQUIRE(heyoka::pow(x, 1.l) == x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1._rq) == x);

#endif

    REQUIRE(heyoka::pow(x, 1.1) != x);
    REQUIRE(heyoka::pow(x, 1.1l) != x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1.1_rq) != x);

#endif
}

TEST_CASE("pow expo 2")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 2.) == square(x));
    REQUIRE(heyoka::pow(x, 2.l) == square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 2._rq) == square(x));

#endif

    REQUIRE(heyoka::pow(x, 2.1) != square(x));
    REQUIRE(heyoka::pow(x, 2.1l) != square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 21._rq) != square(x));

#endif
}
