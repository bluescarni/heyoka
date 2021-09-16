// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("pi stream")
{
    std::ostringstream oss;

    oss << heyoka::pi;

    REQUIRE(oss.str() == "pi");
}

TEST_CASE("pi diff")
{
    REQUIRE(diff(heyoka::pi, "x") == 0_dbl);

    auto x = "x"_var;

    REQUIRE(diff(heyoka::pi * cos(2. * x + 2. * heyoka::pi), "x")
            == heyoka::pi * (-2. * sin(2. * x + 2. * heyoka::pi)));
}

TEST_CASE("pi s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = heyoka::pi + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == heyoka::pi + x);
}
