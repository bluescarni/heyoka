// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/tan.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("tan diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(tan(x * x - y), x) == (1. + square(tan(square(x) - y))) * (2. * x));
    REQUIRE(diff(tan(x * x + y), y) == (1. + square(tan(square(x) + y))));
}

TEST_CASE("tan s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = tan(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == tan(x));
}
