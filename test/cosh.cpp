// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/cosh.hpp>
#include <heyoka/math/sinh.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("cosh diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(cosh(x * x - y), x) == sinh(square(x) - y) * (2. * x));
    REQUIRE(diff(cosh(x * x + y), y) == sinh(square(x) + y));
}

TEST_CASE("cosh s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = cosh(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == cosh(x));
}
