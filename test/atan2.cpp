// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/math/atan2.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("atan2 diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(atan2(y, x), "x") == (-y) / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "y") == x / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "z") == 0_dbl);
    REQUIRE(diff(atan2(x * y, y / x), "x")
            == (y / x * y - (x * y) * (-y / (x * x))) / ((y / x) * (y / x) + (x * y) * (x * y)));
}
