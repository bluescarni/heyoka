// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/math/square.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("mul square simpl")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(x * x == square(x));
    REQUIRE(x * y != square(x));
    REQUIRE((x + y) * (x + y) == square(x + y));
    REQUIRE((y + x) * (x + y) != square(x + y));
}
