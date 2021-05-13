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
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("kepE diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(kepE(x, y), x) == sin(kepE(x, y)) / (1_dbl - x * cos(kepE(x, y))));
    REQUIRE(diff(kepE(x, y), y) == 1_dbl / (1_dbl - x * cos(kepE(x, y))));
    auto E = kepE(x * x, x * y);
    REQUIRE(diff(E, x) == (2_dbl * x * sin(E) + y) / (1_dbl - x * x * cos(E)));
    REQUIRE(diff(E, y) == x / (1_dbl - x * x * cos(E)));
}
