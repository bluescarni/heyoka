// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/asin.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/square.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("asin diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(asin(x * x - y), x) == pow(1. - square(square(x) - y), -.5) * (2. * x));
    REQUIRE(diff(asin(x * x + y), y) == pow(1. - square(square(x) + y), -.5));
}
