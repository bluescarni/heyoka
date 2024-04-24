// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("add sub")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(x + 1_dbl == 1_dbl + x);
    REQUIRE(par[0] + 1_dbl == 1_dbl + par[0]);
    REQUIRE((x + y) + 1_dbl == 1_dbl + (x + y));

    REQUIRE((par[0] + x) + 1_dbl == 1_dbl + (par[0] + x));
}

TEST_CASE("mul div")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(x * 2_dbl == 2_dbl * x);
    REQUIRE(par[0] * 2_dbl == 2_dbl * par[0]);
    REQUIRE((x * y) * 2_dbl == 2_dbl * (x * y));

    REQUIRE((par[0] * x) * 2_dbl == 2_dbl * (par[0] * x));
}
