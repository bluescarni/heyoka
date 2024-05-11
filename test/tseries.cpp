// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <fmt/core.h>

#include <heyoka/expression.hpp>
#include <heyoka/tseries.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("tseries") {}

TEST_CASE("to_tseries")
{
    auto ts = to_tseries({42_dbl}, 12)[0];

    REQUIRE(ts.get_order() == 12u);
    REQUIRE(ts.get_cfs()[0] == 42_dbl);

    for (auto i = 1u; i <= 12u; ++i) {
        REQUIRE(ts.get_cfs()[i] == 0_dbl);
    }

    ts = to_tseries({par[42]}, 12)[0];

    REQUIRE(ts.get_order() == 12u);
    REQUIRE(ts.get_cfs()[0] == par[42]);

    for (auto i = 1u; i <= 12u; ++i) {
        REQUIRE(ts.get_cfs()[i] == 0_dbl);
    }

    ts = to_tseries({"x"_var}, 12)[0];

    REQUIRE(ts.get_order() == 12u);

    for (auto i = 0u; i <= 12u; ++i) {
        REQUIRE(ts.get_cfs()[i] == expression(fmt::format("cf_{}_{}", i, "x")));
    }

    fmt::println("{}", to_tseries({"x"_var + "y"_var}, 12)[0]);
}
