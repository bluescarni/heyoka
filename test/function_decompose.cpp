// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <stdexcept>
#include <tuple>

#include <heyoka/expression.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic auto")
{
    using Catch::Matchers::Message;

    // Error handling.
    REQUIRE_THROWS_MATCHES(function_decompose({}), std::invalid_argument,
                           Message("Cannot decompose a function with no outputs"));

    // A couple of trivial cases.
    auto [dc, nvars] = function_decompose({0_dbl});
    REQUIRE(dc.size() == 1u);
    REQUIRE(dc[0] == 0_dbl);
    REQUIRE(nvars == 0u);

    std::tie(dc, nvars) = function_decompose({par[1]});
    REQUIRE(dc.size() == 1u);
    REQUIRE(dc[0] == par[1]);
    REQUIRE(nvars == 0u);

    auto [x, y] = make_vars("x", "y");

    std::tie(dc, nvars) = function_decompose({x});
    REQUIRE(dc.size() == 2u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == "u_0"_var);
    REQUIRE(nvars == 1u);

    std::tie(dc, nvars) = function_decompose({x, y});
    REQUIRE(dc.size() == 4u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == "u_0"_var);
    REQUIRE(dc[3] == "u_1"_var);
    REQUIRE(nvars == 2u);

    std::tie(dc, nvars) = function_decompose({y, x});
    REQUIRE(dc.size() == 4u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == "u_1"_var);
    REQUIRE(dc[3] == "u_0"_var);
    REQUIRE(nvars == 2u);

    auto tmp = x + y;

    std::tie(dc, nvars) = function_decompose({tmp + tmp});
    REQUIRE(dc.size() == 5u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == "u_0"_var + "u_1"_var);
    REQUIRE(dc[3] == "u_2"_var + "u_2"_var);
    REQUIRE(dc[4] == "u_3"_var);
    REQUIRE(nvars == 2u);

    // Try with nullary function too.
    std::tie(dc, nvars) = function_decompose({tmp + tmp, tmp * heyoka::time});
    REQUIRE(dc.size() == 8u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == heyoka::time);
    REQUIRE(dc[3] == "u_0"_var + "u_1"_var);
    REQUIRE(dc[4] == "u_3"_var + "u_3"_var);
    REQUIRE(dc[5] == "u_3"_var * "u_2"_var);
    REQUIRE(dc[6] == "u_4"_var);
    REQUIRE(dc[7] == "u_5"_var);
    REQUIRE(nvars == 2u);
}
