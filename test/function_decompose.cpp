// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic explicit")
{
    using Catch::Matchers::Message;

    // Error handling.
    REQUIRE_THROWS_MATCHES(function_decompose({}, {}), std::invalid_argument,
                           Message("Cannot decompose a function with no outputs"));

    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE_THROWS_MATCHES(function_decompose({x}, {x, x}), std::invalid_argument,
                           Message("Error in the decomposition of a function: the variable 'x' "
                                   "appears in the user-provided list of variables twice"));

    REQUIRE_THROWS_MATCHES(function_decompose({x}, {x, par[0]}), std::invalid_argument,
                           Message("Error in the decomposition of a function: the "
                                   "user-provided list of variables contains the expression 'p0', "
                                   "which is not a variable"));

    REQUIRE_THROWS_MATCHES(function_decompose({x + y}, {x}), std::invalid_argument,
                           Message("Error in the decomposition of a function: the variable 'y' "
                                   "appears in the function but not in the user-provided list of variables"));

    // A couple of trivial cases.
    auto dc = function_decompose({0_dbl}, {});
    REQUIRE(dc.size() == 1u);
    REQUIRE(dc[0] == 0_dbl);

    dc = function_decompose({par[1]}, {});
    REQUIRE(dc.size() == 1u);
    REQUIRE(dc[0] == par[1]);

    // Simple tests.
    dc = function_decompose({x}, {x});
    REQUIRE(dc.size() == 2u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == "u_0"_var);

    dc = function_decompose({x}, {y, x});
    REQUIRE(dc.size() == 3u);
    REQUIRE(dc[0] == y);
    REQUIRE(dc[1] == x);
    REQUIRE(dc[2] == "u_1"_var);

    dc = function_decompose({x}, {x, y});
    REQUIRE(dc.size() == 3u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == "u_0"_var);

    dc = function_decompose({x, y}, {x, y});
    REQUIRE(dc.size() == 4u);
    REQUIRE(dc[0] == x);
    REQUIRE(dc[1] == y);
    REQUIRE(dc[2] == "u_0"_var);
    REQUIRE(dc[3] == "u_1"_var);

    dc = function_decompose({x, y}, {y, x});
    REQUIRE(dc.size() == 4u);
    REQUIRE(dc[0] == y);
    REQUIRE(dc[1] == x);
    REQUIRE(dc[2] == "u_1"_var);
    REQUIRE(dc[3] == "u_0"_var);
}
