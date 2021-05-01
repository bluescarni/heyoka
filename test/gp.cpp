// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <random>

#include <heyoka/expression.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math.hpp>
#include <heyoka/splitmix64.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

#if 0

TEST_CASE("expression_generator")
{
    splitmix64 engine(123456789ul);
    // Construction
    {
        CHECK_NOTHROW(expression_generator({"x", "y"}, engine));
        // Checks the possibility to generate expressions without variables.
        CHECK_NOTHROW(expression_generator({}, engine));
    }
    // Call operator
    {
        // We check that an expression without variables does not contain variables
        expression_generator generator0({}, engine);
        for (auto i = 0u; i < 5; ++i) {
            for (auto j = i; j < i + 5; ++j) {
                auto ex = generator0(i, j);
                REQUIRE(get_variables(ex).size() == 0u);
            }
        }
        // We check that an expression with one variable does contain one or less variables
        expression_generator generator1({"x"}, engine);
        for (auto i = 0u; i < 5; ++i) {
            for (auto j = i; j < i + 5; ++j) {
                auto ex = generator1(i, j);
                REQUIRE(get_variables(ex).size() <= 1u);
            }
        }
        // We check that an expression with two variables does contain two or less variables
        expression_generator generator2({"x", "y"}, engine);
        for (auto i = 0u; i < 5; ++i) {
            for (auto j = i; j < i + 5; ++j) {
                auto ex = generator2(i, j);
                REQUIRE(get_variables(ex).size() <= 2u);
            }
        }
        // We check that an expression with 1-2 variables and a big depth, contains exactly 1-2 variables (this test may
        // fail if seed is changed or implementation details are)
        for (auto i = 0u; i < 5; ++i) {
            auto ex = generator1(10u, 10u + i);
            REQUIRE(get_variables(ex).size() == 1u);
        }
        for (auto i = 0u; i < 5; ++i) {
            auto ex = generator2(10u, 10u + i);
            REQUIRE(get_variables(ex).size() == 2u);
        }
    }
}

TEST_CASE("setters and getters")
{
    splitmix64 engine(123456789ul);
    expression_generator generator({"x", "y"}, engine);
    REQUIRE(generator.get_u_funcs() == std::vector<expression (*)(expression)>({heyoka::sin, heyoka::cos}));
    REQUIRE(generator.get_b_funcs() == std::vector<expression (*)(expression, expression)>({}));
    REQUIRE(generator.get_range_dbl() == 10.);
    REQUIRE(generator.get_weights() == std::vector<double>({8., 2., 1., 4., 1.}));
    // Test that setters set the requested values.
    auto bfun = std::vector<expression (*)(expression, expression)>({heyoka::pow});
    generator.set_b_funcs(bfun);
    REQUIRE(generator.get_b_funcs() == std::vector<expression (*)(expression, expression)>({heyoka::pow}));
    auto ufun = std::vector<expression (*)(expression)>({heyoka::cos, heyoka::cos, heyoka::cos});
    generator.set_u_funcs(ufun);
    REQUIRE(generator.get_u_funcs()
            == std::vector<expression (*)(expression)>({heyoka::cos, heyoka::cos, heyoka::cos}));
    generator.set_range_dbl(2);
    REQUIRE(generator.get_range_dbl() == 2.);
    generator.set_weights({1., 1., 1., 1., 1.});
    REQUIRE(generator.get_weights() == std::vector<double>({1., 1., 1., 1., 1.}));
    // We test the throws
    REQUIRE_THROWS(generator.set_weights({2., 3.}));
}

TEST_CASE("streaming operator")
{
    splitmix64 engine(123456789ul);
    expression_generator generator({"x", "y"}, engine);
    CHECK_NOTHROW(std::cout << generator);
}

TEST_CASE("count_nodes")
{
    auto [x, y] = make_vars("x", "y");
    {
        expression ex = x;
        REQUIRE(count_nodes(ex) == 1);
    }
    {
        expression ex = x * x;
        REQUIRE(count_nodes(ex) == 2);
    }
    {
        expression ex = cos(x);
        REQUIRE(count_nodes(ex) == 2);
    }
    {
        expression ex = 2_dbl * x * x + y;
        REQUIRE(count_nodes(ex) == 7);
    }
    {
        expression ex = sin(x) * 2_dbl - pow(y, 6_dbl) * x;
        REQUIRE(count_nodes(ex) == 10);
    }
    {
        expression ex = pow(sin(x), cos(x * y)) - x * y * (x - y);
        REQUIRE(count_nodes(ex) == 15);
    }
}

TEST_CASE("fetch from node id")
{
    auto [x, y] = make_vars("x", "y");
    {
        expression ex = x * y;
        REQUIRE(*fetch_from_node_id(ex, 0) == ex);
    }
    {
        expression ex0 = x;
        expression ex1 = y;
        expression ex2 = ex0 * ex1;
        expression ex3 = cos(ex1);
        expression ex4 = ex2 - ex3;
        REQUIRE(*fetch_from_node_id(ex4, 0) == ex4);
        REQUIRE(*fetch_from_node_id(ex4, 1) == ex2);
        REQUIRE(*fetch_from_node_id(ex4, 2) == ex0);
        REQUIRE(*fetch_from_node_id(ex4, 3) == ex1);
        REQUIRE(*fetch_from_node_id(ex4, 4) == ex3);
        REQUIRE(*fetch_from_node_id(ex4, 5) == ex1);
        REQUIRE(fetch_from_node_id(ex4, 6) == nullptr);
    }
}

TEST_CASE("mutations")
{
    splitmix64 engine(123456789ul);
    expression_generator generator({"x", "y"}, engine);
    // ex = (((y * (y * x)) * ((y + 0.688871) + (x * y))) - (y + (x + (y + y))))
    auto ex = generator(2, 4);
    mutate(ex, 0u, generator, 0u, 0u);
    REQUIRE(count_nodes(ex) == 1u);
    REQUIRE_THROWS(mutate(ex, 3u, generator, 0u, 0u));
}

#endif
