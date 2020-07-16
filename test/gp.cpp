// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <random>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

TEST_CASE("expression_generator")
{
    detail::random_engine_type engine(123456789u);
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
    detail::random_engine_type engine(123456789u);
    expression_generator generator({"x", "y"}, engine);
    // Test default values
    REQUIRE(generator.get_bos()
            == std::vector<binary_operator::type>({binary_operator::type::add, binary_operator::type::sub,
                                                   binary_operator::type::mul, binary_operator::type::div}));
    REQUIRE(generator.get_u_funcs() == std::vector<expression (*)(expression)>({heyoka::sin, heyoka::cos}));
    REQUIRE(generator.get_b_funcs() == std::vector<expression (*)(expression, expression)>({}));
    REQUIRE(generator.get_range_dbl() == 10.);
    REQUIRE(generator.get_weights() == std::vector<double>({8., 2., 1., 4., 1.}));
    // Test that setters set the requested values.
    auto bos = std::vector<binary_operator::type>({binary_operator::type::add, binary_operator::type::add,
                                                   binary_operator::type::mul, binary_operator::type::mul});
    generator.set_bos(bos);
    REQUIRE(generator.get_bos()
            == std::vector<binary_operator::type>({binary_operator::type::add, binary_operator::type::add,
                                                   binary_operator::type::mul, binary_operator::type::mul}));
    auto bfun = std::vector<expression (*)(expression, expression)>({heyoka::pow});
    generator.set_b_funcs(bfun);
    REQUIRE(generator.get_b_funcs() == std::vector<expression (*)(expression, expression)>({heyoka::pow}));
    auto ufun = std::vector<expression (*)(expression)>({heyoka::cos, heyoka::cos, heyoka::cos});
    generator.set_u_funcs(ufun);
    REQUIRE(generator.get_u_funcs()
            == std::vector<expression (*)(expression)>({heyoka::cos, heyoka::cos, heyoka::cos}));
    generator.set_range_dbl(2);
    REQUIRE(generator.get_range_dbl() == 2.);
    generator.set_weights({1.,1.,1.,1.,1.});
    REQUIRE(generator.get_weights() == std::vector<double>({1.,1.,1.,1.,1.}));
    // We test the throws
    REQUIRE_THROWS(generator.set_weights({2.,3.}));
}

TEST_CASE("streaming operator")
{
    detail::random_engine_type engine(123456789u);
    expression_generator generator({"x", "y"}, engine);
    std::cout << generator << "\n";
}