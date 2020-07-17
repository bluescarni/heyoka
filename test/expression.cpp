// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

TEST_CASE("eval_dbl")
{
    // We test on a number
    {
        expression ex = 2.345_dbl;
        std::unordered_map<std::string, double> in;
        REQUIRE(eval_dbl(ex, in) == 2.345);
    }
    // We test on a variable
    {
        expression ex = "x"_var;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        REQUIRE(eval_dbl(ex, in) == 2.345);
    }
    // We test on a function call
    {
        expression ex = cos("x"_var);
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        REQUIRE(eval_dbl(ex, in) == std::cos(2.345));
    }
    // We test on a binary operator
    {
        expression ex = "x"_var / 2.345_dbl;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        REQUIRE(eval_dbl(ex, in) == 1.);
    }
    // We test on a deeper tree
    {
        expression ex = "x"_var * "y"_var + cos("x"_var * "y"_var);
        std::unordered_map<std::string, double> in{{"x", 2.345}, {"y", -1.}};
        REQUIRE(eval_dbl(ex, in) == -2.345 + std::cos(-2.345));
    }
    // We test the corner case of a dictionary not containing the variable.
    {
        expression ex = "x"_var * "y"_var;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        REQUIRE_THROWS(eval_dbl(ex, in));
    }
}

TEST_CASE("eval_batch_dbl")
{
    std::vector<double> out(2);
    // We test on a number
    {
        expression ex = 2.345_dbl;
        std::unordered_map<std::string, std::vector<double>> in{{"x", {-2.345, 20.234}}};
        out = std::vector<double>(2);
        eval_batch_dbl(out, ex, in);
        REQUIRE(out == std::vector<double>{2.345, 2.345});
    }
    // We test on a variable
    {
        expression ex = "x"_var;
        std::unordered_map<std::string, std::vector<double>> in{{"x", {-2.345, 20.234}}};
        out = std::vector<double>(2);
        eval_batch_dbl(out, ex, in);
        REQUIRE(out == std::vector<double>{-2.345, 20.234});
    }
    // We test on a function call
    {
        expression ex = cos("x"_var);
        std::unordered_map<std::string, std::vector<double>> in{{"x", {-2.345, 20.234}}};
        out = std::vector<double>(2);
        eval_batch_dbl(out, ex, in);
        REQUIRE(out == std::vector<double>{std::cos(-2.345), std::cos(20.234)});
    }
    // We test on a deeper tree
    {
        expression ex = "x"_var * "y"_var + cos("x"_var * "y"_var);
        std::unordered_map<std::string, std::vector<double>> in;
        in["x"] = std::vector<double>{3., 4.};
        in["y"] = std::vector<double>{-1., -2.};
        out = std::vector<double>(2);
        eval_batch_dbl(out, ex, in);
        REQUIRE(out == std::vector<double>{-3 + std::cos(-3), -8 + std::cos(-8)});
    }
    // We test the corner case of a dictionary not containing the variable.
    {
        expression ex = "x"_var * "y"_var;
        std::unordered_map<std::string, std::vector<double>> in{{"x", {-2.345, 20.234}}};
        out = std::vector<double>(2);
        REQUIRE_THROWS(eval_batch_dbl(out, ex, in));
    }
}

TEST_CASE("operator == and !=")
{
    // Expression 1
    {
        expression ex1 = "x"_var + 3_dbl + "y"_var * (cos("x"_var + 3_dbl)) / pow("x"_var + 3_dbl, "z"_var + 3_dbl);
        expression ex2 = "x"_var + 3_dbl + "y"_var * (cos("x"_var + 3_dbl)) / pow("x"_var + 3_dbl, "z"_var + 3_dbl);
        expression ex3 = "z"_var + 3_dbl + "y"_var * (cos("x"_var + 3_dbl)) / pow("x"_var + 3_dbl, "z"_var + 3_dbl);
        expression ex4 = "x"_var + 3_dbl + "y"_var * (cos("x"_var - 3_dbl)) / pow("x"_var + 3_dbl, "z"_var + 3_dbl);
        REQUIRE(ex1 == ex1);
        REQUIRE(ex1 == ex2);
        REQUIRE(ex1 != ex3);
        REQUIRE(ex1 != ex4);
    }
    // Expression 2
    {
        expression ex1
            = pow("x"_var + sin(-1_dbl), "z"_var + -2_dbl) / ("x"_var / "y"_var + (sin("x"_var + 3.322_dbl)));
        expression ex2
            = pow("x"_var + sin(-1_dbl), "z"_var + -2_dbl) / ("x"_var / "y"_var + (sin("x"_var + 3.322_dbl)));
        expression ex3
            = pow("y"_var + sin(-1_dbl), "z"_var + -2_dbl) / ("x"_var / "y"_var + (sin("x"_var + 3.322_dbl)));
        expression ex4 = pow("x"_var + sin(-1_dbl), "z"_var + 2_dbl) / ("x"_var / "y"_var + (sin("x"_var + 3.322_dbl)));
        expression ex5
            = pow("x"_var + sin(-1_dbl), "z"_var + -2_dbl) / ("x"_var / "y"_var + (cos("x"_var + 3.322_dbl)));
        REQUIRE(ex1 == ex2);
        REQUIRE(ex1 != ex3);
        REQUIRE(ex1 != ex4);
        REQUIRE(ex1 != ex5);
    }
    // Identities that will not hold
    {
        expression ex1 = 1_dbl + cos("x"_var);
        expression ex2 = cos("x"_var) + 1_dbl;
        expression ex3 = cos("x"_var) + 1_dbl + ex1 - ex1;

        REQUIRE(ex1 != ex2);
        REQUIRE(ex3 != ex2);
    }
}

TEST_CASE("compute connections")
{
    // We test the result on a simple polynomial x^2*y + 2
    {
        expression ex = ("x"_var * ("x"_var * "y"_var)) + 2_dbl;
        auto connections = compute_connections(ex);
        REQUIRE(connections.size() == 7u);
        REQUIRE(connections[0] == std::vector<std::size_t>{1, 6});
        REQUIRE(connections[1] == std::vector<std::size_t>{2, 3});
        REQUIRE(connections[2] == std::vector<std::size_t>{});
        REQUIRE(connections[3] == std::vector<std::size_t>{4, 5});
        REQUIRE(connections[4] == std::vector<std::size_t>{});
        REQUIRE(connections[5] == std::vector<std::size_t>{});
        REQUIRE(connections[6] == std::vector<std::size_t>{});
    }
    // We test the result on a known expression with a simple function 2cos(x) + 2yz
    {
        expression ex = cos("x"_var) * 2_dbl + ("y"_var * "z"_var) * 2_dbl;
        auto connections = compute_connections(ex);
        REQUIRE(connections.size() == 10u);

        REQUIRE(connections[0] == std::vector<std::size_t>{1, 5});
        REQUIRE(connections[1] == std::vector<std::size_t>{2, 4});
        REQUIRE(connections[2] == std::vector<std::size_t>{3});
        REQUIRE(connections[3] == std::vector<std::size_t>{});
        REQUIRE(connections[4] == std::vector<std::size_t>{});
        REQUIRE(connections[5] == std::vector<std::size_t>{6, 9});
        REQUIRE(connections[6] == std::vector<std::size_t>{7, 8});
        REQUIRE(connections[7] == std::vector<std::size_t>{});
        REQUIRE(connections[8] == std::vector<std::size_t>{});
        REQUIRE(connections[9] == std::vector<std::size_t>{});
    }
    // We test the result on a known expression including a multiargument function
    {
        expression ex = pow("x"_var, 2_dbl) + ("y"_var * "z"_var) * 2_dbl;
        auto connections = compute_connections(ex);
        REQUIRE(connections.size() == 9u);
        REQUIRE(connections[0] == std::vector<std::size_t>{1, 4});
        REQUIRE(connections[1] == std::vector<std::size_t>{2, 3});
        REQUIRE(connections[2] == std::vector<std::size_t>{});
        REQUIRE(connections[3] == std::vector<std::size_t>{});
        REQUIRE(connections[4] == std::vector<std::size_t>{5, 8});
        REQUIRE(connections[5] == std::vector<std::size_t>{6, 7});
        REQUIRE(connections[6] == std::vector<std::size_t>{});
        REQUIRE(connections[7] == std::vector<std::size_t>{});
        REQUIRE(connections[8] == std::vector<std::size_t>{});
    }
}

TEST_CASE("update_node_values_dbl")
{
    // We test on a number
    {
        expression ex = 2.345_dbl;
        std::unordered_map<std::string, double> in;
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 1u);
        REQUIRE(node_values[0] == 2.345);
    }
    // We test on a variable
    {
        expression ex = "x"_var;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 1u);
        REQUIRE(node_values[0] == 2.345);
    }
    // We test on a function call
    {
        expression ex = cos("x"_var);
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 2u);
        REQUIRE(node_values[0] == std::cos(2.345));
        REQUIRE(node_values[1] == 2.345);
    }
    // We test on a binary operator
    {
        expression ex = "x"_var / 2.345_dbl;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 3u);
        REQUIRE(node_values[0] == 1);
        REQUIRE(node_values[1] == 2.345);
        // Note here that the tree built (after the simplifictions) is *, "x", 1/2.345
        REQUIRE(node_values[2] == 1. / 2.345);
    }
    // We test on a deeper tree
    {
        expression ex = "x"_var * "y"_var + cos("x"_var * "y"_var);
        std::unordered_map<std::string, double> in{{"x", 2.345}, {"y", -1.}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 8u);
        REQUIRE(node_values[0] == -2.345 + std::cos(-2.345));
        REQUIRE(node_values[1] == -2.345);
        REQUIRE(node_values[2] == 2.345);
        REQUIRE(node_values[3] == -1.);
        REQUIRE(node_values[4] == std::cos(-2.345));
        REQUIRE(node_values[5] == -2.345);
        REQUIRE(node_values[6] == 2.345);
        REQUIRE(node_values[7] == -1.);
    }
    // We test the result on a known expression including a multiargument function
    {
        expression ex = pow("x"_var, 2_dbl) + (("x"_var * "y"_var) * 2_dbl);
        std::unordered_map<std::string, double> in{{"x", 2.345}, {"y", -1.}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 9u);
        REQUIRE(node_values[0] == std::pow(2.345, 2.) - 2 * 2.345);
        REQUIRE(node_values[1] == std::pow(2.345, 2.));
        REQUIRE(node_values[2] == 2.345);
        REQUIRE(node_values[3] == 2.);
        REQUIRE(node_values[4] == -2 * 2.345);
        REQUIRE(node_values[5] == -2.345);
        REQUIRE(node_values[6] == 2.345);
        REQUIRE(node_values[7] == -1.);
    }
    // We test the corner case of a dictionary not containing the variable.
    {
        expression ex = "x"_var * "y"_var;
        std::unordered_map<std::string, double> in{{"x", 2.345}};
        auto connections = compute_connections(ex);
        REQUIRE_THROWS(compute_node_values_dbl(ex, in, connections));
    }
}

TEST_CASE("compute_grad_dbl")
{
    // We test that the gradient of x is one
    {
        expression ex = "x"_var;
        auto connections = compute_connections(ex);
        std::unordered_map<std::string, double> point;
        point["x"] = 2.3;
        auto grad = compute_grad_dbl(ex, point, connections);
        REQUIRE(grad["x"] == 1);
    }
    // We test that the gradient of x*y is {x, y}
    {
        expression ex = "x"_var * "y"_var;
        auto connections = compute_connections(ex);
        std::unordered_map<std::string, double> point;
        point["x"] = 2.3;
        point["y"] = 12.43;
        auto grad = compute_grad_dbl(ex, point, connections);
        REQUIRE(grad["x"] == 12.43);
        REQUIRE(grad["y"] == 2.3);
    }
    // We test that the gradient of the mathematical identity sin^2(x) + cos^2(x) = 1 is zero
    {
        expression ex = cos("x"_var) * cos("x"_var) + sin("x"_var) * sin("x"_var);
        auto connections = compute_connections(ex);
        std::unordered_map<std::string, double> point;
        point["x"] = 2.3;
        auto grad = compute_grad_dbl(ex, point, connections);
        REQUIRE(grad["x"] == 0_a);
    }
}

TEST_CASE("diff")
{
    // We test that the derivative of x is one
    {
        expression ex = "x"_var;
        REQUIRE(diff(ex, "x") == 1_dbl);
    }
    // We test that the derivative of sin is cos
    {
        expression ex = sin("x"_var);
        REQUIRE(diff(ex, "x") == cos("x"_var));
    }
}

TEST_CASE("basic")
{
    auto x = "x"_var, y = "y"_var;

    // auto d = taylor_decompose({(1_dbl + x) + (1_dbl + x)});
    // auto d = taylor_decompose({expression{binary_operator{binary_operator::type::add, 1_dbl, 1_dbl}}, x + y *
    // sin(x)});
    // auto d = taylor_decompose({x * (1_dbl + x), x * (1_dbl + cos(y))});

    // for (const auto &ex : d) {
    //     std::cout << ex << '\n';
    // }

    auto ex = sin("x"_var) + 1.1_ldbl;

    llvm_state s{"pippo"};

    s.add_taylor_jet_dbl("fj", {cos(y), sin(x)}, 20);

    std::cout << s.dump() << '\n';

    s.compile();

    std::cout.precision(14);

    double state[(20 + 1) * 2] = {};
    state[0] = 1;
    state[1] = 2;

    s.fetch_taylor_jet_dbl("fj")(state, 3);
    for (auto v : state) {
        std::cout << v << '\n';
    }

    taylor_adaptive_dbl tad({y, -x}, {0, -1}, 0, 1E-19, 1E-19);

    const auto &st = tad.get_state();

    std::cout << "Initial time: " << tad.get_time() << '\n';
    std::cout << "Initial state: " << st[0] << ", " << st[1] << '\n';
    std::cout << "Exact initial state: " << std::cos(tad.get_time() + 3.14159265358979323846 / 2) << ", "
              << -std::sin(tad.get_time() + 3.14159265358979323846 / 2) << "\n\n";

#if 0
    for (auto i = 0; i < 10; ++i) {
        tad.step();
        std::cout << "Cur time: " << tad.get_time() << '\n';
        std::cout << "Cur state: " << st[0] << ", " << st[1] << '\n';
        std::cout << "Exact state: " << std::cos(tad.get_time() + 3.14159265358979323846 / 2) << ", "
                  << -std::sin(tad.get_time() + 3.14159265358979323846 / 2) << "\n\n";
    }
#endif

    tad.propagate_until(100);
    std::cout << "Final state: " << st[0] << ", " << st[1] << '\n';
    std::cout << "Final time: " << tad.get_time() << '\n';
    tad.propagate_until(0);
    std::cout << "Final state: " << st[0] << ", " << st[1] << '\n';
    std::cout << "Final time: " << tad.get_time() << '\n';
#if 0
    s.compile();

    auto f = s.fetch_ldbl<1>("f");

    std::cout << f(5) << '\n';
#endif
}
