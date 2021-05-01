// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <limits>
#include <stdexcept>

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;
using namespace Catch::literals;

template <class T>
void test_eval()
{
    using std::acos;
    using std::acosh;
    using std::asin;
    using std::asinh;
    using std::atan;
    using std::atanh;
    using std::cos;
    using std::erf;
    using std::exp;
    using std::log;
    using std::pow;
    using std::sin;
    using std::sqrt;
    using std::tan;

    auto [x, y] = make_vars("x", "y");

    // We test on a number
    {
        expression ex(T(0.125));
        std::unordered_map<std::string, T> in;
        REQUIRE(eval(ex, in) == T(0.125));
    }
    // We test on a variable
    {
        std::unordered_map<std::string, T> in{{"x", T(0.125)}};
        REQUIRE(eval(x, in) == T(0.125));
    }
    // We test on a function call

    // We test on a binary operator
    {
        std::unordered_map<std::string, T> in{{"x", T(0.125)}};
        REQUIRE(eval(x / T(0.125), in) == 1.);
    }
    // We test on a deeper tree
    {
        expression ex = (x * y + x * y * y * y) / (x * y + x * y * y * y) - x / y;
        std::unordered_map<std::string, T> in{{"x", T(0.125)}, {"y", T(0.125)}};
        REQUIRE(eval(ex, in) == 0.);
    }
    // We test the corner case of a dictionary not containing the variable.
    {
        expression ex = x * y;
        std::unordered_map<std::string, T> in{{"x", T(0.125)}};
        REQUIRE_THROWS(eval(ex, in));
    }
    // We test the implementation of all function.
    {
        std::unordered_map<std::string, T> in{{"x", T(0.125)}};
        REQUIRE(eval(acos(x), in) == acos(T(0.125)));
        REQUIRE(eval(asin(x), in) == asin(T(0.125)));
        REQUIRE(eval(atan(x), in) == atan(T(0.125)));
        REQUIRE(eval(atanh(x), in) == atanh(T(0.125)));
        REQUIRE(eval(cos(x), in) == cos(T(0.125)));
        REQUIRE(eval(sin(x), in) == sin(T(0.125)));
        REQUIRE(eval(log(x), in) == log(T(0.125)));
        REQUIRE(eval(exp(x), in) == exp(T(0.125)));
        REQUIRE(eval(pow(x, x), in) == pow(T(0.125), T(0.125)));
        REQUIRE(eval(tan(x), in) == tan(T(0.125)));
        REQUIRE(eval(sqrt(x), in) == sqrt(T(0.125)));
        REQUIRE(eval(sigmoid(x), in) == T(1.) / (T(1.) + exp(-T(0.125))));
        REQUIRE(eval(erf(x), in) == erf(T(0.125)));
        REQUIRE(eval(neg(x), in) == -T(0.125));
        REQUIRE(eval(square(x), in) == T(0.125) * T(0.125));

        // For these we require only an approximate match. The test may fail
        // otherwise (not clear why, a possible substitution at compile time might
        // be using an implementation for these functions that is not guaranteed to match the std one)
        REQUIRE(eval(acosh(x + heyoka::expression(T(1.))), in) == approximately(acosh(T(1.125))));
        REQUIRE(eval(asinh(x), in) == approximately(asinh(T(0.125))));
    }
}

TEST_CASE("eval")
{
    test_eval<double>();
    test_eval<long double>();

#if defined(HEYOKA_HAVE_REAL128)
    test_eval<mppp::real128>();
#endif
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

#if 0

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
        expression ex = pow("x"_var, 2.1_dbl) + ("y"_var * "z"_var) * 2_dbl;
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
        expression ex = pow("x"_var, 2.1_dbl) + (("x"_var * "y"_var) * 2_dbl);
        std::unordered_map<std::string, double> in{{"x", 2.345}, {"y", -1.}};
        auto connections = compute_connections(ex);
        auto node_values = compute_node_values_dbl(ex, in, connections);
        REQUIRE(node_values.size() == 9u);
        REQUIRE(node_values[0] == std::pow(2.345, 2.1) - 2 * 2.345);
        REQUIRE(node_values[1] == std::pow(2.345, 2.1));
        REQUIRE(node_values[2] == 2.345);
        REQUIRE(node_values[3] == 2.1);
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

#endif

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
#if 0
    // We test that the gradient of the mathematical identity sin^2(x) + cos^2(x) = 1 is zero
    {
        expression ex = cos("x"_var) * cos("x"_var) + sin("x"_var) * sin("x"_var);
        auto connections = compute_connections(ex);
        std::unordered_map<std::string, double> point;
        point["x"] = 2.3;
        auto grad = compute_grad_dbl(ex, point, connections);
        REQUIRE(grad["x"] == 0_a);
    }
#endif
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

TEST_CASE("is_integral")
{
    REQUIRE(!detail::is_integral("x"_var));
    REQUIRE(detail::is_integral(0_dbl));
    REQUIRE(detail::is_integral(1_dbl));
    REQUIRE(detail::is_integral(-1_dbl));
    REQUIRE(detail::is_integral(42_dbl));
    REQUIRE(detail::is_integral(-42_dbl));
    REQUIRE(!detail::is_integral(expression{number{std::numeric_limits<double>::infinity()}}));
    REQUIRE(!detail::is_integral(expression{number{-std::numeric_limits<double>::infinity()}}));
    REQUIRE(!detail::is_integral(expression{number{std::numeric_limits<double>::quiet_NaN()}}));
    REQUIRE(!detail::is_integral(expression{number{-std::numeric_limits<double>::quiet_NaN()}}));
    REQUIRE(!detail::is_integral(-42.1_dbl));
    REQUIRE(!detail::is_integral(.1e-6_dbl));

    REQUIRE(!detail::is_odd_integral_half("x"_var));
    REQUIRE(!detail::is_odd_integral_half(0_dbl));
    REQUIRE(!detail::is_odd_integral_half(-1_dbl));
    REQUIRE(!detail::is_odd_integral_half(1_dbl));
    REQUIRE(!detail::is_odd_integral_half(-2_dbl));
    REQUIRE(!detail::is_odd_integral_half(2_dbl));
    REQUIRE(!detail::is_odd_integral_half(-42_dbl));
    REQUIRE(!detail::is_odd_integral_half(42_dbl));
    REQUIRE(!detail::is_odd_integral_half(-42.123_dbl));
    REQUIRE(!detail::is_odd_integral_half(.1e-7_dbl));
    REQUIRE(!detail::is_odd_integral_half(expression{number{std::numeric_limits<double>::infinity()}}));
    REQUIRE(!detail::is_odd_integral_half(expression{number{-std::numeric_limits<double>::infinity()}}));
    REQUIRE(!detail::is_odd_integral_half(expression{number{std::numeric_limits<double>::quiet_NaN()}}));
    REQUIRE(!detail::is_odd_integral_half(expression{number{-std::numeric_limits<double>::quiet_NaN()}}));
    REQUIRE(detail::is_odd_integral_half(-1_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(1_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(-3_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(3_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(-5_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(5_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(-53231_dbl / 2_dbl));
    REQUIRE(detail::is_odd_integral_half(449281_dbl / 2_dbl));
    REQUIRE(!detail::is_odd_integral_half(-53222_dbl / 2_dbl));
    REQUIRE(!detail::is_odd_integral_half(449282_dbl / 2_dbl));
}

TEST_CASE("get_param_size")
{
    using Catch::Matchers::Message;

    REQUIRE(get_param_size(0_dbl) == 0u);
    REQUIRE(get_param_size(1_dbl) == 0u);

    REQUIRE(get_param_size("x"_var) == 0u);
    REQUIRE(get_param_size("y"_var) == 0u);

    REQUIRE(get_param_size("x"_var + 1_dbl) == 0u);
    REQUIRE(get_param_size(1_dbl + "y"_var) == 0u);
    REQUIRE(get_param_size(cos(1_dbl + "y"_var)) == 0u);
    REQUIRE(get_param_size(sin(cos(1_dbl + "y"_var))) == 0u);

    REQUIRE(get_param_size(par[0]) == 1u);
    REQUIRE(get_param_size(par[123]) == 124u);
    REQUIRE_THROWS_MATCHES(get_param_size(par[std::numeric_limits<std::uint32_t>::max()]), std::overflow_error,
                           Message("Overflow dected in get_n_param()"));
    REQUIRE(get_param_size(par[123] + "x"_var) == 124u);
    REQUIRE(get_param_size("x"_var + par[123]) == 124u);
    REQUIRE(get_param_size(par[123] + 1_dbl) == 124u);
    REQUIRE(get_param_size(2_dbl + par[123]) == 124u);
    REQUIRE(get_param_size(par[123] + par[122]) == 124u);
    REQUIRE(get_param_size(par[122] + par[123]) == 124u);
    REQUIRE(get_param_size(par[500] - sin(cos(par[1] + "y"_var) + par[4])) == 501u);
}

TEST_CASE("binary simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(x + -1. == x - 1.);
    REQUIRE(y - -1. == y + 1.);
}

TEST_CASE("neg simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(x + (-y) == x - y);
    REQUIRE(x - (-y) == x + y);
    REQUIRE(x + (neg(neg(y))) == x + y);
    REQUIRE(x - (neg(neg(y))) == x - y);
    REQUIRE(x + (neg(neg(par[0]))) == x + par[0]);
    REQUIRE(x - (neg(neg(par[0]))) == x - par[0]);

    REQUIRE((-x) * (-y) == x * y);
    REQUIRE((-x) / (-y) == x / y);

    REQUIRE(neg(neg(x)) * neg(neg(y)) == x * y);
    REQUIRE(neg(neg(x)) / neg(neg(y)) == x / y);
    REQUIRE(neg(neg(x)) * neg(neg(par[0])) == x * par[0]);
    REQUIRE(neg(neg(x)) / neg(neg(par[0])) == x / par[0]);
}

TEST_CASE("has time")
{
    namespace hy = heyoka;

    auto [x, y] = make_vars("x", "y");

    REQUIRE(!has_time(x));
    REQUIRE(!has_time(y));
    REQUIRE(!has_time(x + y));
    REQUIRE(!has_time(1_dbl));
    REQUIRE(!has_time(par[0]));
    REQUIRE(!has_time(2_dbl - par[0]));

    REQUIRE(has_time(hy::time));
    REQUIRE(has_time(hy::time + 1_dbl));
    REQUIRE(has_time(par[42] + hy::time));
    REQUIRE(has_time((x + y) * (hy::time + 1_dbl)));
    REQUIRE(has_time((x + y) * (par[0] * hy::time + 1_dbl)));
}
