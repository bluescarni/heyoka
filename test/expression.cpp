// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/config.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

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
        REQUIRE(eval(acos(x), in) == approximately(acos(T(0.125))));
        REQUIRE(eval(asin(x), in) == approximately(asin(T(0.125))));
        REQUIRE(eval(atan(x), in) == approximately(atan(T(0.125))));
        REQUIRE(eval(atanh(x), in) == approximately(atanh(T(0.125))));
        REQUIRE(eval(cos(x), in) == approximately(cos(T(0.125))));
        REQUIRE(eval(sin(x), in) == approximately(sin(T(0.125))));
        REQUIRE(eval(log(x), in) == approximately(log(T(0.125))));
        REQUIRE(eval(exp(x), in) == approximately(exp(T(0.125))));
        REQUIRE(eval(pow(x, x), in) == approximately(pow(T(0.125), T(0.125))));
        REQUIRE(eval(tan(x), in) == approximately(tan(T(0.125))));
        REQUIRE(eval(sqrt(x), in) == approximately(sqrt(T(0.125))));
        REQUIRE(eval(sigmoid(x), in) == approximately(T(1.) / (T(1.) + exp(-T(0.125)))));
        REQUIRE(eval(erf(x), in) == approximately(erf(T(0.125))));
        REQUIRE(eval(neg(x), in) == approximately(-T(0.125)));
        REQUIRE(eval(square(x), in) == approximately(T(0.125) * T(0.125)));
        REQUIRE(eval(acosh(x + heyoka::expression(T(1.))), in) == approximately(acosh(T(1.125))));
        REQUIRE(eval(asinh(x), in) == approximately(asinh(T(0.125))));
    }
}

TEST_CASE("eval")
{
    test_eval<double>();

#if !defined(HEYOKA_ARCH_PPC)
    test_eval<long double>();
#endif

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

        REQUIRE(ex1 == ex2);
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

#endif

TEST_CASE("diff var")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(diff(x, "x") == 1_dbl);
    REQUIRE(diff(y, "x") == 0_dbl);

    REQUIRE(diff(1_dbl, "x") == 0_dbl);
    REQUIRE(std::holds_alternative<double>(std::get<number>(diff(1_dbl, "x").value()).value()));
    REQUIRE(std::holds_alternative<long double>(std::get<number>(diff(1_ldbl, "x").value()).value()));

    REQUIRE(diff(par[42], "x") == 0_dbl);
    REQUIRE(std::holds_alternative<double>(std::get<number>(diff(par[42], "x").value()).value()));

    // Test the caching of derivatives.
    auto foo = x * (x + y), bar = (foo - x) + (2. * foo);
    auto bar_diff = diff(bar, "x");

    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_diff.value()).args()[0].value()).args()[1].value()).get_ptr()
        == std::get<func>(std::get<func>(std::get<func>(bar_diff.value()).args()[1].value()).args()[1].value())
               .get_ptr());
}

TEST_CASE("diff par")
{
    using Catch::Matchers::Message;

    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE_THROWS_MATCHES(
        diff(x, 1_dbl), std::invalid_argument,
        Message("Derivatives are currently supported only with respect to variables and parameters"));

    REQUIRE(diff(x + par[0], par[0]) == 1_dbl);
    REQUIRE(diff(y * par[1], par[1]) == y);
    REQUIRE(diff(y * par[1], par[0]) == 0_dbl);

    REQUIRE(diff(1_dbl, par[0]) == 0_dbl);
    REQUIRE(std::holds_alternative<double>(std::get<number>(diff(1_dbl, par[0]).value()).value()));
    REQUIRE(std::holds_alternative<long double>(std::get<number>(diff(1_ldbl, par[0]).value()).value()));

    REQUIRE(diff("x"_var, par[42]) == 0_dbl);
    REQUIRE(std::holds_alternative<double>(std::get<number>(diff("x"_var, par[42]).value()).value()));

    // Test the caching of derivatives.
    auto foo = par[0] * (par[0] + y), bar = (foo - par[0]) + (2. * foo);
    auto bar_diff = diff(bar, par[0]);

    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_diff.value()).args()[0].value()).args()[1].value()).get_ptr()
        == std::get<func>(std::get<func>(std::get<func>(bar_diff.value()).args()[1].value()).args()[1].value())
               .get_ptr());
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

    // Test with repeated subexpressions.
    auto [x, y, z] = make_vars("x", "y", "z");

    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    REQUIRE(get_param_size(bar) == 0u);

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + par[42];
    bar = par[23] + (foo - x) / (2. * foo) + par[32];

    REQUIRE(get_param_size(bar) == 43u);

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + par[42];
    bar = par[83] + (foo - x) / (2. * foo) + par[32];

    REQUIRE(get_param_size(bar) == 84u);

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + par[42];
    bar = par[23] + (foo - x) / (2. * foo) + par[92];

    REQUIRE(get_param_size(bar) == 93u);
}

TEST_CASE("binary simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(y - -1. == y + 1.);
}

TEST_CASE("add simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(1_dbl + 2_dbl == 3_dbl);
    REQUIRE(1_ldbl + 2_dbl == 3_ldbl);

    REQUIRE(0_dbl + (x + y) == x + y);

    REQUIRE(1_dbl + (1_dbl + x) == 2_dbl + x);
    REQUIRE(1_dbl + (1_dbl + (x + 3_dbl)) == 5_dbl + x);
    REQUIRE(1_dbl + (1_dbl + (4_dbl + (x + 3_dbl))) == 9_dbl + x);

    REQUIRE(x + 0_dbl == x);
    REQUIRE(x + 1_dbl == 1_dbl + x);

    REQUIRE(std::get<func>((1_dbl + y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((1_dbl + y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::add);
    REQUIRE(std::get<func>((1_dbl + y).value()).extract<detail::binary_op>()->args() == std::vector{1_dbl, y});

    REQUIRE(std::get<func>((x + y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((x + y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::add);
    REQUIRE(std::get<func>((x + y).value()).extract<detail::binary_op>()->args() == std::vector{x, y});
}

TEST_CASE("sub simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(1_dbl - 2_dbl == -1_dbl);
    REQUIRE(1_ldbl - 3_dbl == -2_ldbl);

    REQUIRE(0_dbl - x == -x);

    REQUIRE((x + y) - 0_dbl == x + y);
    REQUIRE((x + y) - 1_dbl == x + y + -1_dbl);

    REQUIRE(std::get<func>((1_dbl - y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((1_dbl - y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::sub);
    REQUIRE(std::get<func>((1_dbl - y).value()).extract<detail::binary_op>()->args() == std::vector{1_dbl, y});

    REQUIRE(std::get<func>((x - y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((x - y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::sub);
    REQUIRE(std::get<func>((x - y).value()).extract<detail::binary_op>()->args() == std::vector{x, y});
}

TEST_CASE("mul simpls")
{
    auto [x, y] = make_vars("x", "y");

    // Verify simplification to square(),
    // for non-number arguments.
    REQUIRE(2_dbl * 2_dbl == 4_dbl);
    REQUIRE(x * x == square(x));

    REQUIRE(1_dbl * 2_dbl == 2_dbl);
    REQUIRE(3_ldbl * 2_dbl == 6_ldbl);

    REQUIRE(0_dbl * x == 0_dbl);
    REQUIRE(1_dbl * (x + y) == x + y);

    REQUIRE(1_dbl * (1_dbl * x) == x);
    REQUIRE(1_dbl * (2_dbl * (x * 3_dbl)) == 6_dbl * x);
    REQUIRE(1_dbl * (1_dbl * (-4_dbl * (x * 3_dbl))) == -12_dbl * x);

    REQUIRE(x * 2_dbl == 2_dbl * x);
    REQUIRE(x * -1_dbl == -1_dbl * x);

    REQUIRE(2_dbl * -x == -2_dbl * x);
    REQUIRE(2_dbl * (-3_dbl * -x) == 6_dbl * x);
    REQUIRE(2_dbl * (-x * -3_dbl) == 6_dbl * x);
    REQUIRE(2_dbl * (-3_dbl * (-x * -4_dbl)) == -24_dbl * x);

    REQUIRE(std::get<func>((2_dbl * y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((2_dbl * y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::mul);
    REQUIRE(std::get<func>((2_dbl * y).value()).extract<detail::binary_op>()->args() == std::vector{2_dbl, y});

    REQUIRE(std::get<func>((x * y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((x * y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::mul);
    REQUIRE(std::get<func>((x * y).value()).extract<detail::binary_op>()->args() == std::vector{x, y});

    REQUIRE(-1_dbl * (x + y) == -(x + y));
    REQUIRE((x - y) * -1_dbl == -(x - y));
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

TEST_CASE("div simpls")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    REQUIRE(-x / -y == x / y);

    REQUIRE_THROWS_MATCHES(x / 0_dbl, zero_division_error, Message("Division by zero"));

    REQUIRE(1_dbl / 2_dbl == 0.5_dbl);
    REQUIRE(1_dbl / -2_dbl == -0.5_dbl);

    REQUIRE(x / 1_dbl == x);
    REQUIRE(-x / 1_dbl == -x);
    REQUIRE(x / -1_dbl == -x);
    REQUIRE(-x / -1_dbl == x);
    REQUIRE(-x / 2_dbl == x / -2_dbl);

    REQUIRE(0_dbl / -x == 0_dbl);
    REQUIRE(1_dbl / -x == -1_dbl / x);
    REQUIRE(-2_dbl / -x == 2_dbl / x);

    REQUIRE(x / 2_dbl / 2_dbl == x / 4_dbl);

    REQUIRE(std::get<func>((2_dbl / y).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((2_dbl / y).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::div);
    REQUIRE(std::get<func>((2_dbl / y).value()).extract<detail::binary_op>()->args() == std::vector{2_dbl, y});

    REQUIRE(std::get<func>((y / 2_dbl).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((y / 2_dbl).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::div);
    REQUIRE(std::get<func>((y / 2_dbl).value()).extract<detail::binary_op>()->args() == std::vector{y, 2_dbl});

    REQUIRE(std::get<func>((y / x).value()).extract<detail::binary_op>() != nullptr);
    REQUIRE(std::get<func>((y / x).value()).extract<detail::binary_op>()->op() == detail::binary_op::type::div);
    REQUIRE(std::get<func>((y / x).value()).extract<detail::binary_op>()->args() == std::vector{y, x});
}

TEST_CASE("has time")
{
    namespace hy = heyoka;

    auto [x, y, z] = make_vars("x", "y", "z");

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

    // With common subexpressions.
    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    REQUIRE(!has_time(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = (foo - x) / (2. * foo);

    REQUIRE(has_time(bar));

    foo = hy::time + ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = (foo - x) / (2. * foo);

    REQUIRE(has_time(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = hy::time + (foo - x) / (2. * foo);

    REQUIRE(has_time(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = (foo - x) / (2. * foo) + hy::time;

    REQUIRE(has_time(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = (foo - x) / (2. * foo) + hy::time;

    REQUIRE(has_time(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = hy::time + (foo - x) / (2. * foo) + hy::time;

    REQUIRE(has_time(bar));
}

TEST_CASE("pairwise_prod")
{
    auto [x0, x1, x2, x3, x4, x5] = make_vars("x0", "x1", "x2", "x3", "x4", "x5");

    REQUIRE(pairwise_prod({}) == 1_dbl);
    REQUIRE(pairwise_prod({x0}) == x0);
    REQUIRE(pairwise_prod({x0, x1}) == x0 * x1);
    REQUIRE(pairwise_prod({x0, x1, x2}) == x0 * x1 * x2);
    REQUIRE(pairwise_prod({x0, x1, x2, x3}) == (x0 * x1) * (x2 * x3));
    REQUIRE(pairwise_prod({x0, x1, x2, x3, x4}) == (x0 * x1) * (x2 * x3) * x4);
    REQUIRE(pairwise_prod({x0, x1, x2, x3, x4, x5}) == ((x0 * x1) * (x2 * x3)) * (x4 * x5));
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    {
        boost::archive::binary_oarchive oa(ss);

        oa << x;
    }

    x = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> x;
    }

    REQUIRE(x == "x"_var);

    ss.str("");

    x = par[42];

    {
        boost::archive::binary_oarchive oa(ss);

        oa << x;
    }

    x = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> x;
    }

    REQUIRE(x == par[42]);

    ss.str("");

    x = 0.1_ldbl;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << x;
    }

    x = "x"_var;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> x;
    }

    REQUIRE(x == 0.1_ldbl);

    ss.str("");

    x = "x"_var;

    // Test shallow copies in subexpressions.
    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << bar;
    }

    bar = "x"_var;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> bar;
    }

    // Make sure multiple instances of 'foo' in bar point to the same
    // underlying object.
    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[1].value()).args()[1].value()).get_ptr()
        == std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[0].value()).args()[0].value()).get_ptr());
    REQUIRE(std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[1].value()).args()[1].value()).get_ptr()
            != std::get<func>(foo.value()).get_ptr());
}

TEST_CASE("get_n_nodes")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(get_n_nodes(expression{}) == 1u);
    REQUIRE(get_n_nodes(1_dbl) == 1u);
    REQUIRE(get_n_nodes(par[0]) == 1u);
    REQUIRE(get_n_nodes(x) == 1u);

    REQUIRE(get_n_nodes(x + y) == 3u);
    REQUIRE(get_n_nodes(x - y) == 3u);
    REQUIRE(get_n_nodes(-z) == 2u);
    REQUIRE(get_n_nodes(heyoka::time) == 1u);
    REQUIRE(get_n_nodes(x + (y * z)) == 5u);
    REQUIRE(get_n_nodes((x - y - z) + (y * z)) == 9u);

    // Try with subexpressions repeating in the tree.
    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo),
         bar2 = (copy(foo) - x) / (2. * copy(foo));

    REQUIRE(get_n_nodes(bar) == 35u);
    REQUIRE(get_n_nodes(bar2) == 35u);
}

TEST_CASE("equality")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(x == x);
    REQUIRE(x != y);

    REQUIRE(x + y == x + y);

    auto foo = (x + y) * z, bar = foo;

    REQUIRE(foo == bar);
    REQUIRE(x + foo == x + bar);
}

TEST_CASE("get_variables")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(get_variables(x) == std::vector<std::string>{"x"});
    REQUIRE(get_variables(1_dbl) == std::vector<std::string>{});
    REQUIRE(get_variables(par[0]) == std::vector<std::string>{});
    REQUIRE(get_variables(y + x * z) == std::vector<std::string>{"x", "y", "z"});

    auto tmp = x * z, foo = x - z - 5_dbl;
    REQUIRE(get_variables((y + tmp) / foo * tmp - foo) == std::vector<std::string>{"x", "y", "z"});
}

TEST_CASE("rename_variables")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = 1_dbl;
    rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == 1_dbl);

    ex = par[0];
    rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == par[0]);

    ex = x;
    rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == "a"_var);

    ex = x + y;
    rename_variables(ex, {{"x", "a"}, {"y", "b"}});
    REQUIRE(ex == "a"_var + "b"_var);

    auto tmp = x * z, foo = x - z - 5_dbl;
    ex = (y + tmp) / foo * tmp - foo;
    rename_variables(ex, {{"x", "a"}, {"y", "b"}});
    REQUIRE(ex == ("b"_var + "a"_var * z) / ("a"_var - z - 5_dbl) * ("a"_var * z) - ("a"_var - z - 5_dbl));
}

TEST_CASE("copy")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    auto bar_copy = copy(bar);

    // Copy created a new object.
    REQUIRE(std::get<func>(bar_copy.value()).get_ptr() != std::get<func>(bar.value()).get_ptr());

    // foo was deep-copied into bar_copy.
    REQUIRE(std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[0].value()).args()[0].value()).get_ptr()
            == std::get<func>(foo.value()).get_ptr());
    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_copy.value()).args()[0].value()).args()[0].value()).get_ptr()
        != std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[0].value()).args()[0].value()).get_ptr());

    REQUIRE(std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[1].value()).args()[1].value()).get_ptr()
            == std::get<func>(foo.value()).get_ptr());
    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_copy.value()).args()[1].value()).args()[1].value()).get_ptr()
        != std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[1].value()).args()[1].value()).get_ptr());

    // The foo appearing in bar_copy is the same object, not two separate copies.
    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_copy.value()).args()[1].value()).args()[1].value()).get_ptr()
        == std::get<func>(std::get<func>(std::get<func>(bar_copy.value()).args()[0].value()).args()[0].value())
               .get_ptr());

    // A test in which a function has the same argument twice.
    bar = foo + foo;
    bar_copy = copy(bar);

    REQUIRE(std::get<func>(std::get<func>(bar.value()).args()[0].value()).get_ptr()
            == std::get<func>(foo.value()).get_ptr());
    REQUIRE(std::get<func>(std::get<func>(bar.value()).args()[0].value()).get_ptr()
            == std::get<func>(std::get<func>(bar.value()).args()[1].value()).get_ptr());
    REQUIRE(std::get<func>(std::get<func>(bar_copy.value()).args()[0].value()).get_ptr()
            == std::get<func>(std::get<func>(bar_copy.value()).args()[1].value()).get_ptr());
    REQUIRE(std::get<func>(std::get<func>(bar_copy.value()).args()[0].value()).get_ptr()
            != std::get<func>(std::get<func>(bar.value()).args()[1].value()).get_ptr());
}

TEST_CASE("subs")
{
    auto [x, y, z, a] = make_vars("x", "y", "z", "a");

    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);
    const auto foo_id = std::get<func>(foo.value()).get_ptr();
    const auto bar_id = std::get<func>(bar.value()).get_ptr();

    auto foo_a = ((a + y) * (z + a)) * ((z - a) * (y + a)), bar_a = (foo_a - a) / (2. * foo_a);

    auto bar_subs = subs(bar, {{"x", a}});

    // Ensure foo/bar were not modified.
    REQUIRE(foo == ((x + y) * (z + x)) * ((z - x) * (y + x)));
    REQUIRE(bar == (foo - x) / (2. * foo));
    REQUIRE(std::get<func>(foo.value()).get_ptr() == foo_id);
    REQUIRE(std::get<func>(bar.value()).get_ptr() == bar_id);
    REQUIRE(std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[0].value()).args()[0].value()).get_ptr()
            == foo_id);
    REQUIRE(std::get<func>(std::get<func>(std::get<func>(bar.value()).args()[1].value()).args()[1].value()).get_ptr()
            == foo_id);

    // Check the substitution.
    REQUIRE(bar_subs == bar_a);

    // Check that after substitution, what used to be foo in bar is a shared subexpression in bar_subs.
    REQUIRE(
        std::get<func>(std::get<func>(std::get<func>(bar_subs.value()).args()[0].value()).args()[0].value()).get_ptr()
        == std::get<func>(std::get<func>(std::get<func>(bar_subs.value()).args()[1].value()).args()[1].value())
               .get_ptr());
}

// cfunc N-body with fixed masses.
TEST_CASE("cfunc nbody")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<double> outs, ins;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto batch_size : {1u, 2u, 4u, 5u}) {
                llvm_state s{kw::opt_level = opt_level};

                outs.resize(36u * batch_size);
                ins.resize(36u * batch_size);

                std::generate(ins.begin(), ins.end(), gen);

                add_cfunc<double>(s, "cfunc", exs, kw::batch_size = batch_size, kw::compact_mode = cm);

                s.compile();

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), ins.data(), nullptr);

                for (auto i = 0u; i < 6u; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        // x_i' == vx_i.
                        REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], 100.));
                        // y_i' == vy_i.
                        REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                                == approximately(ins[i * batch_size + batch_size * 6u + j], 100.));
                        // z_i' == vz_i.
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                                == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], 100.));

                        // Accelerations.
                        auto acc_x = 0., acc_y = 0., acc_z = 0.;

                        const auto xi = ins[18u * batch_size + i * batch_size + j];
                        const auto yi = ins[24u * batch_size + i * batch_size + j];
                        const auto zi = ins[30u * batch_size + i * batch_size + j];

                        for (auto k = 0u; k < 6u; ++k) {
                            if (k == i) {
                                continue;
                            }

                            const auto xk = ins[18u * batch_size + k * batch_size + j];
                            const auto dx = xk - xi;

                            const auto yk = ins[24u * batch_size + k * batch_size + j];
                            const auto dy = yk - yi;

                            const auto zk = ins[30u * batch_size + k * batch_size + j];
                            const auto dz = zk - zi;

                            const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                            acc_x += dx * G * masses[k] * rm3;
                            acc_y += dy * G * masses[k] * rm3;
                            acc_z += dz * G * masses[k] * rm3;
                        }

                        REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, 100.));
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, 100.));
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, 100.));
                    }
                }
            }
        }
    }
}

// cfunc N-body with parametric masses.
TEST_CASE("cfunc nbody par")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_par_sys(6, kw::Gconst = G);
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<double> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto batch_size : {1u, 2u, 4u, 5u}) {
                llvm_state s{kw::opt_level = opt_level};

                outs.resize(36u * batch_size);
                ins.resize(36u * batch_size);
                pars.resize(6u * batch_size);

                for (auto i = 0u; i < 6u; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        pars[i * batch_size + j] = masses[i];
                    }
                }

                std::generate(ins.begin(), ins.end(), gen);

                add_cfunc<double>(s, "cfunc", exs, kw::batch_size = batch_size, kw::compact_mode = cm);

                s.compile();

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), ins.data(), pars.data());

                for (auto i = 0u; i < 6u; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        // x_i' == vx_i.
                        REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], 100.));
                        // y_i' == vy_i.
                        REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                                == approximately(ins[i * batch_size + batch_size * 6u + j], 100.));
                        // z_i' == vz_i.
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                                == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], 100.));

                        // Accelerations.
                        auto acc_x = 0., acc_y = 0., acc_z = 0.;

                        const auto xi = ins[18u * batch_size + i * batch_size + j];
                        const auto yi = ins[24u * batch_size + i * batch_size + j];
                        const auto zi = ins[30u * batch_size + i * batch_size + j];

                        for (auto k = 0u; k < 6u; ++k) {
                            if (k == i) {
                                continue;
                            }

                            const auto xk = ins[18u * batch_size + k * batch_size + j];
                            const auto dx = xk - xi;

                            const auto yk = ins[24u * batch_size + k * batch_size + j];
                            const auto dy = yk - yi;

                            const auto zk = ins[30u * batch_size + k * batch_size + j];
                            const auto dz = zk - zi;

                            const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                            acc_x += dx * G * masses[k] * rm3;
                            acc_y += dy * G * masses[k] * rm3;
                            acc_z += dz * G * masses[k] * rm3;
                        }

                        REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, 1000.));
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, 1000.));
                        REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, 1000.));
                    }
                }

                // Run the test on the strided function too.
                const std::size_t extra_stride = 3;
                outs.resize(36u * (batch_size + extra_stride));
                ins.resize(36u * (batch_size + extra_stride));
                pars.resize(6u * (batch_size + extra_stride));

                for (auto i = 0u; i < 6u; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        pars[i * (batch_size + extra_stride) + j] = masses[i];
                    }
                }

                std::generate(ins.begin(), ins.end(), gen);

                auto *cfs_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, std::size_t)>(
                    s.jit_lookup("cfunc.strided"));

                cfs_ptr(outs.data(), ins.data(), pars.data(), batch_size + extra_stride);

                for (auto i = 0u; i < 6u; ++i) {
                    for (auto j = 0u; j < batch_size; ++j) {
                        // x_i' == vx_i.
                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + j]
                                == approximately(ins[i * (batch_size + extra_stride) + j], 100.));
                        // y_i' == vy_i.
                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) + j]
                                == approximately(
                                    ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u + j], 100.));
                        // z_i' == vz_i.
                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 2u + j]
                                == approximately(
                                    ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u * 2u + j],
                                    100.));

                        // Accelerations.
                        auto acc_x = 0., acc_y = 0., acc_z = 0.;

                        const auto xi = ins[18u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                        const auto yi = ins[24u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                        const auto zi = ins[30u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];

                        for (auto k = 0u; k < 6u; ++k) {
                            if (k == i) {
                                continue;
                            }

                            const auto xk
                                = ins[18u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                            const auto dx = xk - xi;

                            const auto yk
                                = ins[24u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                            const auto dy = yk - yi;

                            const auto zk
                                = ins[30u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                            const auto dz = zk - zi;

                            const auto rm3 = std::pow(dx * dx + dy * dy + dz * dz, -3 / 2.);

                            acc_x += dx * G * masses[k] * rm3;
                            acc_y += dy * G * masses[k] * rm3;
                            acc_z += dz * G * masses[k] * rm3;
                        }

                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 3u + j]
                                == approximately(acc_x, 1000.));
                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 4u + j]
                                == approximately(acc_y, 1000.));
                        REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 5u + j]
                                == approximately(acc_z, 1000.));
                    }
                }
            }
        }
    }
}

// A test in which all outputs are equal to numbers or params.
TEST_CASE("cfunc numparams")
{
    std::vector<double> outs, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto batch_size : {1u, 2u, 4u, 5u}) {
                llvm_state s{kw::opt_level = opt_level};

                outs.resize(2u * batch_size);
                pars.resize(batch_size);

                std::generate(pars.begin(), pars.end(), gen);

                add_cfunc<double>(s, "cfunc", {1_dbl, par[0]}, kw::batch_size = batch_size, kw::compact_mode = cm);

                s.compile();

                auto *cf_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), nullptr, pars.data());

                for (auto j = 0u; j < batch_size; ++j) {
                    REQUIRE(outs[j] == 1);
                    REQUIRE(outs[j + batch_size] == pars[j]);
                }

                // Run the test on the strided function too.
                const std::size_t extra_stride = 3;
                outs.resize(2u * (batch_size + extra_stride));
                pars.resize(batch_size + extra_stride);

                std::generate(pars.begin(), pars.end(), gen);

                auto *cfs_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, std::size_t)>(
                    s.jit_lookup("cfunc.strided"));

                cfs_ptr(outs.data(), nullptr, pars.data(), batch_size + extra_stride);

                for (auto j = 0u; j < batch_size; ++j) {
                    REQUIRE(outs[j] == 1);
                    REQUIRE(outs[j + batch_size + extra_stride] == pars[j]);
                }
            }
        }
    }
}

// A test with explicit variable list.
TEST_CASE("cfunc explicit")
{
    llvm_state s;

    auto [x, y, z] = make_vars("x", "y", "z");

    add_cfunc<double>(s, "cfunc", {x + 2_dbl * y + 3_dbl * z}, kw::vars = {z, y, x});

    s.compile();

    auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double out = 0;
    std::vector<double> ins = {10, 11, 12};

    cf_ptr(&out, ins.data(), nullptr);

    REQUIRE(out == 12. + 2. * 11 + 3. * 10);
}

// Test for stride values under the batch size.
TEST_CASE("cfunc bogus stride")
{
    std::vector<double> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&rdist]() { return rdist(rng); };

    auto [x, y, z] = make_vars("x", "y", "z");

    for (auto cm : {false, true}) {
        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            llvm_state s;

            outs.resize(2u * batch_size);
            ins.resize(3u * batch_size);
            pars.resize(2u * batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            add_cfunc<double>(s, "cfunc", {x + 2_dbl * y + par[0] * z, par[1] - x * y}, kw::batch_size = batch_size,
                              kw::compact_mode = cm);

            s.compile();

            auto *cfs_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, std::size_t)>(
                s.jit_lookup("cfunc.strided"));

            cfs_ptr(outs.data(), ins.data(), pars.data(), batch_size - 1u);

            if (batch_size > 1u) {
                for (auto j = 0u; j < batch_size - 1u; ++j) {
                    REQUIRE(outs[j]
                            == approximately(ins[j] + 2. * ins[(batch_size - 1u) + j]
                                                 + pars[j] * ins[(batch_size - 1u) * 2u + j],
                                             100.));
                    REQUIRE(outs[(batch_size - 1u) + j]
                            == approximately(pars[(batch_size - 1u) + j] - ins[j] * ins[(batch_size - 1u) + j], 100.));
                }

                cfs_ptr(outs.data(), ins.data(), pars.data(), 0);

                for (auto j = 0u; j < batch_size; ++j) {
                    REQUIRE(outs[j] == approximately(pars[j] - ins[j] * ins[j], 100.));
                }
            } else {
                REQUIRE(outs[0] == approximately(pars[0] - ins[0] * ins[0], 100.));
            }
        }
    }
}

TEST_CASE("cfunc failure modes")
{
    using Catch::Matchers::Message;

    {
        llvm_state s;

        s.compile();

        REQUIRE_THROWS_MATCHES(add_cfunc<double>(s, "cfunc", {1_dbl, par[0]}), std::invalid_argument,
                               Message("A compiled function cannot be added to an llvm_state after compilation"));
    }

    {
        llvm_state s;

        REQUIRE_THROWS_MATCHES(add_cfunc<double>(s, "cfunc", {1_dbl, par[0]}, kw::batch_size = 0u),
                               std::invalid_argument, Message("The batch size of a compiled function cannot be zero"));
    }

    {
        llvm_state s;

        REQUIRE_THROWS_MATCHES(add_cfunc<double>(s, "cfunc", {1_dbl, par[0]}, kw::parallel_mode = true),
                               std::invalid_argument,
                               Message("Parallel mode can only be enabled in conjunction with compact mode"));
    }

    {
        llvm_state s;

        REQUIRE_THROWS_MATCHES(
            add_cfunc<double>(s, "cfunc", {1_dbl, par[0]}, kw::parallel_mode = true, kw::compact_mode = true),
            std::invalid_argument, Message("Parallel mode has not been implemented yet"));
    }

#if defined(HEYOKA_ARCH_PPC)
    {
        llvm_state s;

        REQUIRE_THROWS_MATCHES(add_cfunc<long double>(s, "cfunc", {1_dbl, par[0]}), not_implemented_error,
                               Message("'long double' computations are not supported on PowerPC"));
    }
#endif
}

TEST_CASE("cfunc vsop2013")
{
    const auto thr = 1e-5;
    const auto date = 365. / 365250;

    const auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    const auto venus_sol1 = vsop2013_cartesian_icrf(2, kw::time = par[0], kw::thresh = thr);
    const auto venus_sol2 = vsop2013_cartesian_icrf(2, kw::time = t, kw::thresh = thr);

    auto ta = taylor_adaptive<double>({prime(x) = venus_sol1[0], prime(y) = venus_sol1[1], prime(z) = venus_sol1[2]},
                                      {0., 0., 0.}, kw::compact_mode = true);

    ta.get_pars_data()[0] = date;

    ta.propagate_until(1.);

    llvm_state s;

    add_cfunc<double>(s, "cfunc", {venus_sol2[0], venus_sol2[1], venus_sol2[2]}, kw::compact_mode = true);

    s.compile();

    auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double out[3] = {};

    cf_ptr(out, &date, nullptr);

    REQUIRE(out[0] == approximately(ta.get_state()[0], 100.));
    REQUIRE(out[1] == approximately(ta.get_state()[1], 100.));
    REQUIRE(out[2] == approximately(ta.get_state()[2], 100.));
}
