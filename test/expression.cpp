// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/div.hpp>
#include <heyoka/detail/sub.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/model/vsop2013.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;
using namespace Catch::literals;

// Helper to ease the removal of neg() in the test code.
auto neg(const expression &e)
{
    return -e;
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
    {
        expression ex1 = 1_dbl + cos("x"_var);
        expression ex2 = cos("x"_var) + 1_dbl;
        expression ex3 = cos("x"_var) + 1_dbl + ex1 - ex1;

        REQUIRE(ex1 == ex2);
        REQUIRE(ex3 == ex2);
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
}

TEST_CASE("sub simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(1_dbl - 2_dbl == -1_dbl);
    REQUIRE(1_ldbl - 3_dbl == -2_ldbl);

    REQUIRE(0_dbl - x == -x);

    REQUIRE((x + y) - 0_dbl == x + y);
    REQUIRE((x + y) - 1_dbl == x + y + -1_dbl);
}

TEST_CASE("mul simpls")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(2_dbl * 2_dbl == 4_dbl);

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
}

// Check that compound ops correctly
// propagate number types.
TEST_CASE("compound ops")
{
    auto x = 1.1_dbl;
    x *= 1.1_ldbl;

    REQUIRE(std::holds_alternative<long double>(std::get<number>(x.value()).value()));
    REQUIRE(x == expression(1.1l * 1.1));

#if defined(HEYOKA_HAVE_REAL128)

    x = 1.1_dbl;
    x *= 1.1_f128;

    REQUIRE(std::holds_alternative<mppp::real128>(std::get<number>(x.value()).value()));
    REQUIRE(x == expression(mppp::real128{"1.1"} * 1.1));

#endif

#if defined(HEYOKA_HAVE_REAL)

    x = expression{mppp::real{"1.1", 128}};
    x *= expression{mppp::real{"1.1", 256}};

    REQUIRE(std::holds_alternative<mppp::real>(std::get<number>(x.value()).value()));
    REQUIRE(x == expression(mppp::real{"1.1", 128} * mppp::real{"1.1", 256}));

#endif
}

TEST_CASE("is_time_dependent")
{
    namespace hy = heyoka;

    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(!is_time_dependent(x));
    REQUIRE(!is_time_dependent(y));
    REQUIRE(!is_time_dependent(x + y));
    REQUIRE(!is_time_dependent(1_dbl));
    REQUIRE(!is_time_dependent(par[0]));
    REQUIRE(!is_time_dependent(2_dbl - par[0]));

    REQUIRE(is_time_dependent(hy::time));
    REQUIRE(is_time_dependent(hy::time + 1_dbl));
    REQUIRE(is_time_dependent(par[42] + hy::time));
    REQUIRE(is_time_dependent((x + y) * (hy::time + 1_dbl)));
    REQUIRE(is_time_dependent((x + y) * (par[0] * hy::time + 1_dbl)));

    // With common subexpressions.
    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    REQUIRE(!is_time_dependent(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = (foo - x) / (2. * foo);

    REQUIRE(is_time_dependent(bar));

    foo = hy::time + ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = (foo - x) / (2. * foo);

    REQUIRE(is_time_dependent(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = hy::time + (foo - x) / (2. * foo);

    REQUIRE(is_time_dependent(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x));
    bar = (foo - x) / (2. * foo) + hy::time;

    REQUIRE(is_time_dependent(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = (foo - x) / (2. * foo) + hy::time;

    REQUIRE(is_time_dependent(bar));

    foo = ((x + y) * (z + x)) * ((z - x) * (y + x)) + hy::time;
    bar = hy::time + (foo - x) / (2. * foo) + hy::time;

    REQUIRE(is_time_dependent(bar));
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
}

TEST_CASE("get_n_nodes")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(get_n_nodes(expression{}) == 1u);
    REQUIRE(get_n_nodes(1_dbl) == 1u);
    REQUIRE(get_n_nodes(par[0]) == 1u);
    REQUIRE(get_n_nodes(x) == 1u);

    REQUIRE(get_n_nodes(x + y) == 3u);
    REQUIRE(get_n_nodes(x - y) == 5u);
    REQUIRE(get_n_nodes(-z) == 3u);
    REQUIRE(get_n_nodes(heyoka::time) == 1u);
    REQUIRE(get_n_nodes(x + (y * z)) == 5u);
    REQUIRE(get_n_nodes((x - y - z) + (y * z)) == 11u);

    // Try with subexpressions repeating in the tree.
    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo),
         bar2 = (copy(foo) - x) / (2. * copy(foo));

    REQUIRE(get_n_nodes(bar) == 37u);
    REQUIRE(get_n_nodes(bar2) == 37u);

    // Build a very large expression such that
    // get_n_nodes() will return 0.
    // NOTE: this has been calibrated for a 64-bit size_t.
    for (auto i = 0; i < 6; ++i) {
        foo = subs(foo, {{x, foo}});
    }

    REQUIRE(get_n_nodes(foo) == 0u);
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

    // The vectorised version
    REQUIRE(get_variables({(y + tmp) / foo * tmp - foo, "a"_var + "b"_var})
            == std::vector<std::string>{"a", "b", "x", "y", "z"});
}

TEST_CASE("rename_variables")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = 1_dbl;
    ex = rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == 1_dbl);

    ex = par[0];
    ex = rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == par[0]);

    ex = x;
    ex = rename_variables(ex, {{"x", "a"}});
    REQUIRE(ex == "a"_var);

    ex = x + y;
    ex = rename_variables(ex, {{"x", "a"}, {"y", "b"}});
    REQUIRE(ex == "a"_var + "b"_var);

    auto tmp = x * z, foo = x - z - 5_dbl;
    ex = (y + tmp) / foo * tmp - foo;
    ex = rename_variables(ex, {{"x", "a"}, {"y", "b"}});
    REQUIRE(ex == ("b"_var + "a"_var * z) / ("a"_var - z - 5_dbl) * ("a"_var * z) - ("a"_var - z - 5_dbl));
}

TEST_CASE("copy")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);

    auto bar_copy = copy(bar);

    // Copy created a new object.
    REQUIRE(std::get<func>(bar_copy.value()).get_ptr() != std::get<func>(bar.value()).get_ptr());
}

TEST_CASE("subs str")
{
    auto [x, y, z, a] = make_vars("x", "y", "z", "a");

    auto foo = ((x + y) * (z + x)) * ((z - x) * (y + x)), bar = (foo - x) / (2. * foo);
    const auto *foo_id = std::get<func>(foo.value()).get_ptr();
    const auto *bar_id = std::get<func>(bar.value()).get_ptr();

    auto foo_a = ((a + y) * (z + a)) * ((z - a) * (y + a)), bar_a = (foo_a - a) / (2. * foo_a);

    auto bar_subs = subs(bar, {{"x", a}});

    // Ensure foo/bar were not modified.
    REQUIRE(foo == ((x + y) * (z + x)) * ((z - x) * (y + x)));
    REQUIRE(bar == (foo - x) / (2. * foo));
    REQUIRE(std::get<func>(foo.value()).get_ptr() == foo_id);
    REQUIRE(std::get<func>(bar.value()).get_ptr() == bar_id);

    // Check the substitution.
    REQUIRE(bar_subs == bar_a);

    // Check normalisation.
    REQUIRE(subs(x + y, {{"x", "b"_var}, {"y", "a"_var}}, true) == "a"_var + "b"_var);
    REQUIRE(subs(std::vector{x + y}, {{"x", "b"_var}, {"y", "a"_var}}, true)[0] == "a"_var + "b"_var);
    REQUIRE(subs(std::vector{x + y}, {{"x", "b"_var}, {"y", "a"_var}})[0] != "a"_var + "b"_var);
}

TEST_CASE("subs")
{
    auto [x, y, z, a] = make_vars("x", "y", "z", "a");

    REQUIRE(subs(x, {{z, x + y}}) == x);
    REQUIRE(subs(1_dbl, {{z, x + y}}) == 1_dbl);
    REQUIRE(subs(x, {{x, x + y}}) == x + y);
    REQUIRE(subs(1_dbl, {{1_dbl, x + y}}) == x + y);

    REQUIRE(subs(x + y, {{x + y, z}}) == z);
    REQUIRE(subs(x + z, {{x + y, z}}) == x + z);

    auto tmp = x + y;
    auto tmp2 = x - y;
    auto ex = tmp - par[0] * tmp;
    auto subs_res = subs(ex, {{tmp, tmp2}});

    REQUIRE(subs_res == tmp - par[0] * tmp2);

    // Check normalisation.
    REQUIRE(subs(x + y, {{x, "b"_var}, {y, "a"_var}}, true) == "a"_var + "b"_var);
    REQUIRE(subs(std::vector{x + y}, {{x, "b"_var}, {y, "a"_var}}, true)[0] == "a"_var + "b"_var);
    REQUIRE(subs(std::vector{x + y}, {{x, "b"_var}, {y, "a"_var}})[0] != "a"_var + "b"_var);
}

// cfunc N-body with fixed masses.
TEST_CASE("cfunc nbody")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);
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

                auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                    s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc nbody mp")
{
    using std::pow;

    const auto prec = 237u;

    auto masses
        = std::vector{mppp::real{1.00000597682, prec}, mppp::real{1 / 1047.355, prec}, mppp::real{1 / 3501.6, prec},
                      mppp::real{1 / 22869., prec},    mppp::real{1 / 19314., prec},   mppp::real{7.4074074e-09, prec}};

    const auto G = mppp::real{0.01720209895 * 0.01720209895 * 365 * 365, prec};

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<mppp::real> outs, ins;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    const auto batch_size = 1u;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            llvm_state s{kw::opt_level = opt_level};

            outs.resize(36u * batch_size);
            ins.resize(36u * batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);

            add_cfunc<mppp::real>(s, "cfunc", exs, kw::prec = prec, kw::compact_mode = cm);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], mppp::real{100.}));
                    // y_i' == vy_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                            == approximately(ins[i * batch_size + batch_size * 6u + j], mppp::real{100.}));
                    // z_i' == vz_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                            == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], mppp::real{100.}));

                    // Accelerations.
                    mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

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

                        const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, mppp::real{100.}));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, mppp::real{100.}));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, mppp::real{100.}));
                }
            }
        }
    }
}

#endif

// cfunc N-body with parametric masses.
TEST_CASE("cfunc nbody par")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = model::nbody(6, kw::Gconst = G, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]});
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

                auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                    s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

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

                auto *cfs_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                        s.jit_lookup("cfunc.strided"));

                cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, batch_size + extra_stride);

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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc nbody par mp")
{
    using std::pow;

    const auto prec = 237u;

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = mppp::real{0.01720209895 * 0.01720209895 * 365 * 365, prec};

    auto sys = model::nbody(6, kw::Gconst = G, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]});
    std::vector<expression> exs;
    for (const auto &p : sys) {
        exs.push_back(p.second);
    }

    std::vector<mppp::real> outs, ins, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    const auto batch_size = 1u;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            llvm_state s{kw::opt_level = opt_level};

            outs.resize(36u * batch_size);
            ins.resize(36u * batch_size);
            pars.resize(6u * batch_size);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    pars[i * batch_size + j] = mppp::real{masses[i], prec};
                }
            }

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);

            add_cfunc<mppp::real>(s, "cfunc", exs, kw::prec = prec, kw::compact_mode = cm);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * batch_size * 6u + j] == approximately(ins[i * batch_size + j], mppp::real{100.}));
                    // y_i' == vy_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size + j]
                            == approximately(ins[i * batch_size + batch_size * 6u + j], mppp::real{100.}));
                    // z_i' == vz_i.
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 2u + j]
                            == approximately(ins[i * batch_size + batch_size * 6u * 2u + j], mppp::real{100.}));

                    // Accelerations.
                    mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

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

                        const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * batch_size * 6u + batch_size * 3u + j] == approximately(acc_x, mppp::real{1000.}));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 4u + j] == approximately(acc_y, mppp::real{1000.}));
                    REQUIRE(outs[i * batch_size * 6u + batch_size * 5u + j] == approximately(acc_z, mppp::real{1000.}));
                }
            }

            // Run the test on the strided function too.
            const std::size_t extra_stride = 3;
            outs.resize(36u * (batch_size + extra_stride));
            ins.resize(36u * (batch_size + extra_stride));
            pars.resize(6u * (batch_size + extra_stride));

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    pars[i * (batch_size + extra_stride) + j] = mppp::real{masses[i], prec};
                }
            }

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);

            auto *cfs_ptr = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *,
                                                      const mppp::real *, std::size_t)>(s.jit_lookup("cfunc.strided"));

            cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, batch_size + extra_stride);

            for (auto i = 0u; i < 6u; ++i) {
                for (auto j = 0u; j < batch_size; ++j) {
                    // x_i' == vx_i.
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + j]
                            == approximately(ins[i * (batch_size + extra_stride) + j], mppp::real{100.}));
                    // y_i' == vy_i.
                    REQUIRE(
                        outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) + j]
                        == approximately(ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u + j],
                                         mppp::real{100.}));
                    // z_i' == vz_i.
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 2u + j]
                            == approximately(
                                ins[i * (batch_size + extra_stride) + (batch_size + extra_stride) * 6u * 2u + j],
                                mppp::real{100.}));

                    // Accelerations.
                    mppp::real acc_x{0., prec}, acc_y{0., prec}, acc_z{0., prec};

                    const auto xi = ins[18u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                    const auto yi = ins[24u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];
                    const auto zi = ins[30u * (batch_size + extra_stride) + i * (batch_size + extra_stride) + j];

                    for (auto k = 0u; k < 6u; ++k) {
                        if (k == i) {
                            continue;
                        }

                        const auto xk = ins[18u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dx = xk - xi;

                        const auto yk = ins[24u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dy = yk - yi;

                        const auto zk = ins[30u * (batch_size + extra_stride) + k * (batch_size + extra_stride) + j];
                        const auto dz = zk - zi;

                        const auto rm3 = pow(dx * dx + dy * dy + dz * dz, mppp::real{-3 / 2., prec});

                        acc_x += dx * G * masses[k] * rm3;
                        acc_y += dy * G * masses[k] * rm3;
                        acc_z += dz * G * masses[k] * rm3;
                    }

                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 3u + j]
                            == approximately(acc_x, mppp::real{1000.}));
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 4u + j]
                            == approximately(acc_y, mppp::real{1000.}));
                    REQUIRE(outs[i * (batch_size + extra_stride) * 6u + (batch_size + extra_stride) * 5u + j]
                            == approximately(acc_z, mppp::real{1000.}));
                }
            }
        }
    }
}

#endif

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

                auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                    s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), nullptr, pars.data(), nullptr);

                for (auto j = 0u; j < batch_size; ++j) {
                    REQUIRE(outs[j] == 1);
                    REQUIRE(outs[j + batch_size] == pars[j]);
                }

                // Run the test on the strided function too.
                const std::size_t extra_stride = 3;
                outs.resize(2u * (batch_size + extra_stride));
                pars.resize(batch_size + extra_stride);

                std::generate(pars.begin(), pars.end(), gen);

                auto *cfs_ptr
                    = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                        s.jit_lookup("cfunc.strided"));

                cfs_ptr(outs.data(), nullptr, pars.data(), nullptr, batch_size + extra_stride);

                for (auto j = 0u; j < batch_size; ++j) {
                    REQUIRE(outs[j] == 1);
                    REQUIRE(outs[j + batch_size + extra_stride] == pars[j]);
                }
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc numparams mp")
{
    const auto prec = 237u;

    const auto batch_size = 1u;

    std::vector<mppp::real> outs, pars;

    std::uniform_real_distribution<double> rdist(-1., 1.);

    auto gen = [&]() { return mppp::real{rdist(rng), static_cast<int>(prec)}; };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {

            llvm_state s{kw::opt_level = opt_level};

            outs.resize(4u * batch_size);
            pars.resize(2u * batch_size);

            std::generate(pars.begin(), pars.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);

            add_cfunc<mppp::real>(s, "cfunc", {1_dbl, par[0], par[1], -2_dbl}, kw::prec = prec, kw::compact_mode = cm);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), nullptr, pars.data(), nullptr);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(outs[j] == 1);
                REQUIRE(outs[j + batch_size] == pars[j]);
                REQUIRE(outs[j + 2u * batch_size] == pars[j + 1u]);
                REQUIRE(outs[j + 3u * batch_size] == -2);
            }

            // Run the test on the strided function too.
            const std::size_t extra_stride = 3;
            outs.resize(4u * (batch_size + extra_stride));
            pars.resize(2u * (batch_size + extra_stride));

            std::generate(pars.begin(), pars.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);

            auto *cfs_ptr = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *,
                                                      const mppp::real *, std::size_t)>(s.jit_lookup("cfunc.strided"));

            cfs_ptr(outs.data(), nullptr, pars.data(), nullptr, batch_size + extra_stride);

            for (auto j = 0u; j < batch_size; ++j) {
                REQUIRE(outs[j] == 1);
                REQUIRE(outs[j + batch_size + extra_stride] == pars[j]);
                REQUIRE(outs[j + 2u * (batch_size + extra_stride)] == pars[j + batch_size + extra_stride]);
                REQUIRE(outs[j + 3u * (batch_size + extra_stride)] == -2);
            }
        }
    }
}

#endif

// A test with explicit variable list.
TEST_CASE("cfunc explicit")
{
    llvm_state s;

    auto [x, y, z] = make_vars("x", "y", "z");

    add_cfunc<double>(s, "cfunc", {x + 2_dbl * y + 3_dbl * z}, kw::vars = {z, y, x});

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double out = 0;
    std::vector<double> ins = {10, 11, 12};

    cf_ptr(&out, ins.data(), nullptr, nullptr);

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

            auto *cfs_ptr
                = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                    s.jit_lookup("cfunc.strided"));

            cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, batch_size - 1u);

            if (batch_size > 1u) {
                for (auto j = 0u; j < batch_size - 1u; ++j) {
                    REQUIRE(outs[j]
                            == approximately(ins[j] + 2. * ins[(batch_size - 1u) + j]
                                                 + pars[j] * ins[(batch_size - 1u) * 2u + j],
                                             100.));
                    REQUIRE(outs[(batch_size - 1u) + j]
                            == approximately(pars[(batch_size - 1u) + j] - ins[j] * ins[(batch_size - 1u) + j], 100.));
                }

                cfs_ptr(outs.data(), ins.data(), pars.data(), nullptr, 0);

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

#if defined(HEYOKA_HAVE_REAL)

    {
        llvm_state s;

        REQUIRE_THROWS_MATCHES(add_cfunc<mppp::real>(s, "cfunc", {1_dbl, par[0]}), std::invalid_argument,
                               Message(fmt::format("An invalid precision value of 0 was passed to add_cfunc() (the "
                                                   "value must be in the [{}, {}] range)",
                                                   mppp::real_prec_min(), mppp::real_prec_max())));
    }

#endif
}

TEST_CASE("cfunc vsop2013")
{
    const auto thr = 1e-5;
    const auto date = 365. / 365250;

    const auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    const auto venus_sol1 = model::vsop2013_cartesian_icrf(2, kw::time = par[0], kw::thresh = thr);
    const auto venus_sol2 = model::vsop2013_cartesian_icrf(2, kw::time = t, kw::thresh = thr);

    auto ta = taylor_adaptive<double>({prime(x) = venus_sol1[0], prime(y) = venus_sol1[1], prime(z) = venus_sol1[2]},
                                      {0., 0., 0.}, kw::compact_mode = true);

    ta.get_pars_data()[0] = date;

    ta.propagate_until(1.);

    llvm_state s;

    add_cfunc<double>(s, "cfunc", {venus_sol2[0], venus_sol2[1], venus_sol2[2]}, kw::compact_mode = true);

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double out[3] = {};

    cf_ptr(out, &date, nullptr, nullptr);

    REQUIRE(out[0] == approximately(ta.get_state()[0], 100.));
    REQUIRE(out[1] == approximately(ta.get_state()[1], 100.));
    REQUIRE(out[2] == approximately(ta.get_state()[2], 100.));
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("mp interop")
{
    using namespace mppp::literals;

    auto [x] = make_vars("x");

    REQUIRE(std::get<mppp::real>(std::get<number>(expression{1.1_r256}.value()).value()) == 1.1_r256);

    REQUIRE(x + 1.1_r256 == x + expression{1.1_r256});
    REQUIRE(1.1_r256 + x == expression{1.1_r256} + x);

    REQUIRE(x - 1.1_r256 == x - expression{1.1_r256});
    REQUIRE(1.1_r256 - x == expression{1.1_r256} - x);

    REQUIRE(x * 1.1_r256 == x * expression{1.1_r256});
    REQUIRE(1.1_r256 * x == expression{1.1_r256} * x);

    REQUIRE(x / 1.1_r256 == x / expression{1.1_r256});
    REQUIRE(1.1_r256 / x == expression{1.1_r256} / x);

    x = expression{"x"};
    auto x2 = expression{"x"};
    x += 1.1_r256;
    x2 += expression{1.1_r256};
    REQUIRE(x == x2);

    x = expression{"x"};
    x2 = expression{"x"};
    x -= 1.1_r256;
    x2 -= expression{1.1_r256};
    REQUIRE(x == x2);

    x = expression{"x"};
    x2 = expression{"x"};
    x *= 1.1_r256;
    x2 *= expression{1.1_r256};
    REQUIRE(x == x2);

    x = expression{"x"};
    x2 = expression{"x"};
    x /= 1.1_r256;
    x2 /= expression{1.1_r256};
    REQUIRE(x == x2);
}

#endif

TEST_CASE("output too long")
{
    auto [x] = make_vars("x");

    auto ex = 1. + x;

    for (auto i = 0; i < 1000; ++i) {
        ex = expression(variable(fmt::format("x_{}", i))) * ex;
    }

    std::ostringstream oss;
    oss << ex;
    auto str = oss.str();

    REQUIRE(str.size() > 3u);
    REQUIRE(str[str.size() - 1u] == '.');
    REQUIRE(str[str.size() - 2u] == '.');
    REQUIRE(str[str.size() - 3u] == '.');
}

TEST_CASE("get_params")
{
    REQUIRE(get_params(expression{}).empty());
    REQUIRE(get_params("x"_var).empty());
    REQUIRE(get_params("x"_var + "y"_var).empty());
    REQUIRE(get_params(par[0]) == std::vector{par[0]});
    REQUIRE(get_params("x"_var - par[10]) == std::vector{par[10]});
    REQUIRE(get_params(("x"_var + par[1]) - ("y"_var - par[10])) == std::vector{par[1], par[10]});

    auto tmp1 = "x"_var + par[3];
    auto tmp2 = par[56] / "y"_var;
    auto ex = "z"_var * (tmp1 - tmp2) + "y"_var * tmp1 / tmp2;

    REQUIRE(get_params(ex) == std::vector{par[3], par[56]});

    // Test the vectorised version too.
    auto ex2 = 3_dbl + par[4];
    REQUIRE(get_params({ex, ex2}) == std::vector{par[3], par[4], par[56]});
}

TEST_CASE("swap")
{
    using std::swap;

    REQUIRE(std::is_nothrow_swappable_v<expression>);

    auto [x, y] = make_vars("x", "y");

    swap(x, y);

    REQUIRE(x == "y"_var);
    REQUIRE(y == "x"_var);
}

TEST_CASE("move semantics")
{
    REQUIRE(std::is_nothrow_move_assignable_v<expression>);
    REQUIRE(std::is_nothrow_move_constructible_v<expression>);

    auto [x, y] = make_vars("x", "y");

    // Check that move construction sets the moved-from
    // object to zero.
    auto ex = x + y;
    auto ex2(std::move(ex));
    REQUIRE(ex2 == x + y);
    REQUIRE(ex == 0_dbl);

    // Check that move assignment sets the moved-from
    // object to zero.
    auto ex3 = 1_dbl;
    ex3 = std::move(ex2);
    REQUIRE(ex3 == x + y);
    REQUIRE(ex2 == 0_dbl);
}

TEST_CASE("less than")
{
    REQUIRE(std::less<expression>{}("x"_var, "y"_var));
    REQUIRE(!std::less<expression>{}("x"_var, "x"_var));
    REQUIRE(!std::less<expression>{}("y"_var, "x"_var));

    REQUIRE(std::less<expression>{}(par[0], par[1]));
    REQUIRE(!std::less<expression>{}(par[0], par[0]));
    REQUIRE(!std::less<expression>{}(par[1], par[0]));

    REQUIRE(std::less<expression>{}(2_dbl, 3_dbl));
    REQUIRE(!std::less<expression>{}(2_dbl, 2_dbl));
    REQUIRE(!std::less<expression>{}(3_dbl, 2_dbl));

    REQUIRE(std::less<expression>{}(par[0], "x"_var + 1_dbl));
    REQUIRE(std::less<expression>{}(1_dbl, "x"_var + 1_dbl));
    REQUIRE(std::less<expression>{}("y"_var, "x"_var + 1_dbl));
    REQUIRE(!std::less<expression>{}("x"_var + 1_dbl, "x"_var + 1_dbl));
    REQUIRE(!std::less<expression>{}("x"_var + 2_dbl, "x"_var + 1_dbl));

    REQUIRE(std::less<expression>{}(2_dbl, par[0]));
    REQUIRE(std::less<expression>{}(2_dbl, "x"_var));
    REQUIRE(std::less<expression>{}(2_dbl, "x"_var + "y"_var));

    REQUIRE(!std::less<expression>{}("x"_var + "y"_var, 2_dbl));
    REQUIRE(!std::less<expression>{}("x"_var + "y"_var, par[0]));
    REQUIRE(!std::less<expression>{}("x"_var + "y"_var, "x"_var));

    REQUIRE(std::less<expression>{}("x"_var, "x"_var + "y"_var));
    REQUIRE(!std::less<expression>{}("x"_var, 2_dbl));
    REQUIRE(!std::less<expression>{}("x"_var, par[0]));

    REQUIRE(!std::less<expression>{}(par[0], 2_dbl));
    REQUIRE(std::less<expression>{}(par[0], "x"_var));
    REQUIRE(std::less<expression>{}(par[0], "x"_var + "y"_var));
}

TEST_CASE("mul compress")
{
    auto [x] = make_vars("x");

    REQUIRE(2_dbl * x + 3_dbl * x == 5_dbl * x);
    REQUIRE(2_dbl * x + x == 3_dbl * x);
    REQUIRE(x + 2_dbl * x == 3_dbl * x);
    REQUIRE(x - 2_dbl * x == -1_dbl * x);
}

TEST_CASE("cfunc prod_to_div")
{
    auto [x, y] = make_vars("x", "y");

    {
        llvm_state s;

        const auto dc = add_cfunc<double>(s, "cfunc", {prod({3_dbl, pow(y, -1_dbl), pow(x, -1.5_dbl)})});

        REQUIRE(dc.size() == 6u);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::div_impl>() != nullptr;
                              })
                == 1);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::pow_impl>() != nullptr;
                              })
                == 1);

        for (const auto &ex : dc) {
            if (std::holds_alternative<func>(ex.value())
                && std::get<func>(ex.value()).extract<detail::pow_impl>() != nullptr) {
                REQUIRE(!is_negative(std::get<number>(std::get<func>(ex.value()).args()[1].value())));
            }
        }
    }

    {
        llvm_state s;

        const auto dc
            = add_cfunc<double>(s, "cfunc", {prod({3_dbl, pow(y, -1_dbl), pow(x, -1.5_dbl)})}, kw::vars = {x, y});

        REQUIRE(dc.size() == 6u);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::div_impl>() != nullptr;
                              })
                == 1);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::pow_impl>() != nullptr;
                              })
                == 1);

        for (const auto &ex : dc) {
            if (std::holds_alternative<func>(ex.value())
                && std::get<func>(ex.value()).extract<detail::pow_impl>() != nullptr) {
                REQUIRE(!is_negative(std::get<number>(std::get<func>(ex.value()).args()[1].value())));
            }
        }
    }
}

TEST_CASE("cfunc sum_to_sub")
{
    auto [x, y] = make_vars("x", "y");

    {
        llvm_state s;

        const auto dc = add_cfunc<double>(s, "cfunc", {sum({1_dbl, prod({-1_dbl, x}), prod({-1_dbl, y})})});

        REQUIRE(dc.size() == 5u);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::sub_impl>() != nullptr;
                              })
                == 1);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::sum_impl>() != nullptr;
                              })
                == 1);
    }

    {
        llvm_state s;

        const auto dc
            = add_cfunc<double>(s, "cfunc", {sum({1_dbl, prod({-1_dbl, x}), prod({-1_dbl, y})})}, kw::vars = {x, y});

        REQUIRE(dc.size() == 5u);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::sub_impl>() != nullptr;
                              })
                == 1);
        REQUIRE(std::count_if(dc.begin(), dc.end(),
                              [](const auto &ex) {
                                  return std::holds_alternative<func>(ex.value())
                                         && std::get<func>(ex.value()).template extract<detail::sum_impl>() != nullptr;
                              })
                == 1);
    }
}

// Test to check that the optimisation for hashing non-recursive
// expressions is correct.
TEST_CASE("hash opt")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    // Non-recursive function.
    {
        auto ex = x + y;

        const auto nr_hash = std::hash<expression>{}(ex);

        auto ex2 = z * ex;

        const auto full_hash = std::hash<expression>{}(ex2);

        // Compute manually the hash of ex2.
        std::size_t seed = std::hash<std::string>{}(std::get<func>(ex2.value()).get_name());

        boost::hash_combine(seed, std::hash<variable>{}(std::get<variable>(z.value())));
        boost::hash_combine(seed, nr_hash);

        REQUIRE(seed == full_hash);
    }

    // Variable.
    {
        auto ex = x;

        const auto nr_hash = std::hash<expression>{}(ex);

        auto ex2 = ex + 2. * y;

        const auto full_hash = std::hash<expression>{}(ex2);

        // Compute manually the hash of ex2.
        std::size_t seed = std::hash<std::string>{}(std::get<func>(ex2.value()).get_name());

        boost::hash_combine(seed, nr_hash);
        boost::hash_combine(seed, std::hash<expression>{}(2. * y));

        REQUIRE(seed == full_hash);
    }

    // Parameter.
    {
        auto ex = par[0];

        const auto nr_hash = std::hash<expression>{}(ex);

        auto ex2 = ex + 2. * y;

        const auto full_hash = std::hash<expression>{}(ex2);

        // Compute manually the hash of ex2.
        std::size_t seed = std::hash<std::string>{}(std::get<func>(ex2.value()).get_name());

        boost::hash_combine(seed, nr_hash);
        boost::hash_combine(seed, std::hash<expression>{}(2. * y));

        REQUIRE(seed == full_hash);
    }

    // Number.
    {
        auto ex = 1.1_dbl;

        const auto nr_hash = std::hash<expression>{}(ex);

        auto ex2 = ex + 2. * y;

        const auto full_hash = std::hash<expression>{}(ex2);

        // Compute manually the hash of ex2.
        std::size_t seed = std::hash<std::string>{}(std::get<func>(ex2.value()).get_name());

        boost::hash_combine(seed, nr_hash);
        boost::hash_combine(seed, std::hash<expression>{}(2. * y));

        REQUIRE(seed == full_hash);
    }
}
