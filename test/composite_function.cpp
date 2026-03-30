// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <variant>
#include <vector>

#include <heyoka/detail/composite_function.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

using detail::composite_function;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    const auto ex = expression{func{detail::composite_function_impl{}}};
    REQUIRE(std::get<func>(ex.value()).get_name() == "composite|null_func()");
    REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|null_func()");
    REQUIRE(std::get<func>(ex.value()).args().empty());

    REQUIRE_THROWS_MATCHES(
        composite_function(par[0]), std::invalid_argument,
        Message("Cannot construct a composite function from the expression 'p0': the expression is not a function"));
}

struct func_00 : func_base {
    func_00() : func_base("f", std::vector<expression>{}) {}
    func_00(const std::string &name) : func_base(name, std::vector<expression>{}) {}
};

struct func_01 : func_base {
    func_01() : func_base("f", std::vector<expression>{}) {}
    func_01(const std::string &name) : func_base(name, name + "#", std::vector<expression>{}) {}
};

TEST_CASE("name")
{
    using Catch::Matchers::Message;

    auto [x, y, z] = make_vars("x", "y", "z");

    {
        const auto ex = composite_function(sin(x));
        REQUIRE(std::get<func>(ex.value()).get_name() == "composite|sin(#0)");
        REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|sin(#0)");
    }

    {
        const auto ex = composite_function(sin(par[0]));
        REQUIRE(std::get<func>(ex.value()).get_name() == "composite|sin(#0)");
        REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|sin(#0)");
    }

    {
        const auto ex = composite_function(sin(x + y) + z);
        REQUIRE(std::get<func>(ex.value()).get_name() == "composite|sum(sin(sum(#0,#1)),#2)");
        REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|sum(sin(sum(#0,#1)),#2)");
    }

    {
        const auto ex = composite_function(3. + sin(par[1] + y));
        REQUIRE(std::get<func>(ex.value()).get_name() == "composite|sum(#0,sin(sum(#1,#2)))");
        REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|sum(#0,sin(sum(#1,#2)))");
    }

    {
        const auto ex = composite_function(3. + pow(par[1], .5));
        REQUIRE(std::get<func>(ex.value()).get_name() == "composite|sum(#0,pow(#1,#2))");
        REQUIRE(std::get<func>(ex.value()).get_llvm_name() == "composite|sum(#0,pow_pos_small_half_1(#1,#2))");
    }

    REQUIRE_THROWS_MATCHES(composite_function(expression(func(func_00("(")))), std::invalid_argument,
                           Message("Invalid character(s) detected in the function name '(' during the construction of "
                                   "a composite function: the characters '()#' are forbidden"));

    REQUIRE_THROWS_MATCHES(
        composite_function(expression(func(func_01("aa")))), std::invalid_argument,
        Message("Invalid character(s) detected in the llvm function name 'aa#' during the construction of "
                "a composite function: the characters '()#' are forbidden"));
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = composite_function(sin(x + y) + z);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = x;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == composite_function(sin(x + y) + z));
}

TEST_CASE("cfunc")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        cfunc<double> cf{{composite_function(sin(cos(x * y) - (x + z)))}, {x, y, z}};
        double out[] = {0};
        std::vector<double> in{1, 2, 3};

        cf(out, in);

        REQUIRE(cf.get_dc().size() == 5u);
        REQUIRE(out[0] == approximately(std::sin(std::cos(in[0] * in[1]) - (in[0] + in[2]))));
    }

    {
        cfunc<double> cf{{composite_function(sin(cos(par[0] * 3.) - (heyoka::time + z)))}, {z}};
        double out[] = {0};
        std::vector<double> in{3}, pars{1};

        cf(out, in, kw::pars = pars, kw::time = 2.);

        REQUIRE(cf.get_dc().size() == 3u);
        REQUIRE(out[0] == approximately(std::sin(std::cos(1. * 3.) - (2. + 3.))));
    }
}
