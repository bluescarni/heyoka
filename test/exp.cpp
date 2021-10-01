// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;

#include <iostream>

TEST_CASE("exp")
{
    auto [x] = make_vars("x");
    // Test the textual output
    std::ostringstream stream;
    stream << exp(x);
    REQUIRE(stream.str() == "exp(x)");
    // Test the expression evaluation
    REQUIRE(eval_dbl(exp(x), {{"x", 0.}}) == 1.);
    REQUIRE(eval_dbl(exp(x), {{"x", 1.}}) == std::exp(1));
    // Test the expression evaluation on batches
    std::vector<double> retval;
    eval_batch_dbl(retval, exp(x), {{"x", {0., 1., 2.}}});
    REQUIRE(retval == std::vector<double>{std::exp(0.), std::exp(1.), std::exp(2.)});
    // Test the automated differentiation (non Taylor, the standard one (backward implemented))
    auto ex = exp(x);
    auto connections = compute_connections(ex);
    std::unordered_map<std::string, double> point;
    point["x"] = 2.3;
    auto grad = compute_grad_dbl(ex, point, connections);
    REQUIRE(grad["x"] == std::exp(2.3));
}

TEST_CASE("exp diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(exp(x * x - y), x) == exp(x * x - y) * (2. * x));
    REQUIRE(diff(exp(x * x - y), y) == -exp(x * x - y));

    REQUIRE(diff(exp(par[0] * par[0] - y), par[0]) == exp(par[0] * par[0] - y) * (2. * par[0]));
    REQUIRE(diff(exp(x * x - par[1]), par[1]) == -exp(x * x - par[1]));
}

TEST_CASE("exp s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = exp(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == exp(x));
}
