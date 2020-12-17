// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

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
