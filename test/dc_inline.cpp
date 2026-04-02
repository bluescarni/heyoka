// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("double output")
{
    const auto [x, y] = make_vars("x", "y");

    const auto ex = x * y + atan2(x + y, x * y) + 1.;

    const auto cf = cfunc<double>({ex, ex}, {x, y}, kw::compact_mode = true);

    const std::vector inputs = {1., 2.};
    std::vector<double> outputs(2);

    REQUIRE(cf.get_dc().size() == 6u);

    cf(outputs, inputs);

    REQUIRE(outputs[0]
            == approximately(inputs[0] * inputs[1] + std::atan2(inputs[0] + inputs[1], inputs[0] * inputs[1]) + 1.));
    REQUIRE(outputs[1] == outputs[0]);
}

TEST_CASE("output reused")
{
    const auto [x, y] = make_vars("x", "y");

    const auto ex1 = x * y + atan2(x + y, x * y);
    const auto ex2 = ex1 + 1.;

    const auto cf = cfunc<double>({ex1, ex2}, {x, y}, kw::compact_mode = true);

    const std::vector inputs = {1., 2.};
    std::vector<double> outputs(2);

    REQUIRE(cf.get_dc().size() == 7u);

    cf(outputs, inputs);

    REQUIRE(outputs[0]
            == approximately(inputs[0] * inputs[1] + std::atan2(inputs[0] + inputs[1], inputs[0] * inputs[1])));
    REQUIRE(outputs[1] == outputs[0] + 1.);
}

TEST_CASE("parnum outputs")
{
    const auto [x, y] = make_vars("x", "y");

    const auto ex1 = x * y + atan2(x + y, x * y);
    const auto ex2 = ex1 + 1.;

    const auto cf = cfunc<double>({ex1, ex2, par[0], 4_dbl}, {x, y}, kw::compact_mode = true);

    const std::vector inputs = {1., 2.};
    std::vector<double> outputs(4);

    REQUIRE(cf.get_dc().size() == 9u);

    cf(outputs, inputs, kw::pars = {3.});

    REQUIRE(outputs[0]
            == approximately(inputs[0] * inputs[1] + std::atan2(inputs[0] + inputs[1], inputs[0] * inputs[1])));
    REQUIRE(outputs[1] == outputs[0] + 1.);
    REQUIRE(outputs[2] == 3);
    REQUIRE(outputs[3] == 4);
}

TEST_CASE("time")
{
    const auto [x, y] = make_vars("x", "y");

    const auto ex1 = x * heyoka::time + atan2(x + heyoka::time, x * heyoka::time);
    const auto ex2 = ex1 + 1.;

    const auto cf = cfunc<double>({ex1, ex2, heyoka::time}, {x}, kw::compact_mode = true);

    const std::vector inputs = {1.};
    std::vector<double> outputs(3);

    REQUIRE(cf.get_dc().size() == 8u);

    cf(outputs, inputs, kw::time = 2.);

    REQUIRE(outputs[0] == approximately(inputs[0] * 2. + std::atan2(inputs[0] + 2., inputs[0] * 2.)));
    REQUIRE(outputs[1] == outputs[0] + 1.);
    REQUIRE(outputs[2] == 2);
}

TEST_CASE("parnum arguments")
{
    const auto [x, y] = make_vars("x", "y");

    const auto ex1 = x * par[0] + atan2(x + par[0], x * par[0]);
    const auto ex2 = ex1 + 1.;

    const auto cf = cfunc<double>({ex1, ex2}, {x}, kw::compact_mode = true);

    const std::vector inputs = {1.};
    std::vector<double> outputs(2);

    REQUIRE(cf.get_dc().size() == 6u);

    cf(outputs, inputs, kw::pars = {2.});

    REQUIRE(outputs[0] == approximately(inputs[0] * 2. + std::atan2(inputs[0] + 2., inputs[0] * 2.)));
    REQUIRE(outputs[1] == outputs[0] + 1.);
}

TEST_CASE("no middle")
{
    const auto [x, y] = make_vars("x", "y");

    const auto cf = cfunc<double>({x, y, par[0], 4_dbl}, {x, y}, kw::compact_mode = true);

    const std::vector inputs = {1., 2.};
    std::vector<double> outputs(4);

    REQUIRE(cf.get_dc().size() == 6u);

    cf(outputs, inputs, kw::pars = {3.});

    REQUIRE(outputs[0] == 1);
    REQUIRE(outputs[1] == 2);
    REQUIRE(outputs[2] == 3);
    REQUIRE(outputs[3] == 4);
}

TEST_CASE("no inputs")
{
    const auto cf = cfunc<double>({par[0], 4_dbl}, {}, kw::compact_mode = true);

    const std::vector<double> inputs = {};
    std::vector<double> outputs(2);

    REQUIRE(cf.get_dc().size() == 2u);

    cf(outputs, inputs, kw::pars = {3.});

    REQUIRE(outputs[0] == 3);
    REQUIRE(outputs[1] == 4);
}

TEST_CASE("func paronly")
{
    const auto x = par[0];
    const auto y = par[1];

    const auto ex1 = x * y + atan2(x + y, x * y);
    const auto ex2 = ex1 + 1.;

    const auto cf = cfunc<double>({ex1, ex2}, {}, kw::compact_mode = true);

    const std::vector<double> inputs = {};
    std::vector<double> outputs(2);

    REQUIRE(cf.get_dc().size() == 5u);

    cf(outputs, inputs, kw::pars = {1., 2.});

    REQUIRE(outputs[0] == approximately(2 + std::atan2(3., 2.)));
    REQUIRE(outputs[1] == outputs[0] + 1.);
}

TEST_CASE("inline limit")
{
    const auto x = make_vars("x");

    auto ex = sin(x);
    for (auto i = 0; i < 500; ++i) {
        ex = sin(ex);
    }

    const auto cf = cfunc<double>({ex}, {x}, kw::compact_mode = true);

    // NOTE: here the inlining limit must result in more than 3 expressions in the decomposition.
    REQUIRE(cf.get_dc().size() > 3u);

    const std::vector inputs = {1.};
    std::vector<double> outputs(1);
    cf(outputs, inputs);

    auto cmp = std::sin(inputs[0]);
    for (auto i = 0; i < 500; ++i) {
        cmp = std::sin(cmp);
    }

    REQUIRE(outputs[0] == approximately(cmp));
}
