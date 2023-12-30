// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/pendulum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("pendulum")
{
    auto [x, v] = make_vars("x", "v");

    const auto L = 0.5 * v * v - (1. - cos(x));

    const auto sys1 = lagrangian(L, {x}, {v});
    const auto sys2 = model::pendulum();

    auto ta1 = taylor_adaptive{sys1, {0.1, 0.2}};
    auto ta2 = taylor_adaptive{sys2, {0.1, 0.2}};

    ta1.propagate_until(10.);
    ta2.propagate_until(10.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
}

// Horizontally-driven pendulum.
TEST_CASE("driven pendulum")
{
    auto [x, v] = make_vars("x", "v");

    auto M = par[0];
    auto b = par[1];
    auto a = par[2];
    auto om = par[3];
    auto g = par[4];

    const auto L = 0.5 * M * b * b * v * v + M * b * v * a * om * cos(x) * cos(om * heyoka::time)
                   + 0.5 * M * a * a * om * om * cos(om * heyoka::time) * cos(om * heyoka::time) + M * g * b * cos(x);

    const auto par_vals = {.1, .2, .3, .4, .5};

    const auto sys1 = lagrangian(L, {x}, {v});
    auto ta1 = taylor_adaptive{sys1, {0.1, 0.2}, kw::pars = par_vals};
    auto ta2
        = taylor_adaptive{{prime(x) = v, prime(v) = a * om * om / b * cos(x) * sin(om * heyoka::time) - g / b * sin(x)},
                          {0.1, 0.2},
                          kw::pars = par_vals};

    ta1.propagate_until(10.);
    ta2.propagate_until(10.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
}
