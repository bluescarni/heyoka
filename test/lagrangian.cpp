// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/model/pendulum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Minimal test.
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

// Horizontally-driven pendulum - test time-dependent
// Lagrangian.
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

// Damped harmonic oscillator - test for the dissipation function. See:
// http://www.physics.hmc.edu/~saeta/courses/p111/uploads/Y2013/lec131023-DSHO.pdf
TEST_CASE("damped oscillator")
{
    auto [x, v] = make_vars("x", "v");

    auto k = par[0];
    auto b = par[1];
    auto m = par[2];

    const auto L = 0.5 * m * v * v - 0.5 * k * x * x;
    const auto D = 0.5 * b * v * v;

    const auto par_vals = {.1, .2, .3};

    const auto sys1 = lagrangian(L, {x}, {v}, D);
    auto ta1 = taylor_adaptive{sys1, {0.1, 0.2}, kw::pars = par_vals};
    auto ta2 = taylor_adaptive{{prime(x) = v, prime(v) = (-k * x - b * v) / m}, {0.1, 0.2}, kw::pars = par_vals};

    ta1.propagate_until(10.);
    ta2.propagate_until(10.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
}

TEST_CASE("two body problem")
{
    auto [x0, y0, z0] = make_vars("x0", "y0", "z0");
    auto [x1, y1, z1] = make_vars("x1", "y1", "z1");

    auto [vx0, vy0, vz0] = make_vars("vx0", "vy0", "vz0");
    auto [vx1, vy1, vz1] = make_vars("vx1", "vy1", "vz1");

    const auto L = 0.5 * (vx0 * vx0 + vy0 * vy0 + vz0 * vz0 + vx1 * vx1 + vy1 * vy1 + vz1 * vz1)
                   + 1. / sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1));

    const auto sys1 = lagrangian(L, {x0, y0, z0, x1, y1, z1}, {vx0, vy0, vz0, vx1, vy1, vz1});
    auto ics1 = {-1., 0., 0., 1., 0., 0., 0., -.5, 0., 0., 0.5, 0.};
    auto ics2 = {-1., 0., 0., 0., -.5, 0., 1., 0., 0., 0., 0.5, 0.};

    auto ta1 = taylor_adaptive{sys1, ics1};
    auto ta2 = taylor_adaptive{model::nbody(2), ics2};

    ta1.propagate_until(5.);
    ta2.propagate_until(5.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
    REQUIRE(ta1.get_state()[2] == approximately(ta2.get_state()[2]));
    REQUIRE(ta1.get_state()[3] == approximately(ta2.get_state()[6]));
    REQUIRE(ta1.get_state()[4] == approximately(ta2.get_state()[7]));
    REQUIRE(ta1.get_state()[5] == approximately(ta2.get_state()[8]));
    REQUIRE(ta1.get_state()[6] == approximately(ta2.get_state()[3]));
    REQUIRE(ta1.get_state()[7] == approximately(ta2.get_state()[4]));
    REQUIRE(ta1.get_state()[8] == approximately(ta2.get_state()[5]));
    REQUIRE(ta1.get_state()[9] == approximately(ta2.get_state()[9]));
    REQUIRE(ta1.get_state()[10] == approximately(ta2.get_state()[10]));
    REQUIRE(ta1.get_state()[11] == approximately(ta2.get_state()[11]));
}

TEST_CASE("error handling")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {x}, {}), std::invalid_argument,
        Message("The number of generalised coordinates (1) must be equal to the number of generalised velocities (0)"));
    REQUIRE_THROWS_MATCHES(lagrangian(x, {}, {}), std::invalid_argument,
                           Message("Cannot define a Lagrangian without state variables"));

    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {x + v}, {v}), std::invalid_argument,
        Message("The list of generalised coordinates contains the expression '(v + x)' which is not a variable"));
    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {"__x"_var}, {v}), std::invalid_argument,
        Message("The list of generalised coordinates contains a variable with the invalid name '__x': names "
                "starting with '__' are reserved for internal use"));

    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {v}, {x + v}), std::invalid_argument,
        Message("The list of generalised velocities contains the expression '(v + x)' which is not a variable"));
    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {v}, {"__x"_var}), std::invalid_argument,
        Message("The list of generalised velocities contains a variable with the invalid name '__x': names "
                "starting with '__' are reserved for internal use"));

    REQUIRE_THROWS_MATCHES(lagrangian(x, {x, x}, {v, v}), std::invalid_argument,
                           Message("The list of generalised coordinates contains duplicates"));
    REQUIRE_THROWS_MATCHES(lagrangian(x, {x, v}, {v, v}), std::invalid_argument,
                           Message("The list of generalised velocities contains duplicates"));

    REQUIRE_THROWS_MATCHES(lagrangian(x, {x, v}, {x, v}), std::invalid_argument,
                           Message("The list of generalised coordinates contains the expression 'x' "
                                   "which also appears as a generalised velocity"));
    REQUIRE_THROWS_MATCHES(lagrangian(x, {x, v}, {v, "v2"_var}), std::invalid_argument,
                           Message("The list of generalised coordinates contains the expression 'v' "
                                   "which also appears as a generalised velocity"));

    REQUIRE_THROWS_MATCHES(
        lagrangian(x + "v2"_var, {x}, {v}), std::invalid_argument,
        Message("The Lagrangian contains the variable 'v2' which is not a generalised position or velocity"));

    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {x}, {v}, x), std::invalid_argument,
        Message("The dissipation function contains the variable 'x' which is not a generalised velocity"));
    REQUIRE_THROWS_MATCHES(
        lagrangian(x, {x}, {v}, "v2"_var), std::invalid_argument,
        Message("The dissipation function contains the variable 'v2' which is not a generalised velocity"));
}
