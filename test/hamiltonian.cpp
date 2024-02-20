// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/hamiltonian.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/cos.hpp>
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
    auto [x, v, p] = make_vars("x", "v", "p");

    auto m = 1.5;
    auto lval = 2.1;
    auto l = par[0];
    auto g = par[1];

    const auto H = p * p / (2. * m * l * l) + m * g * l * (1. - cos(x));

    const auto sys1 = hamiltonian(H, {x}, {p});
    const auto sys2 = model::pendulum(kw::length = l, kw::gconst = g);

    auto ics1 = {0.1, m * lval * lval * 0.2};
    auto ics2 = {0.1, 0.2};
    auto par_vals = {lval, 3.4};

    auto ta1 = taylor_adaptive{sys1, ics1, kw::pars = par_vals};
    auto ta2 = taylor_adaptive{sys2, ics2, kw::pars = par_vals};

    ta1.propagate_until(10.);
    ta2.propagate_until(10.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1] * m * lval * lval));
}

// Horizontally-driven pendulum - test time-dependent
// Hamiltonian.
TEST_CASE("driven pendulum")
{
    auto [x, v, p] = make_vars("x", "v", "p");

    auto M = par[0];
    auto b = par[1];
    auto a = par[2];
    auto om = par[3];
    auto g = par[4];

    auto Mval = .1;
    auto bval = .2;
    auto aval = .3;
    auto omval = .4;

    const auto L = 0.5 * M * b * b * v * v + M * b * v * a * om * cos(x) * cos(om * heyoka::time)
                   + 0.5 * M * a * a * om * om * cos(om * heyoka::time) * cos(om * heyoka::time) + M * g * b * cos(x);

    const auto par_vals = {Mval, bval, aval, omval, .5};

    const auto sys1 = lagrangian(L, {x}, {v});

    // v as a function of p.
    auto v_p = (p - M * b * a * om * cos(x) * cos(om * heyoka::time)) / (M * b * b);

    // Compute the Hamiltonian.
    auto H = v_p * p - subs(L, {{v, v_p}});

    const auto sys2 = hamiltonian(H, {x}, {p});

    auto ta1 = taylor_adaptive{sys1, {0.1, 0.2}, kw::pars = par_vals};
    auto ta2 = taylor_adaptive{
        sys2, {0.1, Mval * bval * bval * 0.2 + Mval * bval * aval * omval * std::cos(.1)}, kw::pars = par_vals};

    ta1.propagate_until(10.);
    ta2.propagate_until(10.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    const auto xfin = ta2.get_state()[0];
    const auto pfin = ta2.get_state()[1];
    REQUIRE(ta1.get_state()[1]
            == approximately((pfin - Mval * bval * aval * omval * std::cos(xfin) * std::cos(omval * 10.))
                             / (Mval * bval * bval)));
}

TEST_CASE("two body problem")
{
    auto [x0, y0, z0] = make_vars("x0", "y0", "z0");
    auto [x1, y1, z1] = make_vars("x1", "y1", "z1");

    auto [vx0, vy0, vz0] = make_vars("vx0", "vy0", "vz0");
    auto [vx1, vy1, vz1] = make_vars("vx1", "vy1", "vz1");

    auto [px0, py0, pz0] = make_vars("px0", "py0", "pz0");
    auto [px1, py1, pz1] = make_vars("px1", "py1", "pz1");

    const auto L = 0.5 * (vx0 * vx0 + vy0 * vy0 + vz0 * vz0 + vx1 * vx1 + vy1 * vy1 + vz1 * vz1)
                   + 1. / sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1));

    const auto sys1 = lagrangian(L, {x0, y0, z0, x1, y1, z1}, {vx0, vy0, vz0, vx1, vy1, vz1});
    auto ics = {-1., 0., 0., 1., 0., 0., 0., -.5, 0., 0., 0.5, 0.};

    auto H = px0 * px0 + py0 * py0 + pz0 * pz0 + px1 * px1 + py1 * py1 + pz1 * pz1
             - subs(L, {{vx0, px0}, {vy0, py0}, {vz0, pz0}, {vx1, px1}, {vy1, py1}, {vz1, pz1}});

    const auto sys2 = hamiltonian(H, {x0, y0, z0, x1, y1, z1}, {px0, py0, pz0, px1, py1, pz1});

    auto ta1 = taylor_adaptive{sys1, ics};
    auto ta2 = taylor_adaptive{sys2, ics};

    ta1.propagate_until(5.);
    ta2.propagate_until(5.);

    REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
    REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
    REQUIRE(ta1.get_state()[2] == approximately(ta2.get_state()[2]));
    REQUIRE(ta1.get_state()[3] == approximately(ta2.get_state()[3]));
    REQUIRE(ta1.get_state()[4] == approximately(ta2.get_state()[4]));
    REQUIRE(ta1.get_state()[5] == approximately(ta2.get_state()[5]));
    REQUIRE(ta1.get_state()[6] == approximately(ta2.get_state()[6]));
    REQUIRE(ta1.get_state()[7] == approximately(ta2.get_state()[7]));
    REQUIRE(ta1.get_state()[8] == approximately(ta2.get_state()[8]));
    REQUIRE(ta1.get_state()[9] == approximately(ta2.get_state()[9]));
    REQUIRE(ta1.get_state()[10] == approximately(ta2.get_state()[10]));
    REQUIRE(ta1.get_state()[11] == approximately(ta2.get_state()[11]));
}

TEST_CASE("error handling")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    REQUIRE_THROWS_MATCHES(
        hamiltonian(x, {x}, {}), std::invalid_argument,
        Message("The number of generalised coordinates (1) must be equal to the number of generalised momenta (0)"));
    REQUIRE_THROWS_MATCHES(hamiltonian(x, {}, {}), std::invalid_argument,
                           Message("Cannot define a Hamiltonian without state variables"));

    REQUIRE_THROWS_MATCHES(
        hamiltonian(x, {x + v}, {v}), std::invalid_argument,
        Message("The list of generalised coordinates contains the expression '(v + x)' which is not a variable"));
    REQUIRE_THROWS_MATCHES(
        hamiltonian(x, {"__x"_var}, {v}), std::invalid_argument,
        Message("The list of generalised coordinates contains a variable with the invalid name '__x': names "
                "starting with '__' are reserved for internal use"));

    REQUIRE_THROWS_MATCHES(
        hamiltonian(x, {v}, {x + v}), std::invalid_argument,
        Message("The list of generalised momenta contains the expression '(v + x)' which is not a variable"));
    REQUIRE_THROWS_MATCHES(
        hamiltonian(x, {v}, {"__x"_var}), std::invalid_argument,
        Message("The list of generalised momenta contains a variable with the invalid name '__x': names "
                "starting with '__' are reserved for internal use"));

    REQUIRE_THROWS_MATCHES(hamiltonian(x, {x, x}, {v, v}), std::invalid_argument,
                           Message("The list of generalised coordinates contains duplicates"));
    REQUIRE_THROWS_MATCHES(hamiltonian(x, {x, v}, {v, v}), std::invalid_argument,
                           Message("The list of generalised momenta contains duplicates"));

    REQUIRE_THROWS_MATCHES(hamiltonian(x, {x, v}, {x, v}), std::invalid_argument,
                           Message("The list of generalised coordinates contains the expression 'x' "
                                   "which also appears as a generalised momentum"));
    REQUIRE_THROWS_MATCHES(hamiltonian(x, {x, v}, {v, "v2"_var}), std::invalid_argument,
                           Message("The list of generalised coordinates contains the expression 'v' "
                                   "which also appears as a generalised momentum"));

    REQUIRE_THROWS_MATCHES(
        hamiltonian(x + "v2"_var, {x}, {v}), std::invalid_argument,
        Message("The Hamiltonian contains the variable 'v2' which is not a generalised position or momentum"));
}
