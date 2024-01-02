// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <iterator>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include <boost/math/constants/constants.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
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

// The damped wheel model from the elastic tides Python example.
TEST_CASE("damped wheel")
{
    namespace stdv = std::views;

    // Number of spokes.
    const auto N = 4u;
    // Central mass m_0 and smaller masses m_i.
    const auto m0 = 10.;
    const auto masses = std::vector<double>(N, 0.05);
    // Length of the spokes (i.e., rest position of the springs).
    const auto a = 1.;
    // Elastic constant.
    auto k = par[0];

    // Generalised coordinates.
    auto [x0, y0, alpha] = make_vars("x0", "y0", "alpha");
    auto qs = std::vector{x0, y0, alpha};
    auto ljs = stdv::iota(0u, N) | stdv::transform([](auto i) { return expression{fmt::format("l{}", i + 1u)}; });
    std::ranges::copy(ljs, std::back_inserter(qs));

    // Generalised velocities.
    auto [vx0, vy0, valpha] = make_vars("vx0", "vy0", "valpha");
    auto qdots = std::vector{vx0, vy0, valpha};
    auto vljs = stdv::iota(0u, N) | stdv::transform([](auto i) { return expression{fmt::format("vl{}", i + 1u)}; });
    std::ranges::copy(vljs, std::back_inserter(qdots));

    // Cartesian positions.
    std::vector<expression> xjs;
    std::ranges::copy(stdv::iota(0u, N) | stdv::transform([&](auto i) {
                          return x0 + (a - ljs[i]) * cos(alpha + i * (2. / N * boost::math::constants::pi<double>()));
                      }),
                      std::back_inserter(xjs));
    std::vector<expression> yjs;
    std::ranges::copy(stdv::iota(0u, N) | stdv::transform([&](auto i) {
                          return y0 + (a - ljs[i]) * sin(alpha + i * (2. / N * boost::math::constants::pi<double>()));
                      }),
                      std::back_inserter(yjs));

    // Compute the cartesian velocities as functions of the generalised coordinates/velocities.
    auto v_jac_xjs = diff_tensors(xjs, kw::diff_args = qs).get_jacobian();
    auto v_jac_yjs = diff_tensors(yjs, kw::diff_args = qs).get_jacobian();
    auto jac_xjs = xt::adapt(v_jac_xjs, {static_cast<int>(N), static_cast<int>(qs.size())});
    auto jac_yjs = xt::adapt(v_jac_yjs, {static_cast<int>(N), static_cast<int>(qs.size())});

    std::vector<expression> vxjs, vyjs;
    for (auto i = 0u; i < N; ++i) {
        auto xrow = xt::view(jac_xjs, i, xt::all());
        vxjs.push_back(std::inner_product(std::begin(xrow), std::end(xrow), std::begin(qdots), 0_dbl));

        auto yrow = xt::view(jac_yjs, i, xt::all());
        vyjs.push_back(std::inner_product(std::begin(yrow), std::end(yrow), std::begin(qdots), 0_dbl));
    }

    // Kinetic energy.
    auto kin_mi = stdv::iota(0u, N)
                  | stdv::transform([&](auto i) { return masses[i] * (vxjs[i] * vxjs[i] + vyjs[i] * vyjs[i]); });
    auto T = 1 / 2. * m0 * (vx0 * vx0 + vy0 * vy0)
             + 1 / 2. * sum(std::vector<expression>{std::begin(kin_mi), std::end(kin_mi)});

    // Potential.
    auto V_terms = stdv::iota(0u, N) | stdv::transform([&](auto i) { return ljs[i] * ljs[i]; });
    auto V = 1 / 2. * k * sum(std::vector<expression>(V_terms.begin(), V_terms.end()));

    // The Lagrangian.
    auto L = T - V;

    // The dissipation function.
    auto fc = par[1];
    auto D_terms = stdv::iota(0u, N) | stdv::transform([&](auto i) { return 0.5 * fc * vljs[i] * vljs[i]; });
    auto D = sum(std::vector<expression>(D_terms.begin(), D_terms.end()));

    // Formulate the equations of motion.
    auto sys = lagrangian(L, qs, qdots, D);

    // Init the integrator.
    auto ics = {0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -0.4251330363293203,
                0.05814973378505772,
                0.4421418964987427,
                0.3821135280593254,
                0.4035319449465937,
                0.473483299702404,
                -0.32295977917610563};
    auto ta = taylor_adaptive(sys, ics, kw::compact_mode = true);

    // Set the values for the spring constant
    // and friction coefficient.
    ta.get_pars_data()[0] = 5.;
    ta.get_pars_data()[1] = .1;

    // Run the numerical integration.
    ta.propagate_until(3.);

    // Compare with the integration as formulated by sympy.
    REQUIRE(ta.get_state()[0] == approximately(-1.27403829953764));
    REQUIRE(ta.get_state()[1] == approximately(0.16376361801359088));
    REQUIRE(ta.get_state()[2] == approximately(1.323428337349071));
    REQUIRE(ta.get_state()[3] == approximately(-0.0038352625671087643));
    REQUIRE(ta.get_state()[4] == approximately(-0.0038785147615456833));
    REQUIRE(ta.get_state()[5] == approximately(-0.004319076923234549));
    REQUIRE(ta.get_state()[6] == approximately(-0.0003951318230739097));
    REQUIRE(ta.get_state()[7] == approximately(-0.4247224818094968));
    REQUIRE(ta.get_state()[8] == approximately(0.05460006135548661));
    REQUIRE(ta.get_state()[9] == approximately(0.4394061018791296));
    REQUIRE(ta.get_state()[10] == approximately(0.003381629943640583));
    REQUIRE(ta.get_state()[11] == approximately(0.006350901803721282));
    REQUIRE(ta.get_state()[12] == approximately(0.004490453710403928));
    REQUIRE(ta.get_state()[13] == approximately(-0.0013988052681859328));
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
