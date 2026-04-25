// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/model/lagrange_prop.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Earth's gravitational parameter in km^3/s^2. Used by the test cases below; the propagator itself takes mu as a free
// variable, but we need to pick a value consistent with the (Earth-centric) LEO/GEO test cases.
constexpr double earth_mu_km3_s2 = 3.986004415e14 * 1e-9;

// A small set of synthetic Keplerian states covering different geometries (LEO, GEO, off-axis). All values are in km
// and km/s.
struct kep_test_case {
    const char *name;
    std::array<double, 3> ri;
    std::array<double, 3> vi;
};

const std::array kep_test_cases = {
    kep_test_case{"leo_perpendicular", {7000.0, 0.0, 0.0}, {0.0, 7.5, 0.0}},
    kep_test_case{"head_on", {7100.0, 0.0, 0.0}, {0.0, 7.45, 0.0}},
    kep_test_case{"vrel_along_x", {7000.0, 100.0, 50.0}, {3.5, 6.5, 1.0}},
    kep_test_case{"geo_like", {42164.0, 0.0, 0.0}, {0.0, 3.075, 0.0}},
};

// Propagation times to sweep over.
constexpr std::array<double, 4> tm_grid = {60.0, 600.0, 3600.0, 21600.0};

// Specific orbital energy: eps = v^2/2 - mu/r. Conserved exactly along a Keplerian orbit.
double orbital_energy(const std::array<double, 6> &state, double mu)
{
    const auto v2 = state[3] * state[3] + state[4] * state[4] + state[5] * state[5];
    const auto r = std::sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    return 0.5 * v2 - mu / r;
}

// Specific angular momentum vector: r x v. Each component is conserved exactly along a Keplerian orbit.
std::array<double, 3> angular_momentum(const std::array<double, 6> &state)
{
    return {state[1] * state[5] - state[2] * state[4], state[2] * state[3] - state[0] * state[5],
            state[0] * state[4] - state[1] * state[3]};
}

// Build the Lagrange propagator as a cfunc with input ordering (x0, y0, z0, vx0, vy0, vz0, mu, tm) and output ordering
// (x, y, z, vx, vy, vz), matching the signature exposed by mizuba's lagrange_prop.
cfunc<double> make_lagrange_cfunc()
{
    const auto pos0 = make_vars("x0", "y0", "z0");
    const auto vel0 = make_vars("vx0", "vy0", "vz0");
    const auto mu = expression{"mu"};
    const auto tm = expression{"tm"};

    const auto [pos, vel] = model::lagrange_prop({pos0[0], pos0[1], pos0[2]}, {vel0[0], vel0[1], vel0[2]}, mu, tm);

    std::vector vars{pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2], mu, tm};
    std::vector outputs{pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]};

    return cfunc<double>{std::move(outputs), std::move(vars)};
}

// Self-consistency test: propagating a state forward by +tm and then backward by -tm must return the original state to
// within machine epsilon.
TEST_CASE("round_trip")
{
    const auto cf = make_lagrange_cfunc();

    std::array<double, 8> inputs{};
    std::array<double, 6> state_fwd{};
    std::array<double, 6> state_round_trip{};

    for (const auto &tc : kep_test_cases) {
        for (const auto tm_s : tm_grid) {
            // Forward propagation by +tm.
            inputs = {tc.ri[0], tc.ri[1], tc.ri[2], tc.vi[0], tc.vi[1], tc.vi[2], earth_mu_km3_s2, tm_s};
            cf(state_fwd, inputs);

            // Backward propagation by -tm starting from the forward result.
            inputs = {state_fwd[0], state_fwd[1], state_fwd[2],    state_fwd[3],
                      state_fwd[4], state_fwd[5], earth_mu_km3_s2, -tm_s};
            cf(state_round_trip, inputs);

            // Compare position and velocity separately using norm-based relative error: for test cases where some
            // components of the initial state are zero, componentwise relative errors would be ill-defined.
            const auto pos_diff2 = (state_round_trip[0] - tc.ri[0]) * (state_round_trip[0] - tc.ri[0])
                                   + (state_round_trip[1] - tc.ri[1]) * (state_round_trip[1] - tc.ri[1])
                                   + (state_round_trip[2] - tc.ri[2]) * (state_round_trip[2] - tc.ri[2]);
            const auto pos_norm2 = tc.ri[0] * tc.ri[0] + tc.ri[1] * tc.ri[1] + tc.ri[2] * tc.ri[2];
            const auto pos_rel_err = std::sqrt(pos_diff2 / pos_norm2);

            const auto vel_diff2 = (state_round_trip[3] - tc.vi[0]) * (state_round_trip[3] - tc.vi[0])
                                   + (state_round_trip[4] - tc.vi[1]) * (state_round_trip[4] - tc.vi[1])
                                   + (state_round_trip[5] - tc.vi[2]) * (state_round_trip[5] - tc.vi[2]);
            const auto vel_norm2 = tc.vi[0] * tc.vi[0] + tc.vi[1] * tc.vi[1] + tc.vi[2] * tc.vi[2];
            const auto vel_rel_err = std::sqrt(vel_diff2 / vel_norm2);

            REQUIRE(pos_rel_err < 5e-14);
            REQUIRE(vel_rel_err < 5e-14);
        }
    }
}

// Verify that the Lagrange propagator conserves the two fundamental Keplerian invariants: specific orbital energy
// (eps = v^2/2 - mu/r) and the specific angular momentum vector (r x v).
TEST_CASE("invariants_preserved")
{
    const auto cf = make_lagrange_cfunc();

    std::array<double, 8> inputs{};
    std::array<double, 6> state_fwd{};

    for (const auto &tc : kep_test_cases) {
        const std::array<double, 6> state0 = {tc.ri[0], tc.ri[1], tc.ri[2], tc.vi[0], tc.vi[1], tc.vi[2]};

        for (const auto tm_s : tm_grid) {
            inputs = {state0[0], state0[1], state0[2], state0[3], state0[4], state0[5], earth_mu_km3_s2, tm_s};
            cf(state_fwd, inputs);

            // Specific orbital energy: scalar invariant.
            const auto eps0 = orbital_energy(state0, earth_mu_km3_s2);
            const auto eps1 = orbital_energy(state_fwd, earth_mu_km3_s2);

            REQUIRE(eps1 == approximately(eps0));

            // Specific angular momentum: vector invariant. Compare via norm of the difference, normalised by the norm
            // of the original (same approach as the round-trip test).
            const auto h0 = angular_momentum(state0);
            const auto h1 = angular_momentum(state_fwd);
            const auto h_diff2 = (h1[0] - h0[0]) * (h1[0] - h0[0]) + (h1[1] - h0[1]) * (h1[1] - h0[1])
                                 + (h1[2] - h0[2]) * (h1[2] - h0[2]);
            const auto h_norm2 = h0[0] * h0[0] + h0[1] * h0[1] + h0[2] * h0[2];
            const auto h_rel_err = std::sqrt(h_diff2 / h_norm2);

            REQUIRE(h_rel_err < 5e-14);
        }
    }
}

// Cross-check the Lagrange propagator against a numerical Taylor integration of the two-body problem.
TEST_CASE("vs_taylor_integrator")
{
    const auto cf = make_lagrange_cfunc();

    // Build the two-body ODE: dr/dt = v, dv/dt = -mu * r / |r|^3, with mu hardcoded to the Earth value.
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");
    const auto r3 = pow(sqrt((x * x) + (y * y) + (z * z)), 3.);
    const auto dyn = std::vector{prime(x) = vx,
                                 prime(y) = vy,
                                 prime(z) = vz,
                                 prime(vx) = -earth_mu_km3_s2 * x / r3,
                                 prime(vy) = -earth_mu_km3_s2 * y / r3,
                                 prime(vz) = -earth_mu_km3_s2 * z / r3};

    std::array<double, 8> inputs{};
    std::array<double, 6> state_lag{};

    for (const auto &tc : kep_test_cases) {
        const std::vector<double> ic = {tc.ri[0], tc.ri[1], tc.ri[2], tc.vi[0], tc.vi[1], tc.vi[2]};

        for (const auto tm_s : tm_grid) {
            // Lagrange propagation.
            inputs = {tc.ri[0], tc.ri[1], tc.ri[2], tc.vi[0], tc.vi[1], tc.vi[2], earth_mu_km3_s2, tm_s};
            cf(state_lag, inputs);

            // Numerical reference: fresh Taylor integrator from the original IC, propagated to tm_s.
            taylor_adaptive<double> ta{dyn, ic};
            ta.propagate_until(tm_s);
            const auto &state_ta = ta.get_state();

            // Norm-based relative error on position and velocity vs the numerical reference.
            const auto pos_diff2 = (state_lag[0] - state_ta[0]) * (state_lag[0] - state_ta[0])
                                   + (state_lag[1] - state_ta[1]) * (state_lag[1] - state_ta[1])
                                   + (state_lag[2] - state_ta[2]) * (state_lag[2] - state_ta[2]);
            const auto pos_norm2 = state_ta[0] * state_ta[0] + state_ta[1] * state_ta[1] + state_ta[2] * state_ta[2];
            const auto pos_rel_err = std::sqrt(pos_diff2 / pos_norm2);

            const auto vel_diff2 = (state_lag[3] - state_ta[3]) * (state_lag[3] - state_ta[3])
                                   + (state_lag[4] - state_ta[4]) * (state_lag[4] - state_ta[4])
                                   + (state_lag[5] - state_ta[5]) * (state_lag[5] - state_ta[5]);
            const auto vel_norm2 = state_ta[3] * state_ta[3] + state_ta[4] * state_ta[4] + state_ta[5] * state_ta[5];
            const auto vel_rel_err = std::sqrt(vel_diff2 / vel_norm2);

            REQUIRE(pos_rel_err < 5e-14);
            REQUIRE(vel_rel_err < 5e-14);
        }
    }
}
