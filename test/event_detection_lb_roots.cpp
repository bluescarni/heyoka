// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

// Tests for event detection when a root of an event equation falls exactly on the lower bound of an interval in the
// real-root isolation algorithm, either at the beginning of a timestep or at a bisection midpoint. Each root must be
// detected exactly once, and the other roots in the same interval must be detected too.
//
// NOTE: in all tests, x' = 1 and max_delta_t = 0.5, so that every timestep has a size of 0.5 and x(t) = x(0) + t, all
// computed exactly. The event equations are products of factors (x - r) with dyadic roots r, and the initial value of x
// is chosen so that one of the roots falls exactly on the boundary between two timesteps. That root is excluded from
// the earlier timestep (whose range is half-open), and it falls at the beginning of the later one.

namespace
{

// A detected event: time coordinate and sign of the time derivative of the event equation.
using ev_t = std::pair<double, int>;

// The data of a propagation: initial value of x, final time and expected events.
struct prop_data {
    double x0;
    double t_final;
    std::vector<ev_t> events;
};

// The test cases: event equation and data for a forward and a backward propagation.
auto make_test_cases(const expression &x)
{
    // NOTE: the roots are x = 1 and x = 1.25.
    const auto eq0 = (x - 1.) * (x - 1.25);
    // NOTE: the roots are x = 1, x = 1.25 and x = 1.375.
    const auto eq1 = (x - 1.) * (x - 1.25) * (x - 1.375);

    return std::vector<std::tuple<expression, prop_data, prop_data>>{
        // Forward: the root x = 1 falls at the beginning of the timestep [1, 1.5), and the root x = 1.25 at its midpoint.
        //
        // Backward: the root x = 1.25 falls at the beginning of the timestep [-1, -1.5), and the root x = 1 at its
        // midpoint.
        {eq0, {0., 2., {{1., -1}, {1.25, 1}}}, {2.25, -2., {{-1., 1}, {-1.25, -1}}}},
        // Forward: the root x = 1 falls at the beginning of the timestep [1, 1.5), which contains the other roots. The
        // root x = 1.25 falls at the midpoint of the timestep.
        //
        // Backward: the root x = 1.375 falls at the beginning of the timestep [-1, -1.5), which contains the other
        // roots.
        {eq1, {0., 2., {{1., 1}, {1.25, -1}, {1.375, 1}}}, {2.375, -2., {{-1., 1}, {-1.125, -1}, {-1.375, 1}}}}};
}

// Helper to compare the detected events with the expected ones.
void check_events(const std::vector<ev_t> &detected, const std::vector<ev_t> &expected)
{
    REQUIRE(detected.size() == expected.size());

    for (std::size_t i = 0; i < detected.size(); ++i) {
        REQUIRE(std::abs(detected[i].first - expected[i].first) <= 1e-14);
        REQUIRE(detected[i].second == expected[i].second);
    }
}

} // namespace

TEST_CASE("scalar")
{
    auto x = make_vars("x");

    for (const auto &[ev_eq, fwd, bwd] : make_test_cases(x)) {
        for (const auto *pd : {&fwd, &bwd}) {
            std::vector<ev_t> detected;

            auto ta = taylor_adaptive<double>{
                {prime(x) = 1_dbl},
                {pd->x0},
                kw::nt_events = {nt_event<double>(
                    ev_eq, [&detected](auto &, double t, int d_sgn) { detected.emplace_back(t, d_sgn); })}};

            ta.propagate_until(pd->t_final, kw::max_delta_t = .5);

            check_events(detected, pd->events);
        }
    }
}

// NOTE: batch element 0 is propagated forward, batch element 1 backward.
TEST_CASE("batch")
{
    auto x = make_vars("x");

    for (const auto &[ev_eq, fwd, bwd] : make_test_cases(x)) {
        std::vector<std::vector<ev_t>> detected(2);

        auto ta = taylor_adaptive_batch<double>{
            {prime(x) = 1_dbl},
            {fwd.x0, bwd.x0},
            2,
            kw::nt_events = {nt_event_batch<double>(ev_eq, [&detected](auto &, double t, int d_sgn, std::uint32_t idx) {
                detected[idx].emplace_back(t, d_sgn);
            })}};

        ta.propagate_until({fwd.t_final, bwd.t_final}, kw::max_delta_t = {.5, .5});

        check_events(detected[0], fwd.events);
        check_events(detected[1], bwd.events);
    }
}
