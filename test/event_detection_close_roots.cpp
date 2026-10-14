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

// Tests for event detection when an event equation has roots which are closer to each other (or to a root at the
// beginning of the timestep) than eps * |h|. The real-root isolation algorithm stops bisecting intervals narrower than
// eps (in the [0, 1) range of the timestep) and drops them, so that such roots are not detected. The other roots of
// the same event equation must still be detected.
//
// NOTE: in all tests, x' = 1 and x(0) = 0, so that x(t) = t. The propagation consists of the single timestep [0, 0.5)
// (forward) or [0, -0.5) (backward). The event equations are even or odd in x, so that the roots in the backward
// timestep mirror those in the forward timestep. The close roots are powers of two, so that they are only slightly
// perturbed by the rounding of the Taylor coefficients of the event equations.

namespace
{

// A detected event: time coordinate and sign of the time derivative of the event equation.
using ev_t = std::pair<double, int>;

// The data of a propagation: final time and expected events.
struct prop_data {
    double t_final;
    std::vector<ev_t> events;
};

// The test cases: event equation and data for a forward and a backward propagation.
auto make_test_cases(const expression &x)
{
    // NOTE: the roots are x = +-2**-60, x = +-2**-59 and x = +-sqrt(0.09). In the forward timestep, the two smallest
    // positive roots are 2**-60 apart, i.e., 2**-59 in the [0, 1) range of the timestep, which is less than eps. They
    // are not detected, while the root x ~ 0.3 is.
    const auto eq0 = (x * x - std::ldexp(1., -120)) * (x * x - std::ldexp(1., -118)) * (x * x - 0.09);
    // NOTE: the roots are x = 0, x = +-2**-60 and x = +-sqrt(0.09). In the forward timestep, the root x = 2**-60 is
    // 2**-59 away from the root at the beginning of the timestep in the [0, 1) range of the timestep. The root at the
    // beginning of the timestep and the root x ~ 0.3 are detected, the root x = 2**-60 is not.
    const auto eq1 = x * (x * x - std::ldexp(1., -120)) * (x * x - 0.09);

    return std::vector<std::tuple<expression, prop_data, prop_data>>{
        {eq0, {.5, {{.3, 1}}}, {-.5, {{-.3, -1}}}}, {eq1, {.5, {{0., 1}, {.3, 1}}}, {-.5, {{0., 1}, {-.3, 1}}}}};
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
                {0.},
                kw::nt_events = {nt_event<double>(
                    ev_eq, [&detected](auto &, double t, int d_sgn) { detected.emplace_back(t, d_sgn); })}};

            const auto [oc, min_h, max_h, n_steps, c_out, cb] = ta.propagate_until(pd->t_final);
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(n_steps == 1u);

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
            {0., 0.},
            2,
            kw::nt_events = {nt_event_batch<double>(ev_eq, [&detected](auto &, double t, int d_sgn, std::uint32_t idx) {
                detected[idx].emplace_back(t, d_sgn);
            })}};

        ta.propagate_until({fwd.t_final, bwd.t_final});
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
            REQUIRE(std::get<3>(res) == 1u);
        }

        check_events(detected[0], fwd.events);
        check_events(detected[1], bwd.events);
    }
}
