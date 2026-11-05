// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
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
// beginning of the timestep) than eps * |h|. Such roots cannot be separated at the working precision: they are reported
// as a single root, whose direction is determined from the evaluation of the time derivative of the event equation. The
// other roots of the same event equation must still be detected.
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
    // are reported as a single root at the midpoint of the interval [0, 2**-53) of the [0, 1) range of the timestep
    // (i.e., at x = 2**-55), where the derivative of the event equation is negative. The root x ~ 0.3 is detected as
    // usual. The event equation is even in x, thus the direction of the single root is positive in the backward
    // timestep.
    const auto eq0 = (x * x - std::ldexp(1., -120)) * (x * x - std::ldexp(1., -118)) * (x * x - 0.09);
    // NOTE: the roots are x = 0, x = +-2**-60 and x = +-sqrt(0.09). In the forward timestep, the root x = 2**-60 is
    // 2**-59 away from the root at the beginning of the timestep in the [0, 1) range of the timestep. The two roots are
    // reported as a single root at x = 2**-55, where the derivative of the event equation is negative. The root x ~ 0.3
    // is detected as usual. The event equation is odd in x, thus the direction of the single root is negative in the
    // backward timestep too.
    const auto eq1 = x * (x * x - std::ldexp(1., -120)) * (x * x - 0.09);

    // The time coordinate of the single root.
    const auto t_single = std::ldexp(1., -55);

    return std::vector<std::tuple<expression, prop_data, prop_data>>{
        {eq0,
         {.t_final = .5, .events = {{t_single, -1}, {.3, 1}}},
         {.t_final = -.5, .events = {{-t_single, 1}, {-.3, -1}}}},
        {eq1,
         {.t_final = .5, .events = {{t_single, -1}, {.3, 1}}},
         {.t_final = -.5, .events = {{-t_single, -1}, {-.3, 1}}}}};
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
                kw::nt_events = {
                    nt_event<double>(ev_eq,
                                     [&detected](auto &, double t, int d_sgn) { detected.emplace_back(t, d_sgn); }),
                }};

            const auto [oc, min_h, max_h, n_steps, c_out, cb] = ta.propagate_until(pd->t_final);
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(n_steps == 1u);

            check_events(detected, pd->events);
        }
    }
}

// The same event equations as terminal events. The propagation stops at the first expected event, which for both event
// equations is the single root reported for the close roots.
TEST_CASE("scalar terminal")
{
    auto x = make_vars("x");

    for (const auto &[ev_eq, fwd, bwd] : make_test_cases(x)) {
        for (const auto *pd : {&fwd, &bwd}) {
            // The direction passed to the callback of the terminal event.
            int cb_d_sgn = 0;

            // NOTE: the callback returns false, so that the terminal event stops the propagation.
            auto ta = taylor_adaptive<double>{{prime(x) = 1_dbl},
                                              {0.},
                                              kw::t_events = {t_event<double>(
                                                  ev_eq, kw::callback = [&cb_d_sgn](auto &, int d_sgn) {
                                                      cb_d_sgn = d_sgn;
                                                      return false;
                                                  })}};

            const auto [oc, min_h, max_h, n_steps, c_out, cb] = ta.propagate_until(pd->t_final);
            // NOTE: a stopping terminal event with index 0 is encoded as the outcome -1.
            REQUIRE(oc == taylor_outcome{-1});
            REQUIRE(n_steps == 1u);
            REQUIRE(std::abs(ta.get_time() - pd->events[0].first) <= 1e-14);
            REQUIRE(cb_d_sgn == pd->events[0].second);
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

// The batch counterpart of the "scalar terminal" test.
//
// NOTE: batch element 0 is propagated forward, batch element 1 backward.
TEST_CASE("batch terminal")
{
    auto x = make_vars("x");

    for (const auto &[ev_eq, fwd, bwd] : make_test_cases(x)) {
        // The directions passed to the callback of the terminal event, for each batch element.
        std::vector<int> cb_d_sgn(2);

        // NOTE: the callback returns false, so that the terminal event stops the propagation.
        auto ta = taylor_adaptive_batch<double>{
            {prime(x) = 1_dbl},
            {0., 0.},
            2,
            kw::t_events = {t_event_batch<double>(
                ev_eq, kw::callback = [&cb_d_sgn](auto &, int d_sgn, std::uint32_t idx) {
                    cb_d_sgn[idx] = d_sgn;
                    return false;
                })}};

        ta.propagate_until({fwd.t_final, bwd.t_final});
        for (const auto &res : ta.get_propagate_res()) {
            // NOTE: a stopping terminal event with index 0 is encoded as the outcome -1.
            REQUIRE(std::get<0>(res) == taylor_outcome{-1});
            REQUIRE(std::get<3>(res) == 1u);
        }

        for (std::uint32_t i = 0; i < 2u; ++i) {
            const auto &pd = (i == 0u) ? fwd : bwd;
            REQUIRE(std::abs(ta.get_time()[i] - pd.events[0].first) <= 1e-14);
            REQUIRE(cb_d_sgn[i] == pd.events[0].second);
        }
    }
}

// Event detection fails if the location of a root cannot be narrowed down to the working precision. The event equation
// (x - 3)**6 has a root of multiplicity 6 at x = 3. In a single timestep of size 10.1 starting from x = 0, its Taylor
// polynomial is exactly (t - 3)**6 (the polynomial of the "find_real_poly_roots wide undecided interval" test, padded
// with zero coefficients up to the order of the integrator), and event detection throws. The integrator must be left
// unchanged, and it must remain usable.
//
// NOTE: the product is written out explicitly so that the Taylor coefficients of the event equation are computed
// exactly. With the default tolerance, the Taylor coefficients of the event equation at the two highest orders are
// zero, so that the event equation does not limit the timestep.
TEST_CASE("scalar detection failure")
{
    using Catch::Matchers::StartsWith;

    auto x = make_vars("x");

    const auto xm3 = x - 3.;

    auto ta = taylor_adaptive<double>{{prime(x) = 1_dbl},
                                      {0.},
                                      kw::nt_events = {nt_event<double>(xm3 * xm3 * xm3 * xm3 * xm3 * xm3,
                                                                        [](auto &, double, int) {})}};

    // Record the user-visible data.
    const auto state = ta.get_state();
    const auto tc = ta.get_tc();
    const auto time = ta.get_time();
    const auto last_h = ta.get_last_h();

    REQUIRE_THROWS_AS(ta.step(10.1), std::runtime_error);
    REQUIRE_THROWS_WITH(ta.step(10.1),
                        StartsWith("find_real_poly_roots() was unable to isolate the real roots of a polynomial"));

    REQUIRE(ta.get_state() == state);
    REQUIRE(ta.get_tc() == tc);
    REQUIRE(ta.get_time() == time);
    REQUIRE(ta.get_last_h() == last_h);

    // There are no roots in the timestep [0, 1).
    REQUIRE(std::get<0>(ta.step(1.)) == taylor_outcome::time_limit);
}

// The batch counterpart of the "scalar detection failure" test. Event detection fails in batch element 0, while it
// would succeed in batch element 1 (whose timestep [0, 1) contains no roots). No batch element must be updated.
TEST_CASE("batch detection failure")
{
    using Catch::Matchers::StartsWith;

    auto x = make_vars("x");

    const auto xm3 = x - 3.;

    auto ta = taylor_adaptive_batch<double>{
        {prime(x) = 1_dbl},
        {0., 0.},
        2,
        kw::nt_events = {nt_event_batch<double>(xm3 * xm3 * xm3 * xm3 * xm3 * xm3,
                                                [](auto &, double, int, std::uint32_t) {})}};

    // Record the user-visible data.
    const auto state = ta.get_state();
    const auto tc = ta.get_tc();
    const auto time = ta.get_time();
    const auto last_h = ta.get_last_h();
    const auto step_res = ta.get_step_res();

    REQUIRE_THROWS_AS(ta.step({10.1, 1.}), std::runtime_error);
    REQUIRE_THROWS_WITH(ta.step({10.1, 1.}),
                        StartsWith("find_real_poly_roots() was unable to isolate the real roots of a polynomial"));

    REQUIRE(ta.get_state() == state);
    REQUIRE(ta.get_tc() == tc);
    REQUIRE(ta.get_time() == time);
    REQUIRE(ta.get_last_h() == last_h);
    REQUIRE(ta.get_step_res() == step_res);

    // There are no roots in the timestep [0, 1).
    ta.step({1., 1.});
    for (const auto &res : ta.get_step_res()) {
        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
    }
}

// The event equations of the "scalar" test with a direction filter. The direction of the single root reported for the
// close roots is determined from the evaluation of the time derivative of the event equation, and the root must be
// detected only if this direction matches the direction of the event.
//
// NOTE: there is no batch counterpart, as the direction filter is implemented in code shared by the scalar and batch
// integrators.
TEST_CASE("scalar direction filter")
{
    auto x = make_vars("x");

    for (const auto &[ev_eq, fwd, bwd] : make_test_cases(x)) {
        for (const auto *pd : {&fwd, &bwd}) {
            for (const auto dir : {event_direction::negative, event_direction::positive}) {
                std::vector<ev_t> detected;

                auto ta = taylor_adaptive<double>{
                    {prime(x) = 1_dbl},
                    {0.},
                    kw::nt_events = {nt_event<double>(
                        ev_eq, [&detected](auto &, double t, int d_sgn) { detected.emplace_back(t, d_sgn); },
                        kw::direction = dir)}};

                const auto [oc, min_h, max_h, n_steps, c_out, cb] = ta.propagate_until(pd->t_final);
                REQUIRE(oc == taylor_outcome::time_limit);
                REQUIRE(n_steps == 1u);

                // The expected events are those whose direction matches the direction of the event.
                std::vector<ev_t> expected;
                std::ranges::copy_if(pd->events, std::back_inserter(expected),
                                     [dir](const auto &ev) { return ev.second == static_cast<int>(dir); });

                check_events(detected, expected);
            }
        }
    }
}
