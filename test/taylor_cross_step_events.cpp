// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <mp++/real.hpp>

#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// A sign change across a step boundary caused by a callback: a ball bouncing on the floor at x = 0, with a terminal
// event flipping the velocity. The non-terminal event v = 0 has no root within the steps around the bounce, but its sign
// jumps from negative to positive at the step boundary, which must be reported as a crossing at the bounce time.
TEST_CASE("bounce")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto n_bounces = 0u;

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {1., -0.5},
        kw::t_events = {t_ev_t(
            x,
            kw::callback =
                [&n_bounces](taylor_adaptive<double> &t, int) {
                    ++n_bounces;
                    t.get_state_data()[1] = -t.get_state()[1];
                    return true;
                },
            kw::direction = event_direction::negative)},
        kw::nt_events = {nt_ev_t(
            v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) { v_records.emplace_back(tm, d_sgn); })}};

    // NOTE: x(t) = 1 - t/2 - t**2/2 reaches the floor at t = 1, where v = -1.5. After the bounce v = 1.5, and the next
    // (genuine) root of v is at the apex, t = 2.5. We stop before it.
    const auto oc = std::get<0>(ta.propagate_until(2.));

    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(n_bounces == 1u);
    REQUIRE(v_records.size() == 1u);
    REQUIRE(v_records[0].first == approximately(1.));
    REQUIRE(v_records[0].second == 1);
}

// Backward-in-time variant of the bounce test.
TEST_CASE("bounce backward")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto n_bounces = 0u;

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {1., 0.5},
        kw::t_events = {t_ev_t(
            x,
            kw::callback =
                [&n_bounces](taylor_adaptive<double> &t, int) {
                    ++n_bounces;
                    t.get_state_data()[1] = -t.get_state()[1];
                    return true;
                },
            kw::direction = event_direction::positive)},
        kw::nt_events = {nt_ev_t(
            v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) { v_records.emplace_back(tm, d_sgn); })}};

    // NOTE: x(t) = 1 + t/2 - t**2/2 reaches the floor at t = -1, where v = 1.5 (x increasing in time, hence the positive
    // direction of the terminal event). After the bounce v = -1.5: as a function of time, v jumps upwards across the
    // bounce, hence the expected direction +1. The next (genuine) root of v is at t = -2.5. We stop before it.
    const auto oc = std::get<0>(ta.propagate_until(-2.));

    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(n_bounces == 1u);
    REQUIRE(v_records.size() == 1u);
    REQUIRE(v_records[0].first == approximately(-1.));
    REQUIRE(v_records[0].second == 1);
}

// Direction filtering of cross-step events, for both terminal and non-terminal events.
TEST_CASE("direction filter")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal events.
    std::vector<std::pair<double, int>> pos_records, neg_records;

    auto n_bounces = 0u;

    // NOTE: same setup as in the bounce test. The jump of v at the bounce has direction +1: it must be filtered out by
    // the events with negative direction.
    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {1., -0.5},
        kw::t_events = {t_ev_t(
                            x,
                            kw::callback =
                                [&n_bounces](taylor_adaptive<double> &t, int) {
                                    ++n_bounces;
                                    t.get_state_data()[1] = -t.get_state()[1];
                                    return true;
                                },
                            kw::direction = event_direction::negative),
                        t_ev_t(v, kw::direction = event_direction::negative)},
        kw::nt_events
        = {nt_ev_t(
               v, [&pos_records](taylor_adaptive<double> &, double tm, int d_sgn) { pos_records.emplace_back(tm, d_sgn); },
               kw::direction = event_direction::positive),
           nt_ev_t(
               v, [&neg_records](taylor_adaptive<double> &, double tm, int d_sgn) { neg_records.emplace_back(tm, d_sgn); },
               kw::direction = event_direction::negative)}};

    const auto oc = std::get<0>(ta.propagate_until(2.));

    // NOTE: the terminal event on v must not stop the integration.
    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(n_bounces == 1u);
    REQUIRE(pos_records.size() == 1u);
    REQUIRE(pos_records[0].first == approximately(1.));
    REQUIRE(pos_records[0].second == 1);
    REQUIRE(neg_records.empty());
}

// Cooldown filtering of terminal cross-step events: a terminal event which is on cooldown must not be triggered by a
// sign jump across a step boundary.
TEST_CASE("terminal cooldown filter")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;

    // Helper to run a bounce setup with a terminal event on v with cooldown cd, forward or backward in time. It returns
    // the (time, direction) records of the terminal event on v.
    //
    // NOTE: the backward setup is the time reversal of the forward one: the event times flip sign, the directions are
    // unchanged.
    const auto run = [](double cd, bool backward) {
        const auto s = backward ? -1. : 1.;

        auto [x, v] = make_vars("x", "v");

        std::vector<std::pair<double, int>> v_records;

        auto n_bounces = 0u;

        auto ta = taylor_adaptive<double>{
            {prime(x) = v, prime(v) = -1_dbl},
            {1., s},
            kw::t_events = {t_ev_t(
                                x,
                                kw::callback =
                                    [&n_bounces](taylor_adaptive<double> &t, int) {
                                        ++n_bounces;
                                        t.get_state_data()[1] = -t.get_state()[1];
                                        return true;
                                    },
                                kw::direction = backward ? event_direction::positive : event_direction::negative),
                            t_ev_t(
                                v,
                                kw::callback =
                                    [&v_records](taylor_adaptive<double> &t, int d_sgn) {
                                        v_records.emplace_back(t.get_time(), d_sgn);
                                        return true;
                                    },
                                kw::cooldown = cd)}};

        // NOTE: forward: x(t) = 1 + t - t**2/2. v has a genuine root at the apex, t = 1 (direction -1). The floor is
        // reached at t = 1 + sqrt(3), where v = -sqrt(3) jumps to sqrt(3) (direction +1). The next apex is at
        // t = 1 + 2 * sqrt(3) > 4.
        const auto oc = std::get<0>(ta.propagate_until(4. * s));

        REQUIRE(oc == taylor_outcome::time_limit);
        REQUIRE(n_bounces == 1u);

        return v_records;
    };

    for (const auto backward : {false, true}) {
        const auto s = backward ? -1. : 1.;

        // Cooldown still active at the bounce: the jump is filtered out.
        auto v_records = run(10., backward);
        REQUIRE(v_records.size() == 1u);
        REQUIRE(v_records[0].first == approximately(s));
        REQUIRE(v_records[0].second == -1);

        // Cooldown expired at the bounce: the jump is reported.
        v_records = run(.1, backward);
        REQUIRE(v_records.size() == 2u);
        REQUIRE(v_records[0].first == approximately(s));
        REQUIRE(v_records[0].second == -1);
        REQUIRE(v_records[1].first == approximately(s * (1. + std::sqrt(3.))));
        REQUIRE(v_records[1].second == 1);
    }
}

// Null steps (user-supplied zero timestep) must leave the stored signs unchanged.
TEST_CASE("null step")
{
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {0., -1.},
        kw::nt_events = {nt_ev_t(
            v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) { v_records.emplace_back(tm, d_sgn); })}};

    // First step: the sign of v at the end of the step is negative.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(v_records.empty());

    // Jump of v through zero between steps.
    ta.get_state_data()[1] = 1.;

    // Null steps: they must neither reset the stored sign nor update it with the new (positive) sign of v.
    ta.step(0.);
    ta.step(-0.);
    REQUIRE(v_records.empty());

    // Next step: the jump is reported at the beginning of the step.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(v_records.size() == 1u);
    REQUIRE(v_records[0].first == .25);
    REQUIRE(v_records[0].second == 1);
}

// Null steps due to a terminal event triggering at the beginning of the step must also leave the stored signs
// unchanged. In particular, a non-terminal cross-step event discarded because it does not happen before the terminal
// event must be reported in the next step.
TEST_CASE("terminal event null step")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {1., -1.},
        kw::t_events = {t_ev_t(x)},
        kw::nt_events = {nt_ev_t(
            v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) { v_records.emplace_back(tm, d_sgn); })}};

    // First step: the signs of x and v at the end of the step are positive and negative respectively.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(v_records.empty());

    // Set x to exactly zero and make v jump through zero.
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[1] = 1.;

    // Second step: the terminal event triggers at the beginning of the step, the cross-step event of v (at the same
    // time) is discarded.
    auto [oc, h] = ta.step(.25);
    REQUIRE(oc == taylor_outcome{-1});
    REQUIRE(h == 0.);
    REQUIRE(v_records.empty());

    // Third step: the terminal event is on cooldown, the jump of v is reported at the beginning of the step.
    std::tie(oc, h) = ta.step(.25);
    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(h == .25);
    REQUIRE(v_records.size() == 1u);
    REQUIRE(v_records[0].first == .25);
    REQUIRE(v_records[0].second == 1);
}

// After a reset of the stored signs (manual, or via the time setters), the next step boundary is not checked.
TEST_CASE("sign resets")
{
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    // The reset methods.
    //
    // NOTE: the time setters reset the stored signs even if the time does not change.
    const std::vector<std::function<void(taylor_adaptive<double> &)>> resets
        = {[](taylor_adaptive<double> &ta) { ta.reset_event_signs(); },
           [](taylor_adaptive<double> &ta) { ta.set_time(ta.get_time()); },
           [](taylor_adaptive<double> &ta) {
               const auto [hi, lo] = ta.get_dtime();
               ta.set_dtime(hi, lo);
           }};

    for (const auto &reset : resets) {
        auto [x, v] = make_vars("x", "v");

        // The (time, direction) records of the non-terminal event.
        std::vector<std::pair<double, int>> v_records;

        auto ta = taylor_adaptive<double>{
            {prime(x) = v, prime(v) = -1_dbl},
            {0., -1.},
            kw::nt_events = {nt_ev_t(v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) {
                v_records.emplace_back(tm, d_sgn);
            })}};

        // First step: the sign of v at the end of the step is negative.
        REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);

        // Jump of v through zero, followed by the reset.
        ta.get_state_data()[1] = 1.;
        reset(ta);

        // Next step: the jump is not reported.
        REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
        REQUIRE(v_records.empty());
    }
}

// An err_nf_state outcome resets the stored signs: the next step boundary is not checked.
TEST_CASE("err_nf_state reset")
{
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -1_dbl},
        {0., -1.},
        kw::nt_events = {nt_ev_t(
            v, [&v_records](taylor_adaptive<double> &, double tm, int d_sgn) { v_records.emplace_back(tm, d_sgn); })}};

    // First step: the sign of v at the end of the step is negative.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);

    // Second step: x overflows to -inf at the end of the step, while v stays negative.
    //
    // NOTE: the timestep is finite (and thus the time stays finite), as the deduced timestep is clamped to 0.25.
    ta.get_state_data()[0] = -std::numeric_limits<double>::max();
    ta.get_state_data()[1] = -1e308;
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::err_nf_state);

    // Restore a finite state, with a jump of v through zero.
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[1] = 1.;

    // Third step: the jump is not reported.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(v_records.empty());
}

// The stored signs are carried over by copies and serialisation.
TEST_CASE("copy and s11n")
{
    using ta_t = taylor_adaptive<double>;
    using t_ev_t = ta_t::t_event_t;

    // The copy methods.
    const std::vector<std::function<ta_t(const ta_t &)>> copies
        = {[](const ta_t &ta) { return ta_t(ta); },
           [](const ta_t &ta) {
               ta_t ret;
               ret = ta;
               return ret;
           },
           [](const ta_t &ta) {
               std::stringstream ss;
               {
                   boost::archive::binary_oarchive oa(ss);
                   oa << ta;
               }
               ta_t ret;
               {
                   boost::archive::binary_iarchive ia(ss);
                   ia >> ret;
               }
               return ret;
           }};

    for (const auto &copy : copies) {
        auto [x, v] = make_vars("x", "v");

        // NOTE: we use a terminal event without callback (so that it can be serialised), and we detect its trigger via
        // the outcome of the step.
        auto ta = ta_t{{prime(x) = v, prime(v) = -1_dbl}, {0., -1.}, kw::t_events = {t_ev_t(v)}};

        // First step: the sign of v at the end of the step is negative.
        REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);

        // Jump of v through zero.
        ta.get_state_data()[1] = 1.;

        // Copy, then step: the jump triggers the terminal event at the beginning of the step.
        auto ta2 = copy(ta);
        const auto [oc, h] = ta2.step(.25);
        REQUIRE(oc == taylor_outcome{-1});
        REQUIRE(h == 0.);
    }
}

// The automatic cooldown deduction must not produce a zero cooldown in case of underflow.
TEST_CASE("cooldown underflow")
{
    using ta_t = taylor_adaptive<float>;
    using t_ev_t = ta_t::t_event_t;

    auto x = make_vars("x");

    // NOTE: x(t) = x0 + t with x0 ~ -1e-44 (subnormal), and the event function is g = 1e38 * x, with g0 ~ -1e-6. The
    // infinity norm of the state vector (event function included) is |g0| < 1, thus g_eps = max(tol, eps * |g0|)
    // ~ 1e-13. At the root, |dg/dt| = 1e38, hence the cooldown heuristic g_eps / |dg/dt| * 10 underflows to zero.
    auto ta = ta_t{{prime(x) = expression{1.f}},
                   {-1e-44f},
                   kw::tol = 1e-20f,
                   kw::t_events = {t_ev_t(expression{1e38f} * x)}};

    const auto [oc, h] = ta.step(1.f);
    REQUIRE(oc == taylor_outcome{-1});
    REQUIRE(h > 0);

    // The deduced cooldown is the smallest positive float.
    const auto &cd = ta.get_te_cooldowns()[0];
    REQUIRE(cd);
    REQUIRE(cd->first == 0);
    REQUIRE(cd->second == std::numeric_limits<float>::denorm_min());

    // After the event x is exactly zero: the next step starts on the root of g. Thanks to the nonzero cooldown, the
    // event must not trigger again at t = 0.
    REQUIRE(ta.get_state()[0] == 0);
    const auto [oc2, h2] = ta.step(1.f);
    REQUIRE(oc2 == taylor_outcome::time_limit);
    REQUIRE(h2 == 1);
}

// Sign update via interval arithmetic: if the event function has a root within the step, the fex check fails and the
// sign at the end of the step is computed via interval Horner evaluation.
TEST_CASE("sign update ival")
{
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    for (const auto backward : {false, true}) {
        // NOTE: the backward setup is the time reversal of the forward one: the event times flip sign, the directions
        // are unchanged.
        const auto s = backward ? -1. : 1.;

        auto x = make_vars("x");

        // The (time, direction) records of the non-terminal event.
        std::vector<std::pair<double, int>> x_records;

        auto ta = taylor_adaptive<double>{
            {prime(x) = 1_dbl},
            {-.5 * s},
            kw::nt_events = {nt_ev_t(x, [&x_records](taylor_adaptive<double> &, double tm, int d_sgn) {
                x_records.emplace_back(tm, d_sgn);
            })}};

        // First step: x crosses zero at t = 0.5 * s and its value at the end of the step is 0.5 * s.
        REQUIRE(std::get<0>(ta.step(s)) == taylor_outcome::time_limit);
        REQUIRE(x_records.size() == 1u);
        REQUIRE(x_records[0].first == .5 * s);
        REQUIRE(x_records[0].second == 1);

        // Jump of x through zero between steps.
        ta.get_state_data()[0] = -s;

        // Second step: the jump is reported at the beginning of the step. x does not cross zero within the step.
        REQUIRE(std::get<0>(ta.step(.5 * s)) == taylor_outcome::time_limit);
        REQUIRE(x_records.size() == 2u);
        REQUIRE(x_records[1].first == s);
        REQUIRE(x_records[1].second == -1);
    }
}

// Sign update via exact arithmetic: if the event function is exactly zero at the end of the step, interval arithmetic
// cannot decide its sign, and the exact evaluation stores a zero sign.
TEST_CASE("sign update exact zero")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    for (const auto backward : {false, true}) {
        const auto s = backward ? -1. : 1.;

        // NOTE: the value of x after the jump at the end of the first step is 0.5 * j. Both jumps must be reported,
        // which can happen only if the stored sign is zero.
        for (const auto j : {1, -1}) {
            auto x = make_vars("x");

            // The (time, direction) records of the non-terminal event.
            std::vector<std::pair<double, int>> x_records;

            // NOTE: the terminal and non-terminal events have the same event function.
            auto ta = taylor_adaptive<double>{
                {prime(x) = 1_dbl},
                {-s},
                kw::t_events = {t_ev_t(x)},
                kw::nt_events = {nt_ev_t(x, [&x_records](taylor_adaptive<double> &, double tm, int d_sgn) {
                    x_records.emplace_back(tm, d_sgn);
                })}};

            // First step: the terminal event stops the integration exactly at the root of x, t = s. The root of the
            // non-terminal event is discarded, as it does not happen before the terminal event. The value of the
            // non-terminal event function at the end of the step is exactly zero.
            const auto [oc, h] = ta.step(2. * s);
            REQUIRE(oc == taylor_outcome{-1});
            REQUIRE(h == s);
            REQUIRE(x_records.empty());
            REQUIRE(ta.get_state()[0] == 0);

            // Jump of x away from zero.
            ta.get_state_data()[0] = .5 * j;

            // Second step: the jump of the non-terminal event is reported at the beginning of the step (the terminal
            // event is on cooldown). x does not cross zero within the step.
            REQUIRE(std::get<0>(ta.step(.25 * s)) == taylor_outcome::time_limit);
            REQUIRE(x_records.size() == 1u);
            REQUIRE(x_records[0].first == s);
            REQUIRE(x_records[0].second == (backward ? -j : j));
        }
    }
}

// Sign update via exact arithmetic: if the event function is nonzero at the end of the step, but too close to zero for
// interval arithmetic to decide its sign, the exact evaluation stores the correct sign.
TEST_CASE("sign update exact nonzero")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    const auto eps = std::numeric_limits<double>::epsilon();

    for (const auto backward : {false, true}) {
        // NOTE: the backward setup is the time reversal of the forward one: the event times flip sign, the directions
        // are unchanged.
        const auto s = backward ? -1. : 1.;

        // NOTE: the sign stored for the non-terminal event at the end of the first step must be the sign of s. Without
        // a jump, no event must be reported at the beginning of the second step. With a jump of x to -0.5 * s, an event
        // must be reported. Both behaviours can happen only if the stored sign is the sign of s.
        for (const auto jump : {false, true}) {
            auto x = make_vars("x");

            // The (time, direction) records of the non-terminal event.
            std::vector<std::pair<double, int>> x_records;

            // NOTE: the root of the terminal event function, x - s * eps, is at t = s * (1 + eps), right after the root
            // of x at t = s.
            auto ta = taylor_adaptive<double>{
                {prime(x) = 1_dbl},
                {-s},
                kw::t_events = {t_ev_t(x - s * eps)},
                kw::nt_events = {nt_ev_t(x, [&x_records](taylor_adaptive<double> &, double tm, int d_sgn) {
                    x_records.emplace_back(tm, d_sgn);
                })}};

            // First step: the non-terminal event triggers at t = s, then the terminal event stops the integration at
            // t = s * (1 + eps). There, the value of x is s * eps: interval arithmetic cannot decide its sign, the
            // exact evaluation can.
            const auto [oc, h] = ta.step(2. * s);
            REQUIRE(oc == taylor_outcome{-1});
            REQUIRE(h == s * (1 + eps));
            REQUIRE(x_records.size() == 1u);
            REQUIRE(x_records[0].first == s);
            REQUIRE(x_records[0].second == 1);
            REQUIRE(ta.get_state()[0] == s * eps);

            if (jump) {
                // Jump of x through zero.
                ta.get_state_data()[0] = -.5 * s;
            }

            // Second step: the terminal event is on cooldown, x does not cross zero within the step.
            REQUIRE(std::get<0>(ta.step(.25 * s)) == taylor_outcome::time_limit);

            if (jump) {
                // The jump is reported at the beginning of the step.
                REQUIRE(x_records.size() == 2u);
                REQUIRE(x_records[1].first == s * (1 + eps));
                REQUIRE(x_records[1].second == -1);
            } else {
                REQUIRE(x_records.size() == 1u);
            }
        }
    }
}

// Sign update in case of overflow: if the interval evaluation of the event function at the end of the step overflows,
// the exact evaluation stores the correct sign.
TEST_CASE("sign update overflow")
{
    using nt_ev_t = taylor_adaptive<double>::nt_event_t;

    for (const auto backward : {false, true}) {
        // NOTE: the backward setup is the time reversal of the forward one: the event times flip sign, the directions
        // are unchanged.
        const auto s = backward ? -1. : 1.;

        // NOTE: the sign stored at the end of the first step must be the sign of s. Without a jump, no event must be
        // reported at the beginning of the second step. With a jump of x to -s, an event must be reported. Both
        // behaviours can happen only if the stored sign is the sign of s.
        for (const auto jump : {false, true}) {
            auto x = make_vars("x");

            // The (time, direction) records of the non-terminal event.
            std::vector<std::pair<double, int>> g_records;

            // NOTE: the event function is g = 1e308 * x, whose Taylor coefficients are -1e308 * s and 1e308.
            auto ta = taylor_adaptive<double>{
                {prime(x) = 1_dbl},
                {-s},
                kw::nt_events = {nt_ev_t(1e308 * x, [&g_records](taylor_adaptive<double> &, double tm, int d_sgn) {
                    g_records.emplace_back(tm, d_sgn);
                })}};

            // First step: g triggers at t = s. At the end of the step, g = 1e308 * s, but the interval evaluation
            // overflows (1e308 * 2 > DBL_MAX) and the sign is computed by the exact evaluation.
            REQUIRE(std::get<0>(ta.step(2. * s)) == taylor_outcome::time_limit);
            REQUIRE(g_records.size() == 1u);
            REQUIRE(g_records[0].first == s);
            REQUIRE(g_records[0].second == 1);
            REQUIRE(ta.get_state()[0] == s);

            if (jump) {
                // Jump of x (and thus of g) through zero.
                ta.get_state_data()[0] = -s;
            }

            // Second step: g does not cross zero within the step.
            REQUIRE(std::get<0>(ta.step(.25 * s)) == taylor_outcome::time_limit);

            if (jump) {
                // The jump is reported at the beginning of the step.
                REQUIRE(g_records.size() == 2u);
                REQUIRE(g_records[1].first == 2. * s);
                REQUIRE(g_records[1].second == -1);
            } else {
                REQUIRE(g_records.size() == 1u);
            }
        }
    }
}

// A reset of the stored signs from within an event callback must take effect. That is, the signs at the end of the step
// must be computed before the invocation of the callbacks.
TEST_CASE("sign reset from callback")
{
    using ta_t = taylor_adaptive<double>;
    using nt_ev_t = ta_t::nt_event_t;

    auto v = make_vars("v");

    // The (time, direction) records of the non-terminal event on v.
    std::vector<std::pair<double, int>> v_records;

    // The number of invocations of the resetting callback.
    auto n_resets = 0u;

    auto ta = ta_t{{prime(v) = -1_dbl},
                   {-1.},
                   kw::nt_events = {nt_ev_t(v,
                                            [&v_records](ta_t &, double tm, int d_sgn) {
                                                v_records.emplace_back(tm, d_sgn);
                                            }),
                                    nt_ev_t(v + 1.1, [&n_resets](ta_t &t, double, int) {
                                        ++n_resets;
                                        t.reset_event_signs();
                                    })}};

    // First step: v + 1.1 crosses zero at t ~ 0.1, and its callback resets the stored signs. v stays negative.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(n_resets == 1u);
    REQUIRE(v_records.empty());

    // Jump of v (and of v + 1.1) through zero.
    ta.get_state_data()[0] = 1.;

    // Second step: no jump is reported. Neither v nor v + 1.1 cross zero within the step.
    REQUIRE(std::get<0>(ta.step(.25)) == taylor_outcome::time_limit);
    REQUIRE(n_resets == 1u);
    REQUIRE(v_records.empty());
}

// The stored signs survive the end of a propagate_*() call: a jump between consecutive propagations is reported.
TEST_CASE("consecutive propagations")
{
    using ta_t = taylor_adaptive<double>;
    using nt_ev_t = ta_t::nt_event_t;

    auto v = make_vars("v");

    // The (time, direction) records of the non-terminal event.
    std::vector<std::pair<double, int>> v_records;

    auto ta = ta_t{{prime(v) = -1_dbl}, {-1.}, kw::nt_events = {nt_ev_t(v, [&v_records](ta_t &, double tm, int d_sgn) {
                                                     v_records.emplace_back(tm, d_sgn);
                                                 })}};

    // First propagation: at the end, v is negative.
    REQUIRE(std::get<0>(ta.propagate_until(.5)) == taylor_outcome::time_limit);
    REQUIRE(v_records.empty());

    // Jump of v through zero, then propagate_for(). v does not cross zero during the propagation.
    ta.get_state_data()[0] = 1.;
    REQUIRE(std::get<0>(ta.propagate_for(.25)) == taylor_outcome::time_limit);
    REQUIRE(v_records == std::vector<std::pair<double, int>>{{.5, 1}});

    // Jump of v through zero, then propagate_grid(). v does not cross zero during the propagation.
    ta.get_state_data()[0] = -1.;
    REQUIRE(std::get<0>(ta.propagate_grid({.75, 1.})) == taylor_outcome::time_limit);
    REQUIRE(v_records == std::vector<std::pair<double, int>>{{.5, 1}, {.75, -1}});
}

// Smoke test for all the fixed-precision floating-point types, based on the "sign update exact zero" test.
TEST_CASE("fp types")
{
    tuple_for_each(fixed_fp_types, [](auto fp_x) {
        using fp_t = decltype(fp_x);
        using ta_t = taylor_adaptive<fp_t>;
        using t_ev_t = typename ta_t::t_event_t;
        using nt_ev_t = typename ta_t::nt_event_t;

        auto x = make_vars("x");

        // The (time, direction) records of the non-terminal event.
        std::vector<std::pair<fp_t, int>> x_records;

        // NOTE: the terminal and non-terminal events have the same event function.
        auto ta = ta_t{{prime(x) = 1_dbl},
                       {fp_t(-1)},
                       kw::t_events = {t_ev_t(x)},
                       kw::nt_events = {nt_ev_t(x, [&x_records](ta_t &, fp_t tm, int d_sgn) {
                           x_records.emplace_back(tm, d_sgn);
                       })}};

        // First step: the terminal event stops the integration exactly at the root of x, t = 1. The value of the
        // non-terminal event function at the end of the step is exactly zero.
        const auto [oc, h] = ta.step(fp_t(2));
        REQUIRE(oc == taylor_outcome{-1});
        REQUIRE(h == 1);
        REQUIRE(x_records.empty());
        REQUIRE(ta.get_state()[0] == 0);

        // Jump of x away from zero.
        ta.get_state_data()[0] = fp_t(.5);

        // Second step: the jump is reported at the beginning of the step.
        REQUIRE(std::get<0>(ta.step(fp_t(.25))) == taylor_outcome::time_limit);
        REQUIRE(x_records.size() == 1u);
        REQUIRE(x_records[0].first == 1);
        REQUIRE(x_records[0].second == 1);
    });
}

// Smoke test for mppp::real, based on the "sign update ival" and "sign update exact zero" tests.
TEST_CASE("real")
{
    using fp_t = mppp::real;
    using ta_t = taylor_adaptive<fp_t>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;

    for (const auto prec : {30, 123}) {
        auto x = make_vars("x");

        // The (time, direction) records of the non-terminal event.
        std::vector<std::pair<fp_t, int>> x_records;

        const auto rec_cb = [&x_records](ta_t &, fp_t tm, int d_sgn) { x_records.emplace_back(std::move(tm), d_sgn); };

        // Sign update via interval arithmetic.
        {
            auto ta = ta_t{{prime(x) = 1_dbl}, {fp_t(-.5, prec)}, kw::nt_events = {nt_ev_t(x, rec_cb)}};

            // First step: x crosses zero at t = 0.5 and its value at the end of the step is 0.5.
            REQUIRE(std::get<0>(ta.step(fp_t(1, prec))) == taylor_outcome::time_limit);
            REQUIRE(x_records.size() == 1u);
            REQUIRE(x_records[0].first == .5);
            REQUIRE(x_records[0].first.get_prec() == prec);
            REQUIRE(x_records[0].second == 1);

            // Jump of x through zero.
            ta.get_state_data()[0] = fp_t(-1, prec);

            // Second step: the jump is reported at the beginning of the step.
            REQUIRE(std::get<0>(ta.step(fp_t(.5, prec))) == taylor_outcome::time_limit);
            REQUIRE(x_records.size() == 2u);
            REQUIRE(x_records[1].first == 1);
            REQUIRE(x_records[1].first.get_prec() == prec);
            REQUIRE(x_records[1].second == -1);
        }

        x_records.clear();

        // Sign update via exact arithmetic.
        {
            // NOTE: the terminal and non-terminal events have the same event function.
            auto ta = ta_t{
                {prime(x) = 1_dbl}, {fp_t(-1, prec)}, kw::t_events = {t_ev_t(x)}, kw::nt_events = {nt_ev_t(x, rec_cb)}};

            // First step: the terminal event stops the integration exactly at the root of x, t = 1. The value of the
            // non-terminal event function at the end of the step is exactly zero.
            const auto [oc, h] = ta.step(fp_t(2, prec));
            REQUIRE(oc == taylor_outcome{-1});
            REQUIRE(h == 1);
            REQUIRE(x_records.empty());
            REQUIRE(ta.get_state()[0] == 0);

            // Jump of x away from zero.
            ta.get_state_data()[0] = fp_t(.5, prec);

            // Second step: the jump is reported at the beginning of the step.
            REQUIRE(std::get<0>(ta.step(fp_t(.25, prec))) == taylor_outcome::time_limit);
            REQUIRE(x_records.size() == 1u);
            REQUIRE(x_records[0].first == 1);
            REQUIRE(x_records[0].first.get_prec() == prec);
            REQUIRE(x_records[0].second == 1);
        }
    }
}
