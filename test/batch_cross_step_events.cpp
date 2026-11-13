// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Two-lane version of the scalar bounce test: a ball bouncing on the floor at x = 0, with a terminal event flipping the
// velocity. The bounce happens only in lane 0, and the sign jump of the non-terminal event v = 0 at the step boundary
// must be reported only in that lane.
TEST_CASE("bounce")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (lane, time, direction) records of the non-terminal event.
    std::vector<std::tuple<std::uint32_t, double, int>> v_records;

    // The number of bounces in each lane.
    std::vector<unsigned> n_bounces(2);

    // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1].
    auto ta = ta_t{{prime(x) = v, prime(v) = -1_dbl},
                   {1., 10., -.5, -.5},
                   2,
                   kw::t_events = {t_ev_t(
                       x,
                       kw::callback =
                           [&n_bounces](ta_t &t, int, std::uint32_t batch_idx) {
                               ++n_bounces[batch_idx];
                               t.get_state_data()[2u + batch_idx] = -t.get_state()[2u + batch_idx];
                               return true;
                           },
                       kw::direction = event_direction::negative, kw::cross_step = true)},
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // NOTE: in lane 0, x(t) = 1 - t/2 - t**2/2 reaches the floor at t = 1, where v = -1.5. After the bounce v = 1.5,
    // and the next (genuine) root of v is at the apex, t = 2.5. In lane 1, x(t) = 10 - t/2 - t**2/2 stays positive up
    // to t = 2, and v has no roots. We stop at t = 2.
    ta.propagate_until(2.);

    for (const auto &res : ta.get_propagate_res()) {
        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
    }
    REQUIRE(n_bounces == std::vector<unsigned>{1, 0});
    REQUIRE(v_records.size() == 1u);
    REQUIRE(std::get<0>(v_records[0]) == 0u);
    REQUIRE(std::get<1>(v_records[0]) == approximately(1.));
    REQUIRE(std::get<2>(v_records[0]) == 1);
}

// Backward-in-time variant of the bounce test.
TEST_CASE("bounce backward")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;

    auto [x, v] = make_vars("x", "v");

    // The (lane, time, direction) records of the non-terminal event.
    std::vector<std::tuple<std::uint32_t, double, int>> v_records;

    // The number of bounces in each lane.
    std::vector<unsigned> n_bounces(2);

    // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1].
    auto ta = ta_t{{prime(x) = v, prime(v) = -1_dbl},
                   {1., 10., .5, .5},
                   2,
                   kw::t_events = {t_ev_t(
                       x,
                       kw::callback =
                           [&n_bounces](ta_t &t, int, std::uint32_t batch_idx) {
                               ++n_bounces[batch_idx];
                               t.get_state_data()[2u + batch_idx] = -t.get_state()[2u + batch_idx];
                               return true;
                           },
                       kw::direction = event_direction::positive, kw::cross_step = true)},
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // NOTE: in lane 0, x(t) = 1 + t/2 - t**2/2 reaches the floor at t = -1, where v = 1.5 (x increasing in time, hence
    // the positive direction of the terminal event). After the bounce v = -1.5: as a function of time, v jumps upwards
    // across the bounce, hence the expected direction +1. The next (genuine) root of v is at t = -2.5. In lane 1,
    // x(t) = 10 + t/2 - t**2/2 stays positive down to t = -2, and v has no roots. We stop at t = -2.
    ta.propagate_until(-2.);

    for (const auto &res : ta.get_propagate_res()) {
        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
    }
    REQUIRE(n_bounces == std::vector<unsigned>{1, 0});
    REQUIRE(v_records.size() == 1u);
    REQUIRE(std::get<0>(v_records[0]) == 0u);
    REQUIRE(std::get<1>(v_records[0]) == approximately(-1.));
    REQUIRE(std::get<2>(v_records[0]) == 1);
}

// Direction filtering of cross-step events, for both terminal and non-terminal events. The jumps through zero at the
// step boundary have opposite directions in the two lanes.
TEST_CASE("direction filter")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto v = make_vars("v");

    // Helper to take a first step, at the end of which v is negative in lane 0 and positive in lane 1, and then to make
    // v jump through zero in both lanes (upwards in lane 0, downwards in lane 1).
    const auto first_step_and_jump = [](ta_t &ta) {
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        ta.get_state_data()[0] = 1.;
        ta.get_state_data()[1] = -1.;
    };

    // Non-terminal events.
    {
        // The (lane, time, direction) records of the non-terminal events.
        rec_t pos_records, neg_records;

        auto ta = ta_t{{prime(v) = -1_dbl},
                       {-1., 2.},
                       2,
                       kw::nt_events = {nt_ev_t(
                                            v,
                                            [&pos_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                pos_records.emplace_back(batch_idx, tm, d_sgn);
                                            },
                                            kw::direction = event_direction::positive, kw::cross_step = true),
                                        nt_ev_t(
                                            v,
                                            [&neg_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                neg_records.emplace_back(batch_idx, tm, d_sgn);
                                            },
                                            kw::direction = event_direction::negative, kw::cross_step = true)}};

        first_step_and_jump(ta);

        // Second step: v does not cross zero within the step in either lane. Each jump is reported only by the event
        // with the matching direction.
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(pos_records == rec_t{{0u, .25, 1}});
        REQUIRE(neg_records == rec_t{{1u, .25, -1}});
    }

    // Terminal event.
    {
        auto ta = ta_t{{prime(v) = -1_dbl},
                       {-1., 2.},
                       2,
                       kw::t_events = {t_ev_t(v, kw::direction = event_direction::negative, kw::cross_step = true)}};

        first_step_and_jump(ta);

        // Second step: the terminal event triggers at the beginning of the step only in lane 1.
        ta.step(std::vector<double>{.25, .25});
        REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome::time_limit, .25});
        REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome{-1}, 0.});
    }
}

// Cooldown filtering of terminal cross-step events: a terminal event which is on cooldown must not be triggered by a
// sign jump across a step boundary. In a bounce setup, a terminal event on v with a cooldown of 1 triggers at the apex.
// In lane 0 the bounce happens more than 1 after the apex (cooldown expired), in lane 1 less than 1 after the apex
// (cooldown still active).
TEST_CASE("terminal cooldown filter")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;

    for (const auto backward : {false, true}) {
        // NOTE: the backward setup is the time reversal of the forward one: the event times flip sign, the directions
        // are unchanged.
        const auto s = backward ? -1. : 1.;

        auto [x, v] = make_vars("x", "v");

        // The (time, direction) records of the terminal event on v, for each lane.
        std::vector<std::vector<std::pair<double, int>>> v_records(2);

        // The number of bounces in each lane.
        std::vector<unsigned> n_bounces(2);

        // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1].
        auto ta
            = ta_t{{prime(x) = v, prime(v) = -1_dbl},
                   {1., .125, s, .5 * s},
                   2,
                   kw::t_events = {t_ev_t(
                                       x,
                                       kw::callback =
                                           [&n_bounces](ta_t &t, int, std::uint32_t batch_idx) {
                                               ++n_bounces[batch_idx];
                                               t.get_state_data()[2u + batch_idx] = -t.get_state()[2u + batch_idx];
                                               return true;
                                           },
                                       kw::direction = backward ? event_direction::positive : event_direction::negative,
                                       kw::cross_step = true),
                                   t_ev_t(
                                       v,
                                       kw::callback =
                                           [&v_records](ta_t &t, int d_sgn, std::uint32_t batch_idx) {
                                               v_records[batch_idx].emplace_back(t.get_time()[batch_idx], d_sgn);
                                               return true;
                                           },
                                       kw::cooldown = 1., kw::cross_step = true)}};

        // NOTE: forward, lane 0: x(t) = 1 + t - t**2/2. The apex is at t = 1, the floor is reached at t = 1 + sqrt(3),
        // the next apex is at t = 1 + 2 * sqrt(3) > 4. Lane 1: x(t) = 1/8 + t/2 - t**2/2. The apex is at t = 1/2, the
        // floor is reached at t = 1/2 + sqrt(1/2), the next apex is at t = 1/2 + 2 * sqrt(1/2) > 3/2.
        ta.propagate_until(std::vector<double>{4. * s, 1.5 * s});

        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }
        REQUIRE(n_bounces == std::vector<unsigned>{1, 1});

        // Lane 0: cooldown expired at the bounce, the jump is reported.
        REQUIRE(v_records[0].size() == 2u);
        REQUIRE(v_records[0][0].first == approximately(s));
        REQUIRE(v_records[0][0].second == -1);
        REQUIRE(v_records[0][1].first == approximately(s * (1. + std::sqrt(3.))));
        REQUIRE(v_records[0][1].second == 1);

        // Lane 1: cooldown still active at the bounce, the jump is filtered out.
        REQUIRE(v_records[1].size() == 1u);
        REQUIRE(v_records[1][0].first == approximately(.5 * s));
        REQUIRE(v_records[1][0].second == -1);
    }
}

// Null steps (user-supplied zero timestep) must leave the stored signs unchanged, also when they happen only in some
// lanes.
TEST_CASE("null step")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto v = make_vars("v");

    // The (lane, time, direction) records of the non-terminal event.
    rec_t v_records;

    auto ta = ta_t{{prime(v) = -1_dbl},
                   {-1., 2.},
                   2,
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // First step: at the end of the step, v is negative in lane 0 and positive in lane 1.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(v_records.empty());

    // Jump of v through zero in both lanes.
    ta.get_state_data()[0] = 1.;
    ta.get_state_data()[1] = -1.;

    // Null step in lane 0 only: the jump is reported in lane 1, while lane 0 keeps its stored sign. v does not cross
    // zero within the step in lane 1.
    ta.step(std::vector<double>{0., .25});
    REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome::time_limit, 0.});
    REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome::time_limit, .25});
    REQUIRE(v_records == rec_t{{1u, .25, -1}});

    // Null steps in both lanes, with both signs of zero.
    ta.step(std::vector<double>{0., -0.});
    ta.step(std::vector<double>{-0., 0.});
    REQUIRE(v_records == rec_t{{1u, .25, -1}});

    // Next step: the jump is reported in lane 0. Nothing new happens in lane 1.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(v_records == rec_t{{1u, .25, -1}, {0u, .25, 1}});
}

// Null steps due to a terminal event triggering at the beginning of the step must also leave the stored signs
// unchanged. In particular, a non-terminal cross-step event discarded because it does not happen before the terminal
// event must be reported in the next step. The terminal event triggers only in lane 0.
TEST_CASE("terminal event null step")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto [x, v] = make_vars("x", "v");

    // The (lane, time, direction) records of the non-terminal event.
    rec_t v_records;

    // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1].
    auto ta = ta_t{{prime(x) = v, prime(v) = -1_dbl},
                   {1., 2., -1., -1.},
                   2,
                   kw::t_events = {t_ev_t(x, kw::cross_step = true)},
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // First step: at the end of the step, x is positive and v is negative in both lanes.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(v_records.empty());

    // Make v jump through zero in both lanes. Set x to exactly zero in lane 0 only.
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[1] = .5;
    ta.get_state_data()[2] = 1.;
    ta.get_state_data()[3] = 1.;

    // Second step: in lane 0, the terminal event triggers at the beginning of the step and the cross-step event of v
    // (at the same time) is discarded. In lane 1, the jump of v is reported. Neither x nor v cross zero within the step
    // in lane 1.
    ta.step(std::vector<double>{.25, .25});
    REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome{-1}, 0.});
    REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome::time_limit, .25});
    REQUIRE(v_records == rec_t{{1u, .25, 1}});

    // Third step: in lane 0, the terminal event is on cooldown and the jump of v is reported at the beginning of the
    // step. Nothing new happens in lane 1.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(v_records == rec_t{{1u, .25, 1}, {0u, .25, 1}});
}

// After a reset of the stored signs (manual, or via the time setters), the next step boundary is not checked in the
// lanes which were reset.
TEST_CASE("sign resets")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    // The reset methods, together with the expected records after the reset.
    //
    // NOTE: the time setters reset the stored signs in all lanes, even if the time does not change.
    const std::vector<std::pair<std::function<void(ta_t &)>, rec_t>> resets
        = {{[](ta_t &ta) { ta.reset_event_signs(); }, rec_t{}},
           {[](ta_t &ta) { ta.reset_event_signs(0); }, rec_t{{1u, .25, -1}}},
           {[](ta_t &ta) { ta.reset_event_signs(1); }, rec_t{{0u, .25, 1}}},
           {[](ta_t &ta) { ta.set_time(ta.get_time()); }, rec_t{}},
           {[](ta_t &ta) { ta.set_time(ta.get_time()[0]); }, rec_t{}},
           {[](ta_t &ta) {
                const auto [hi, lo] = ta.get_dtime();
                ta.set_dtime(hi, lo);
            },
            rec_t{}},
           {[](ta_t &ta) { ta.set_dtime(ta.get_dtime().first[0], ta.get_dtime().second[0]); }, rec_t{}}};

    for (const auto &[reset, expected] : resets) {
        auto v = make_vars("v");

        // The (lane, time, direction) records of the non-terminal event.
        rec_t v_records;

        auto ta = ta_t{{prime(v) = -1_dbl},
                       {-1., 2.},
                       2,
                       kw::nt_events = {nt_ev_t(
                           v,
                           [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                               v_records.emplace_back(batch_idx, tm, d_sgn);
                           },
                           kw::cross_step = true)}};

        // First step: at the end of the step, v is negative in lane 0 and positive in lane 1.
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        // Jump of v through zero in both lanes, followed by the reset.
        ta.get_state_data()[0] = 1.;
        ta.get_state_data()[1] = -1.;
        reset(ta);

        // Next step: the jumps are reported only in the lanes which were not reset. v does not cross zero within the
        // step in either lane.
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(v_records == expected);
    }
}

// An err_nf_state outcome resets the stored signs in the lane in which it happens, and only in that lane.
TEST_CASE("err_nf_state reset")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto [x, v] = make_vars("x", "v");

    // The (lane, time, direction) records of the non-terminal event.
    rec_t v_records;

    // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1].
    auto ta = ta_t{{prime(x) = v, prime(v) = -1_dbl},
                   {0., 0., -1., -1.},
                   2,
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // First step: at the end of the step, v is negative in both lanes.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }

    // In lane 0, x will overflow to -inf at the end of the next step, while v stays negative. In lane 1, v jumps
    // through zero.
    ta.get_state_data()[0] = -std::numeric_limits<double>::max();
    ta.get_state_data()[2] = -1e308;
    ta.get_state_data()[3] = 1.;

    // Second step: err_nf_state in lane 0, the jump is reported in lane 1.
    //
    // NOTE: the timesteps are finite (and thus the times stay finite), as the deduced timesteps are clamped to 0.25.
    ta.step(std::vector<double>{.25, .25});
    REQUIRE(std::get<0>(ta.get_step_res()[0]) == taylor_outcome::err_nf_state);
    REQUIRE(std::get<0>(ta.get_step_res()[1]) == taylor_outcome::time_limit);
    REQUIRE(v_records == rec_t{{1u, .25, 1}});

    // Restore a finite state in lane 0, with a jump of v through zero. Make v jump through zero in lane 1 too.
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[2] = 1.;
    ta.get_state_data()[3] = -1.;

    // Third step: the jump is not reported in lane 0, whose signs were reset, while it is reported in lane 1. v does
    // not cross zero within the step in either lane.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(v_records == rec_t{{1u, .25, 1}, {1u, .5, -1}});
}

// The stored signs of all lanes are carried over by copies and serialisation.
TEST_CASE("copy and s11n")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;

    // The copy methods.
    const std::vector<std::function<ta_t(const ta_t &)>> copies = {[](const ta_t &ta) { return ta_t(ta); },
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
        auto v = make_vars("v");

        // NOTE: we use a terminal event without callback (so that it can be serialised), and we detect its trigger via
        // the outcome of the step.
        auto ta = ta_t{{prime(v) = -1_dbl}, {-1., 2.}, 2, kw::t_events = {t_ev_t(v, kw::cross_step = true)}};

        // First step: at the end of the step, v is negative in lane 0 and positive in lane 1.
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        // Jump of v through zero in both lanes.
        ta.get_state_data()[0] = 1.;
        ta.get_state_data()[1] = -1.;

        // Copy, then step: in both lanes, the jump triggers the terminal event at the beginning of the step.
        auto ta2 = copy(ta);
        ta2.step(std::vector<double>{.25, .25});
        for (const auto &res : ta2.get_step_res()) {
            REQUIRE(res == std::tuple{taylor_outcome{-1}, 0.});
        }
    }
}

// Sign update via interval arithmetic: if the event function has a root within the step, the fex check fails and the
// sign at the end of the step is computed via interval Horner evaluation. Lane 0 integrates forward in time, lane 1
// backward.
TEST_CASE("sign update ival")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto [x, v] = make_vars("x", "v");

    // The (lane, time, direction) records of the non-terminal event.
    rec_t x_records;

    // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1]. v is constant.
    auto ta = ta_t{{prime(x) = v, prime(v) = 0_dbl},
                   {-.5, -.5, 1., -1.},
                   2,
                   kw::nt_events = {nt_ev_t(
                       x,
                       [&x_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           x_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // First step: in lane 0, x(t) = -1/2 + t crosses zero at t = 1/2 and its value at the end of the step is 1/2. In
    // lane 1, x(t) = -1/2 - t crosses zero at t = -1/2 and its value at the end of the step is also 1/2.
    //
    // NOTE: the Taylor polynomials of x differ in the two lanes, and each one has a negative value at the end of the
    // step of the other lane. Thus, using the coefficients of the wrong lane in the sign update would produce the wrong
    // sign.
    ta.step(std::vector<double>{1., -1.});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(x_records == rec_t{{0u, .5, 1}, {1u, -.5, -1}});

    // Jump of x through zero in both lanes.
    ta.get_state_data()[0] = -1.;
    ta.get_state_data()[1] = -1.;

    // Second step: the jumps are reported at the beginning of the step. As functions of time, x jumps downwards in lane
    // 0 and upwards in lane 1. x does not cross zero within the step in either lane.
    ta.step(std::vector<double>{.5, -.5});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(x_records == rec_t{{0u, .5, 1}, {1u, -.5, -1}, {0u, 1., -1}, {1u, -1., 1}});
}

// Sign update via exact arithmetic: if the event function is exactly zero at the end of the step, interval arithmetic
// cannot decide its sign, and the exact evaluation stores a zero sign. Lane 0 integrates forward in time, lane 1
// backward.
TEST_CASE("sign update exact zero")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    // NOTE: the value of x in both lanes after the jump at the end of the first step is 0.5 * j. Both jumps must be
    // reported, which can happen only if the stored sign is zero.
    for (const auto j : {1, -1}) {
        auto [x, v] = make_vars("x", "v");

        // The (lane, time, direction) records of the non-terminal event.
        rec_t x_records;

        // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1]. v is constant. The terminal and
        // non-terminal events have the same event function.
        auto ta = ta_t{{prime(x) = v, prime(v) = 0_dbl},
                       {-1., -1., 1., -1.},
                       2,
                       kw::t_events = {t_ev_t(x, kw::cross_step = true)},
                       kw::nt_events = {nt_ev_t(
                           x,
                           [&x_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                               x_records.emplace_back(batch_idx, tm, d_sgn);
                           },
                           kw::cross_step = true)}};

        // First step: in lane 0, x(t) = -1 + t, in lane 1, x(t) = -1 - t. The terminal event stops the integration
        // exactly at the root of x, t = 1 in lane 0 and t = -1 in lane 1. The root of the non-terminal event is
        // discarded, as it does not happen before the terminal event. The value of the non-terminal event function at
        // the end of the step is exactly zero.
        //
        // NOTE: the Taylor polynomials of x differ in the two lanes, and each one has a negative value at the end of
        // the step of the other lane. Thus, using the coefficients of the wrong lane in the sign update would produce
        // the wrong sign.
        ta.step(std::vector<double>{2., -2.});
        REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome{-1}, 1.});
        REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome{-1}, -1.});
        REQUIRE(x_records.empty());
        REQUIRE(ta.get_state()[0] == 0);
        REQUIRE(ta.get_state()[1] == 0);

        // Jump of x away from zero in both lanes.
        ta.get_state_data()[0] = .5 * j;
        ta.get_state_data()[1] = .5 * j;

        // Second step: the jumps of the non-terminal event are reported at the beginning of the step (the terminal
        // event is on cooldown). The direction of the jump in lane 1 is flipped with respect to lane 0, as lane 1
        // integrates backward in time. x does not cross zero within the step in either lane.
        ta.step(std::vector<double>{.25, -.25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(x_records == rec_t{{0u, 1., j}, {1u, -1., -j}});
    }
}

// Sign update via exact arithmetic: if the event function is nonzero at the end of the step, but too close to zero for
// interval arithmetic to decide its sign, the exact evaluation stores the correct sign. Lane 0 integrates forward in
// time, lane 1 backward.
TEST_CASE("sign update exact nonzero")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    const auto eps = std::numeric_limits<double>::epsilon();

    // NOTE: the sign stored for the non-terminal event at the end of the first step must be positive in both lanes.
    // Without a jump, no event must be reported at the beginning of the second step. With a jump of x to -0.5, an event
    // must be reported. Both behaviours can happen only if the stored sign is positive.
    for (const auto jump : {false, true}) {
        auto [x, v] = make_vars("x", "v");

        // The (lane, time, direction) records of the non-terminal event.
        rec_t x_records;

        // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1]. v is constant. The root of the
        // terminal event function, x - eps, is at t = 1 + eps in lane 0 and at t = -(1 + eps) in lane 1, right after
        // the root of x.
        auto ta = ta_t{{prime(x) = v, prime(v) = 0_dbl},
                       {-1., -1., 1., -1.},
                       2,
                       kw::t_events = {t_ev_t(x - eps, kw::cross_step = true)},
                       kw::nt_events = {nt_ev_t(
                           x,
                           [&x_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                               x_records.emplace_back(batch_idx, tm, d_sgn);
                           },
                           kw::cross_step = true)}};

        // First step: in lane 0, x(t) = -1 + t, in lane 1, x(t) = -1 - t. The non-terminal event triggers at t = 1 in
        // lane 0 and at t = -1 in lane 1, then the terminal event stops the integration. There, the value of x is eps
        // in both lanes: interval arithmetic cannot decide its sign, the exact evaluation can.
        //
        // NOTE: the Taylor polynomials of x differ in the two lanes, and each one has a negative value at the end of
        // the step of the other lane. Thus, using the coefficients of the wrong lane in the sign update would produce
        // the wrong sign.
        ta.step(std::vector<double>{2., -2.});
        REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome{-1}, 1 + eps});
        REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome{-1}, -(1 + eps)});
        REQUIRE(x_records == rec_t{{0u, 1., 1}, {1u, -1., -1}});
        REQUIRE(ta.get_state()[0] == eps);
        REQUIRE(ta.get_state()[1] == eps);

        if (jump) {
            // Jump of x through zero in both lanes.
            ta.get_state_data()[0] = -.5;
            ta.get_state_data()[1] = -.5;
        }

        // Second step: the terminal event is on cooldown, x does not cross zero within the step in either lane.
        ta.step(std::vector<double>{.25, -.25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        if (jump) {
            // The jumps are reported at the beginning of the step. As functions of time, x jumps downwards in lane 0
            // and upwards in lane 1.
            REQUIRE(x_records == rec_t{{0u, 1., 1}, {1u, -1., -1}, {0u, 1 + eps, -1}, {1u, -(1 + eps), 1}});
        } else {
            REQUIRE(x_records == rec_t{{0u, 1., 1}, {1u, -1., -1}});
        }
    }
}

// Sign update in case of overflow: if the interval evaluation of the event function at the end of the step overflows,
// the exact evaluation stores the correct sign. Lane 0 integrates forward in time, lane 1 backward.
TEST_CASE("sign update overflow")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    // NOTE: the sign stored at the end of the first step must be positive in both lanes. Without a jump, no event must
    // be reported at the beginning of the second step. With a jump of x to -1, an event must be reported. Both
    // behaviours can happen only if the stored sign is positive.
    for (const auto jump : {false, true}) {
        auto [x, v] = make_vars("x", "v");

        // The (lane, time, direction) records of the non-terminal event.
        rec_t g_records;

        // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1]. v is constant. The event
        // function is g = 1e308 * x.
        auto ta = ta_t{{prime(x) = v, prime(v) = 0_dbl},
                       {-1., -1., 1., -1.},
                       2,
                       kw::nt_events = {nt_ev_t(
                           1e308 * x,
                           [&g_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                               g_records.emplace_back(batch_idx, tm, d_sgn);
                           },
                           kw::cross_step = true)}};

        // First step: in lane 0, x(t) = -1 + t, in lane 1, x(t) = -1 - t. g triggers at t = 1 in lane 0 and at t = -1
        // in lane 1. At the end of the step, g = 1e308 in both lanes, but the interval evaluation overflows
        // (1e308 * 2 > DBL_MAX) and the sign is computed by the exact evaluation.
        //
        // NOTE: the Taylor polynomials of g differ in the two lanes, and each one has a negative value at the end of
        // the step of the other lane. Thus, using the coefficients of the wrong lane in the sign update would produce
        // the wrong sign.
        ta.step(std::vector<double>{2., -2.});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(g_records == rec_t{{0u, 1., 1}, {1u, -1., -1}});
        REQUIRE(ta.get_state()[0] == 1);
        REQUIRE(ta.get_state()[1] == 1);

        if (jump) {
            // Jump of x (and thus of g) through zero in both lanes.
            ta.get_state_data()[0] = -1.;
            ta.get_state_data()[1] = -1.;
        }

        // Second step: g does not cross zero within the step in either lane.
        ta.step(std::vector<double>{.25, -.25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        if (jump) {
            // The jumps are reported at the beginning of the step. As functions of time, g jumps downwards in lane 0
            // and upwards in lane 1.
            REQUIRE(g_records == rec_t{{0u, 1., 1}, {1u, -1., -1}, {0u, 2., -1}, {1u, -2., 1}});
        } else {
            REQUIRE(g_records == rec_t{{0u, 1., 1}, {1u, -1., -1}});
        }
    }
}

// The automatic cooldown deduction must not produce a zero cooldown in case of underflow. The underflow happens only in
// lane 0.
TEST_CASE("cooldown underflow")
{
    using ta_t = taylor_adaptive_batch<float>;
    using t_ev_t = ta_t::t_event_t;

    auto x = make_vars("x");

    // NOTE: x(t) = x0 + t and the event function is g = 1e38 * x.
    //
    // In lane 0, x0 ~ -1e-44 (subnormal) and g0 ~ -1e-6. The infinity norm of the state vector (event function
    // included) is |g0| < 1, thus g_eps = max(tol, eps * |g0|) ~ 1e-13. At the root, |dg/dt| = 1e38, hence the cooldown
    // heuristic g_eps / |dg/dt| * 10 underflows to zero.
    //
    // In lane 1, x0 = -1/2 and g0 = -5e37. The infinity norm of the state vector is |g0| > 1, thus
    // g_eps = max(tol, eps) * |g0| ~ 6e30, and the cooldown heuristic produces ~6e-7.
    auto ta = ta_t{{prime(x) = expression{1.f}},
                   {-1e-44f, -.5f},
                   2,
                   kw::tol = 1e-20f,
                   kw::t_events = {t_ev_t(expression{1e38f} * x, kw::cross_step = true)}};

    ta.step(std::vector<float>{1.f, 1.f});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome{-1});
        REQUIRE(h > 0);
    }

    // In lane 0, the deduced cooldown is the smallest positive float. In lane 1, it is a normal float.
    const auto &cds = ta.get_te_cooldowns();
    REQUIRE(cds[0][0]);
    REQUIRE(cds[0][0]->first == 0);
    REQUIRE(cds[0][0]->second == std::numeric_limits<float>::denorm_min());
    REQUIRE(cds[1][0]);
    REQUIRE(cds[1][0]->first == 0);
    REQUIRE(cds[1][0]->second > std::numeric_limits<float>::min());

    // After the event x is exactly zero in both lanes: the next step starts on the root of g. Thanks to the nonzero
    // cooldowns, the event must not trigger again at t = 0.
    REQUIRE(ta.get_state()[0] == 0);
    REQUIRE(ta.get_state()[1] == 0);
    ta.step(std::vector<float>{1.f, 1.f});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
        REQUIRE(h == 1);
    }
}

// A reset of the stored signs from within an event callback must take effect. That is, the signs at the end of the step
// must be computed in all lanes before the invocation of the callbacks. The reset happens in a callback of lane 0, and
// it must take effect in lane 1 too.
TEST_CASE("sign reset from callback")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto v = make_vars("v");

    // The (lane, time, direction) records of the non-terminal event on v.
    rec_t v_records;

    // The number of invocations of the resetting callback.
    auto n_resets = 0u;

    auto ta = ta_t{{prime(v) = -1_dbl},
                   {-1., -2.},
                   2,
                   kw::nt_events = {nt_ev_t(
                                        v,
                                        [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                            v_records.emplace_back(batch_idx, tm, d_sgn);
                                        },
                                        kw::cross_step = true),
                                    nt_ev_t(
                                        v + 1.1,
                                        [&n_resets](ta_t &t, double, int, std::uint32_t) {
                                            ++n_resets;
                                            t.reset_event_signs();
                                        },
                                        kw::cross_step = true)}};

    // First step: v + 1.1 crosses zero at t ~ 0.1 only in lane 0, and its callback resets the stored signs in all
    // lanes. v stays negative in both lanes.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(n_resets == 1u);
    REQUIRE(v_records.empty());

    // Jump of v (and of v + 1.1) through zero in both lanes.
    ta.get_state_data()[0] = 1.;
    ta.get_state_data()[1] = 1.;

    // Second step: no jump is reported in either lane. Neither v nor v + 1.1 cross zero within the step.
    ta.step(std::vector<double>{.25, .25});
    for (const auto &[oc, h] : ta.get_step_res()) {
        REQUIRE(oc == taylor_outcome::time_limit);
    }
    REQUIRE(n_resets == 1u);
    REQUIRE(v_records.empty());
}

// The stored signs survive the end of a propagate_*() call: a jump between consecutive propagations is reported.
TEST_CASE("consecutive propagations")
{
    using ta_t = taylor_adaptive_batch<double>;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    auto v = make_vars("v");

    // The (lane, time, direction) records of the non-terminal event.
    rec_t v_records;

    auto ta = ta_t{{prime(v) = -1_dbl},
                   {-1., 2.},
                   2,
                   kw::nt_events = {nt_ev_t(
                       v,
                       [&v_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                           v_records.emplace_back(batch_idx, tm, d_sgn);
                       },
                       kw::cross_step = true)}};

    // Helper to check that all the outcomes of a propagation are time_limit.
    const auto check_res = [&ta]() {
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }
    };

    // First propagation: at the end, v is negative in lane 0 and positive in lane 1.
    ta.propagate_until(.5);
    check_res();
    REQUIRE(v_records.empty());

    // Jump of v through zero in both lanes (upwards in lane 0, downwards in lane 1), then propagate_for(). v does not
    // cross zero during the propagation in either lane.
    ta.get_state_data()[0] = 1.;
    ta.get_state_data()[1] = -1.;
    ta.propagate_for(.25);
    check_res();
    REQUIRE(v_records == rec_t{{0u, .5, 1}, {1u, .5, -1}});

    // Jump of v through zero in both lanes (downwards in lane 0, upwards in lane 1), then propagate_grid(). v does not
    // cross zero during the propagation in either lane.
    //
    // NOTE: the grid is stored point-major, i.e., [t0_0, t0_1, t1_0, t1_1].
    ta.get_state_data()[0] = -1.;
    ta.get_state_data()[1] = 1.;
    ta.propagate_grid({.75, .75, 1., 1.});
    check_res();
    REQUIRE(v_records == rec_t{{0u, .5, 1}, {1u, .5, -1}, {0u, .75, -1}, {1u, .75, 1}});
}

// Smoke test for all the fixed-precision floating-point types, based on the "sign update exact zero" test.
TEST_CASE("fp types")
{
    tuple_for_each(fixed_fp_types, [](auto fp_x) {
        using fp_t = decltype(fp_x);
        using ta_t = taylor_adaptive_batch<fp_t>;
        using t_ev_t = typename ta_t::t_event_t;
        using nt_ev_t = typename ta_t::nt_event_t;
        using rec_t = std::vector<std::tuple<std::uint32_t, fp_t, int>>;

        auto [x, v] = make_vars("x", "v");

        // The (lane, time, direction) records of the non-terminal event.
        rec_t x_records;

        // NOTE: the state vector is stored variable-major, i.e., [x_0, x_1, v_0, v_1]. v is constant. The terminal and
        // non-terminal events have the same event function.
        auto ta = ta_t{{prime(x) = v, prime(v) = 0_dbl},
                       {fp_t(-1), fp_t(-1), fp_t(1), fp_t(-1)},
                       2,
                       kw::t_events = {t_ev_t(x, kw::cross_step = true)},
                       kw::nt_events = {nt_ev_t(
                           x,
                           [&x_records](ta_t &, fp_t tm, int d_sgn, std::uint32_t batch_idx) {
                               x_records.emplace_back(batch_idx, tm, d_sgn);
                           },
                           kw::cross_step = true)}};

        // First step: in lane 0, x(t) = -1 + t, in lane 1, x(t) = -1 - t. The terminal event stops the integration
        // exactly at the root of x, t = 1 in lane 0 and t = -1 in lane 1. The value of the non-terminal event function
        // at the end of the step is exactly zero.
        ta.step(std::vector<fp_t>{fp_t(2), fp_t(-2)});
        REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome{-1}, fp_t(1)});
        REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome{-1}, fp_t(-1)});
        REQUIRE(x_records.empty());

        // Jump of x away from zero in both lanes.
        ta.get_state_data()[0] = fp_t(.5);
        ta.get_state_data()[1] = fp_t(.5);

        // Second step: the jumps are reported at the beginning of the step.
        ta.step(std::vector<fp_t>{fp_t(.25), fp_t(-.25)});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(x_records == rec_t{{0u, fp_t(1), 1}, {1u, fp_t(-1), -1}});
    });
}

// Event functions whose Taylor polynomial is the zero polynomial. In-step detection reports the zero polynomial as a
// single root at the beginning of the step with zero direction, cross-step detection skips it (its constant coefficient
// is zero) and stores a zero sign at the end of the step.
TEST_CASE("zero polynomial")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    // NOTE: in all the tests below x' = 0, so that the Taylor polynomial of the event function x is the zero polynomial
    // in the lanes in which x is zero. We always pass explicit timesteps, as the automatically-deduced ones are
    // infinite.

    // Helper to check the step outcomes.
    const auto check_step = [](const ta_t &ta, const std::vector<std::tuple<taylor_outcome, double>> &expected) {
        REQUIRE(ta.get_step_res() == expected);
    };

    // Non-terminal events. x is zero only in lane 0.
    {
        auto x = make_vars("x");

        // The (lane, time, direction) records of the non-terminal events, with any and positive direction
        // respectively.
        rec_t any_records, pos_records;

        auto ta = ta_t{{prime(x) = 0_dbl},
                       {0., 1.},
                       2,
                       kw::nt_events = {nt_ev_t(
                                            x,
                                            [&any_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                any_records.emplace_back(batch_idx, tm, d_sgn);
                                            },
                                            kw::cross_step = true),
                                        nt_ev_t(
                                            x,
                                            [&pos_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                pos_records.emplace_back(batch_idx, tm, d_sgn);
                                            },
                                            kw::direction = event_direction::positive, kw::cross_step = true)}};

        // Two steps: in lane 0, a root with zero direction is reported at the beginning of each step (and filtered out
        // by the event with positive direction). No cross-step event is reported in either lane.
        ta.step(std::vector<double>{.25, .25});
        ta.step(std::vector<double>{.25, .25});
        check_step(ta, {{taylor_outcome::time_limit, .25}, {taylor_outcome::time_limit, .25}});
        REQUIRE(any_records == rec_t{{0u, 0., 0}, {0u, .25, 0}});
        REQUIRE(pos_records.empty());

        // Jump of x away from zero in lane 0 (from the stored zero sign), and through zero in lane 1. The jumps are
        // reported at the beginning of the next step.
        ta.get_state_data()[0] = 1.;
        ta.get_state_data()[1] = -1.;
        ta.step(std::vector<double>{.25, .25});
        check_step(ta, {{taylor_outcome::time_limit, .25}, {taylor_outcome::time_limit, .25}});
        REQUIRE(any_records == rec_t{{0u, 0., 0}, {0u, .25, 0}, {0u, .5, 1}, {1u, .5, -1}});
        REQUIRE(pos_records == rec_t{{0u, .5, 1}});

        // Nothing else happens in the next step.
        ta.step(std::vector<double>{.25, .25});
        check_step(ta, {{taylor_outcome::time_limit, .25}, {taylor_outcome::time_limit, .25}});
        REQUIRE(any_records == rec_t{{0u, 0., 0}, {0u, .25, 0}, {0u, .5, 1}, {1u, .5, -1}});
        REQUIRE(pos_records == rec_t{{0u, .5, 1}});
    }

    // Terminal event with automatic cooldown. x is zero only in lane 0.
    {
        auto x = make_vars("x");

        // The trigger times of the terminal event, for each lane.
        std::vector<std::vector<double>> trigger_times(2);

        auto ta = ta_t{{prime(x) = 0_dbl},
                       {0., 1.},
                       2,
                       kw::t_events = {t_ev_t(
                           x,
                           kw::callback =
                               [&trigger_times](ta_t &t, int, std::uint32_t batch_idx) {
                                   trigger_times[batch_idx].push_back(t.get_time()[batch_idx]);
                                   return true;
                               },
                           kw::cross_step = true)}};

        // First step: the terminal event triggers at the beginning of the step in lane 0.
        ta.step(std::vector<double>{.25, .25});
        check_step(ta, {{taylor_outcome{0}, 0.}, {taylor_outcome::time_limit, .25}});
        REQUIRE(trigger_times == std::vector<std::vector<double>>{{0.}, {}});

        // NOTE: the derivative of the event function is zero, hence the cooldown heuristic produces an infinite value,
        // which is replaced by the largest finite value.
        REQUIRE(ta.get_te_cooldowns()[0][0]);
        REQUIRE(ta.get_te_cooldowns()[0][0]->second == std::numeric_limits<double>::max());
        REQUIRE(!ta.get_te_cooldowns()[1][0]);

        // The next steps proceed normally, the terminal event stays on cooldown in lane 0.
        for (auto i = 0; i < 3; ++i) {
            ta.step(std::vector<double>{.25, .25});
            check_step(ta, {{taylor_outcome::time_limit, .25}, {taylor_outcome::time_limit, .25}});
        }
        REQUIRE(trigger_times == std::vector<std::vector<double>>{{0.}, {}});
    }

    // Terminal event with custom cooldown. x is zero in both lanes, the timesteps differ.
    {
        auto x = make_vars("x");

        // The trigger times of the terminal event, for each lane.
        std::vector<std::vector<double>> trigger_times(2);

        auto ta = ta_t{{prime(x) = 0_dbl},
                       {0., 0.},
                       2,
                       kw::t_events = {t_ev_t(
                           x,
                           kw::callback =
                               [&trigger_times](ta_t &t, int, std::uint32_t batch_idx) {
                                   trigger_times[batch_idx].push_back(t.get_time()[batch_idx]);
                                   return true;
                               },
                           kw::cooldown = .3, kw::cross_step = true)}};

        // The terminal event triggers at the beginning of a step (with a zero timestep) whenever it is not on cooldown.
        // After each trigger, the cooldown of 0.3 lasts for two steps of size 0.25 in lane 0 and 0.2 in lane 1.
        for (auto i = 0; i < 2; ++i) {
            ta.step(std::vector<double>{.25, .2});
            check_step(ta, {{taylor_outcome{0}, 0.}, {taylor_outcome{0}, 0.}});
            for (auto j = 0; j < 2; ++j) {
                ta.step(std::vector<double>{.25, .2});
                check_step(ta, {{taylor_outcome::time_limit, .25}, {taylor_outcome::time_limit, .2}});
            }
        }
        REQUIRE(trigger_times == std::vector<std::vector<double>>{{0., .5}, {0., .4}});
    }
}

// Cross-step detection is opt-in: events constructed without the cross-step flag do not report jumps across step
// boundaries.
TEST_CASE("opt in")
{
    using ta_t = taylor_adaptive_batch<double>;
    using t_ev_t = ta_t::t_event_t;
    using nt_ev_t = ta_t::nt_event_t;
    using rec_t = std::vector<std::tuple<std::uint32_t, double, int>>;

    // Helper to take a first step, at the end of which v is negative in lane 0 and positive in lane 1, and then to make v
    // jump through zero in both lanes (upwards in lane 0, downwards in lane 1).
    const auto first_step_and_jump = [](ta_t &ta) {
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }

        ta.get_state_data()[0] = 1.;
        ta.get_state_data()[1] = -1.;
    };

    // Non-terminal events: two events on v, only the second one with the cross-step flag.
    {
        auto v = make_vars("v");

        // The (lane, time, direction) records of the non-terminal events.
        rec_t def_records, cs_records;

        auto ta = ta_t{{prime(v) = -1_dbl},
                       {-1., 2.},
                       2,
                       kw::nt_events = {nt_ev_t(v,
                                                [&def_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                    def_records.emplace_back(batch_idx, tm, d_sgn);
                                                }),
                                        nt_ev_t(
                                            v,
                                            [&cs_records](ta_t &, double tm, int d_sgn, std::uint32_t batch_idx) {
                                                cs_records.emplace_back(batch_idx, tm, d_sgn);
                                            },
                                            kw::cross_step = true)}};

        first_step_and_jump(ta);

        // Second step: the jumps are reported only by the event with the cross-step flag. v does not cross zero within
        // the step in either lane.
        ta.step(std::vector<double>{.25, .25});
        for (const auto &[oc, h] : ta.get_step_res()) {
            REQUIRE(oc == taylor_outcome::time_limit);
        }
        REQUIRE(def_records.empty());
        REQUIRE(cs_records == rec_t{{0u, .25, 1}, {1u, .25, -1}});
    }

    // Terminal events: two events on v without callback, only the second one with the cross-step flag. The second
    // event has positive direction, so that it triggers only in lane 0.
    {
        auto v = make_vars("v");

        auto ta = ta_t{{prime(v) = -1_dbl},
                       {-1., 2.},
                       2,
                       kw::t_events = {t_ev_t(v),
                                       t_ev_t(v, kw::direction = event_direction::positive, kw::cross_step = true)}};

        first_step_and_jump(ta);

        // Second step: only the second event triggers at the beginning of the step in lane 0. Nothing happens in lane 1.
        //
        // NOTE: if both events triggered in lane 0, the one with the lowest index would take precedence.
        ta.step(std::vector<double>{.25, .25});
        REQUIRE(ta.get_step_res()[0] == std::tuple{taylor_outcome{-2}, 0.});
        REQUIRE(ta.get_step_res()[1] == std::tuple{taylor_outcome::time_limit, .25});
    }
}
