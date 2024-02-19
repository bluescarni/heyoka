// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <vector>

#include <heyoka/callable.hpp>
#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const expression &x)
{
    return x * x;
}

TEST_CASE("nte copy semantics")
{
    using ev_t = taylor_adaptive_batch<double>::nt_event_t;

    auto v = make_vars("v");

    auto ex = v + 3_dbl;

    // Expression is shallow-copied on construction.
    ev_t ev(ex, [](auto &, double, int, std::uint32_t) {});
    REQUIRE(std::get<func>(ex.value()).get_ptr() == std::get<func>(ev.get_expression().value()).get_ptr());

    // Copy ctor.
    auto ev2 = ev;

    REQUIRE(std::get<func>(ev.get_expression().value()).get_ptr()
            == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Self assignment.
    auto orig_id = std::get<func>(ev2.get_expression().value()).get_ptr();
    ev2 = *&ev2;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Copy assignment.
    ev2 = ev;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());
}

TEST_CASE("te copy semantics")
{
    using ev_t = taylor_adaptive_batch<double>::t_event_t;

    auto v = make_vars("v");

    auto ex = v + 3_dbl;

    // Expression is shallow-copied on construction.
    ev_t ev(ex);
    REQUIRE(std::get<func>(ex.value()).get_ptr() == std::get<func>(ev.get_expression().value()).get_ptr());

    // Copy ctor.
    auto ev2 = ev;

    REQUIRE(std::get<func>(ev.get_expression().value()).get_ptr()
            == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Self assignment.
    auto orig_id = std::get<func>(ev2.get_expression().value()).get_ptr();
    ev2 = *&ev2;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Copy assignment.
    ev2 = ev;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());
}

TEST_CASE("nte single step")
{
    auto tester = [](unsigned opt_level) {
        using fp_t = double;

        const auto batch_size = 4u;

        auto [x, v] = make_vars("x", "v");

        // Batches of initial conditions and parameters.
        const std::vector<fp_t> ic_vals = {0.00, 0.01, 0.02, 0.03, 1.85, 1.86, 1.87, 1.88};
        const std::vector<fp_t> par_vals = {0.10, 0.11, 0.12, 0.13};

        // Dynamics.
        auto dynamics = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

        // Events.
        const auto ev_eq = x + .1;
        std::vector<fp_t> trig_times, v_vals;
        nt_event<fp_t> ev{ev_eq,
                          [&trig_times, &v_vals](auto &tint, fp_t tm, int) {
                              trig_times.push_back(tm);
                              tint.update_d_output(tm);
                              v_vals.push_back(tint.get_d_output()[1]);
                          },
                          kw::direction = event_direction::negative};
        std::vector<std::vector<fp_t>> trig_times_batch(batch_size), v_vals_batch(batch_size);
        nt_event_batch<fp_t> ev_batch{
            ev_eq,
            [&trig_times_batch, batch_size, &v_vals_batch](auto &tint, fp_t tm, int, std::uint32_t batch_idx) {
                trig_times_batch[batch_idx].push_back(tm);
                tint.update_d_output(std::vector<fp_t>(batch_size, tm));
                v_vals_batch[batch_idx].push_back(tint.get_d_output()[batch_size + batch_idx]);
            },
            kw::direction = event_direction::negative};

        auto ta = taylor_adaptive<fp_t>{dynamics, {0., 0.}, kw::opt_level = opt_level, kw::nt_events = {ev}};
        auto ta_batch = taylor_adaptive_batch<fp_t>{
            dynamics, ic_vals, batch_size, kw::opt_level = opt_level, kw::pars = par_vals, kw::nt_events = {ev_batch}};

        // Integrate the batch integrator up to t = 20.
        while (std::any_of(ta_batch.get_time().begin(), ta_batch.get_time().end(), [](auto tm) { return tm < 20; })) {
            ta_batch.step();

            for (std::uint32_t i = 0; i < batch_size; ++i) {
                REQUIRE(std::get<0>(ta_batch.get_step_res()[i]) == taylor_outcome::success);
            }
        }

        // For each batch index, integrate the scalar integrator up to
        // t = 20 and compare the results.
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            // Reset the scalar integrator.
            trig_times.clear();
            v_vals.clear();
            ta.set_time(0.);
            ta.get_state_data()[0] = ic_vals[i];
            ta.get_state_data()[1] = ic_vals[batch_size + i];
            ta.get_pars_data()[0] = par_vals[i];

            while (ta.get_time() < 20) {
                ta.step();
            }

            REQUIRE(trig_times.size() == trig_times_batch[i].size());
            for (decltype(trig_times.size()) j = 0; j < trig_times.size(); ++j) {
                REQUIRE(trig_times[j] == approximately(trig_times_batch[i][j], 1000.));
                REQUIRE(v_vals[j] == approximately(v_vals_batch[i][j], 1000.));
            }
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tester(opt_level);
    }
}

TEST_CASE("te single step")
{
    auto tester = [](unsigned opt_level) {
        using fp_t = double;

        const auto batch_size = 4u;

        auto [x, v] = make_vars("x", "v");

        // Batches of initial conditions and parameters.
        const std::vector<fp_t> ic_vals = {0.00, 0.01, 0.02, 0.03, 1.85, 1.86, 1.87, 1.88};
        const std::vector<fp_t> par_vals = {0.10, 0.11, 0.12, 0.13};

        // The expected number of triggers for each batch index.
        const std::vector<unsigned> ex_n_trig = {2, 1, 1, 1};

        // Dynamics.
        auto dynamics = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

        // Events.
        const auto ev_eq = x + .1;
        std::vector<fp_t> trig_times, v_vals;
        t_event<fp_t> ev{ev_eq,
                         kw::callback =
                             [&trig_times, &v_vals](auto &tint, int) {
                                 trig_times.push_back(tint.get_time());
                                 v_vals.push_back(tint.get_state()[1]);
                                 return true;
                             },
                         kw::direction = event_direction::negative};
        std::vector<std::vector<fp_t>> trig_times_batch(batch_size), v_vals_batch(batch_size);
        t_event_batch<fp_t> ev_batch{
            ev_eq,
            kw::callback =
                [&trig_times_batch, batch_size, &v_vals_batch](auto &tint, int, std::uint32_t batch_idx) {
                    trig_times_batch[batch_idx].push_back(tint.get_time()[batch_idx]);
                    v_vals_batch[batch_idx].push_back(tint.get_state()[batch_size + batch_idx]);
                    return true;
                },
            kw::direction = event_direction::negative};

        auto ta = taylor_adaptive<fp_t>{dynamics, {0., 0.}, kw::opt_level = opt_level, kw::t_events = {ev}};
        auto ta_batch = taylor_adaptive_batch<fp_t>{
            dynamics, ic_vals, batch_size, kw::opt_level = opt_level, kw::pars = par_vals, kw::t_events = {ev_batch}};

        // Integrate the batch integrator up to t = 20.
        while (std::any_of(ta_batch.get_time().begin(), ta_batch.get_time().end(), [](auto tm) { return tm < 20; })) {
            ta_batch.step();

            for (std::uint32_t i = 0; i < batch_size; ++i) {
                const auto oc = std::get<0>(ta_batch.get_step_res()[i]);
                REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) == 0));
            }
        }

        // For each batch index, integrate the scalar integrator up to
        // t = 20 and compare the results.
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            // Reset the scalar integrator.
            trig_times.clear();
            v_vals.clear();
            ta.set_time(0.);
            ta.get_state_data()[0] = ic_vals[i];
            ta.get_state_data()[1] = ic_vals[batch_size + i];
            ta.get_pars_data()[0] = par_vals[i];

            while (ta.get_time() < 20) {
                ta.step();
            }

            REQUIRE(trig_times.size() == ex_n_trig[i]);
            REQUIRE(trig_times.size() == trig_times_batch[i].size());
            for (decltype(trig_times.size()) j = 0; j < trig_times.size(); ++j) {
                REQUIRE(trig_times[j] == approximately(trig_times_batch[i][j], 1000.));
                REQUIRE(v_vals[j] == approximately(v_vals_batch[i][j], 1000.));
            }
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tester(opt_level);
    }
}

// Test for an event triggering exactly at the end of a timestep.
TEST_CASE("nte linear box")
{
    using ev_t = taylor_adaptive_batch<double>::nt_event_t;

    auto x = make_vars("x");

    auto counter = 0u;

    auto ta_ev = taylor_adaptive_batch<double>{{prime(x) = par[0]},
                                               {0., 0., 0., 0.},
                                               4,
                                               kw::nt_events
                                               = {ev_t(x - 1.,
                                                       [&counter](auto &tint, double tm, int, std::uint32_t batch_idx) {
                                                           REQUIRE(tm == approximately(1 / tint.get_pars()[batch_idx]));
                                                           ++counter;
                                                       })},
                                               kw::pars = {1., 2., 4., 8.}};

    // Check that the event triggers at the beginning of the second step.
    ta_ev.step({1., 1 / 2., 1 / 4., 1 / 8.});

    REQUIRE(counter == 0u);
    REQUIRE(std::all_of(ta_ev.get_step_res().begin(), ta_ev.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    ta_ev.step({1., 1 / 2., 1 / 4., 1 / 8.});
    REQUIRE(counter == 4u);
    REQUIRE(std::all_of(ta_ev.get_step_res().begin(), ta_ev.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));
}

// Test for an event triggering exactly at the end of a timestep.
TEST_CASE("te linear box")
{
    using ev_t = taylor_adaptive_batch<double>::t_event_t;

    auto x = make_vars("x");

    auto counter = 0u;

    auto ta_ev = taylor_adaptive_batch<double>{{prime(x) = par[0]},
                                               {0., 0., 0., 0.},
                                               4,
                                               kw::t_events = {ev_t(
                                                   x - 1., kw::callback =
                                                               [&counter](auto &, int, std::uint32_t) {
                                                                   ++counter;
                                                                   return true;
                                                               })},
                                               kw::pars = {1., 2., 4., 8.}};

    // Check that the event triggers at the beginning of the second step.
    ta_ev.step({1., 1 / 2., 1 / 4., 1 / 8.});

    REQUIRE(counter == 0u);
    REQUIRE(std::all_of(ta_ev.get_step_res().begin(), ta_ev.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    ta_ev.step({1., 1 / 2., 1 / 4., 1 / 8.});
    REQUIRE(counter == 4u);
    REQUIRE(std::all_of(ta_ev.get_step_res().begin(), ta_ev.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; }));
    REQUIRE(std::all_of(ta_ev.get_step_res().begin(), ta_ev.get_step_res().end(),
                        [](const auto &t) { return std::get<1>(t) == approximately(0.); }));
}

TEST_CASE("nte glancing blow test")
{
    std::cout << "Starting glancing blow test...\n";

    // NOTE: in this test two spherical particles are
    // "colliding" in a glancing fashion, meaning that
    // the polynomial representing the evolution in time
    // of the mutual distance has a repeated root at
    // t = collision time.
    using fp_t = double;

    auto [x0, vx0, x1, vx1] = make_vars("x0", "vx0", "x1", "vx1");
    auto [y0, vy0, y1, vy1] = make_vars("y0", "vy0", "y1", "vy1");

    using ev_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;

    auto counter = 0u;

    // First setup: one particle still, the other moving with uniform velocity.
    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x0) = vx0, prime(y0) = vy0, prime(x1) = vx1, prime(y1) = vy1, prime(vx0) = 0_dbl, prime(vy0) = 0_dbl,
         prime(vx1) = 0_dbl, prime(vy1) = 0_dbl},
        {0.,   0.,   0.,   0.,   //
         0.,   0.,   0.,   0.,   //
         -10., -10., -10., -10., //
         6.,   2,    7.,   8.,   // Glancer on batch index 1.
         0.,   0.,   0.,   0.,   //
         0.,   0.,   0.,   0.,   //
         1.,   1.,   1.,   1.,   //
         0.,   0.,   0.,   0.},
        4,
        kw::nt_events = {ev_t(square_wrapper(x0 - x1) + square_wrapper(y0 - y1) - 4.,
                              [&counter](auto &, fp_t t, int, std::uint32_t batch_idx) {
                                  REQUIRE((t - 10.) * (t - 10.) <= std::numeric_limits<fp_t>::epsilon());
                                  REQUIRE(batch_idx == 1u);

                                  ++counter;
                              })}};

    for (auto i = 0; i < 20; ++i) {
        ta.step({1.3, 1.3, 1.3, 1.3});

        REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                            [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));
    }

    // Any number of events up to 2 is acceptable here.
    REQUIRE(counter <= 2u);

    counter = 0;

    // Second setup: one particle still, the other accelerating towards positive
    // x direction with constant acceleration.
    ta = taylor_adaptive_batch<fp_t>{{prime(x0) = vx0, prime(y0) = vy0, prime(x1) = vx1, prime(y1) = vy1,
                                      prime(vx0) = 0_dbl, prime(vy0) = 0_dbl, prime(vx1) = .1_dbl, prime(vy1) = 0_dbl},
                                     {0.,   0.,   0.,   0.,   //
                                      0.,   0.,   0.,   0.,   //
                                      -10., -10., -10., -10., //
                                      6.,   2,    7.,   8.,   // Glancer on batch index 1.
                                      0.,   0.,   0.,   0.,   //
                                      0.,   0.,   0.,   0.,   //
                                      1.,   1.,   1.,   1.,   //
                                      0.,   0.,   0.,   0.},
                                     4,
                                     kw::nt_events = {ev_t(square_wrapper(x0 - x1) + square_wrapper(y0 - y1) - 4.,
                                                           [&counter](auto &, fp_t, int, std::uint32_t batch_idx) {
                                                               REQUIRE(batch_idx == 1u);
                                                               ++counter;
                                                           })}};

    for (auto i = 0; i < 20; ++i) {
        ta.step({1.3, 1.3, 1.3, 1.3});

        REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                            [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));
    }

    REQUIRE(counter <= 2u);

    std::cout << "Glancing blow test finished\n";
}

TEST_CASE("nte multizero")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using ev_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;

    auto counter = std::vector{0u, 0u, 0u, 0u};

    // In this test, we define two events:
    // - the velocity is smaller in absolute
    //   value than a small limit,
    // - the velocity is exactly zero.
    // It is likely that both events are going to fire
    // in the same timestep, with the first event
    // firing twice. The sequence of events must
    // be 0 1 0 repeated a few times.

    auto cur_time = std::vector{0., 0., 0., 0.};

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::nt_events = {ev_t(v * v - 1e-10,
                              [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t > cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta_.get_time()[batch_idx] > t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta_.update_d_output({t, t, t, t});

                                  const auto vel = ta_.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t > cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta_.get_time()[batch_idx] > t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta_.update_d_output({t, t, t, t});

                             const auto vel = ta_.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                             ++counter[batch_idx];

                             cur_time[batch_idx] = t;
                         })}};

    ta.propagate_until({4., 4., 4., 4.});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(counter[i] == 12u);
    }

    std::fill(counter.begin(), counter.end(), 0u);
    std::fill(cur_time.begin(), cur_time.end(), 0.);

    // Run the same test with sub-eps tolerance too.
    ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::tol = std::numeric_limits<fp_t>::epsilon() / 100,
        kw::nt_events = {ev_t(v * v - 1e-10,
                              [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t > cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta_.get_time()[batch_idx] > t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta_.update_d_output({t, t, t, t});

                                  const auto vel = ta_.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t > cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta_.get_time()[batch_idx] > t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta_.update_d_output({t, t, t, t});

                             const auto vel = ta_.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                             ++counter[batch_idx];

                             cur_time[batch_idx] = t;
                         })}};

    ta.propagate_until({4., 4., 4., 4.});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(counter[i] == 12u);
    }

    std::fill(counter.begin(), counter.end(), 0u);
    std::fill(cur_time.begin(), cur_time.end(), 0.);

    // We re-run the test, but this time we want to detect
    // only when the velocity goes from positive to negative.
    // Thus the sequence of events will be:
    // - 0 1 0
    // - 0 0
    // - 0 1 0
    // - 0 0

    ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                     {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                     4,
                                     kw::nt_events
                                     = {ev_t(v * v - 1e-10,
                                             [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                                 using std::abs;

                                                 // Make sure the callbacks are called in order.
                                                 REQUIRE(t > cur_time[batch_idx]);

                                                 // Ensure the state of ta has
                                                 // been propagated until after the
                                                 // event.
                                                 REQUIRE(ta_.get_time()[batch_idx] > t);

                                                 REQUIRE((counter[batch_idx] == 0u
                                                          || (counter[batch_idx] >= 2u && counter[batch_idx] <= 6u)
                                                          || (counter[batch_idx] >= 7u && counter[batch_idx] <= 9u)));

                                                 ta_.update_d_output({t, t, t, t});

                                                 const auto vel = ta_.get_d_output()[4u + batch_idx];
                                                 REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                                 ++counter[batch_idx];

                                                 cur_time[batch_idx] = t;
                                             }),
                                        ev_t(
                                            v,
                                            [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                                using std::abs;

                                                // Make sure the callbacks are called in order.
                                                REQUIRE(t > cur_time[batch_idx]);

                                                // Ensure the state of ta has
                                                // been propagated until after the
                                                // event.
                                                REQUIRE(ta_.get_time()[batch_idx] > t);

                                                REQUIRE((counter[batch_idx] == 1u || counter[batch_idx] == 6u));

                                                ta_.update_d_output({t, t, t, t});

                                                const auto vel = ta_.get_d_output()[4u + batch_idx];
                                                REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                                ++counter[batch_idx];

                                                cur_time[batch_idx] = t;
                                            },
                                            kw::direction = event_direction::negative)}};

    ta.propagate_until({4., 4., 4., 4.});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(counter[i] == 10u);
    }

    std::fill(counter.begin(), counter.end(), 0u);
    std::fill(cur_time.begin(), cur_time.end(), 0.);

    // Sub-eps tolerance too.
    ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                     {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                     4,
                                     kw::tol = std::numeric_limits<fp_t>::epsilon() / 100,
                                     kw::nt_events
                                     = {ev_t(v * v - 1e-10,
                                             [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                                 using std::abs;

                                                 // Make sure the callbacks are called in order.
                                                 REQUIRE(t > cur_time[batch_idx]);

                                                 // Ensure the state of ta has
                                                 // been propagated until after the
                                                 // event.
                                                 REQUIRE(ta_.get_time()[batch_idx] > t);

                                                 REQUIRE((counter[batch_idx] == 0u
                                                          || (counter[batch_idx] >= 2u && counter[batch_idx] <= 6u)
                                                          || (counter[batch_idx] >= 7u && counter[batch_idx] <= 9u)));

                                                 ta_.update_d_output({t, t, t, t});

                                                 const auto vel = ta_.get_d_output()[4u + batch_idx];
                                                 REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                                 ++counter[batch_idx];

                                                 cur_time[batch_idx] = t;
                                             }),
                                        ev_t(
                                            v,
                                            [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                                using std::abs;

                                                // Make sure the callbacks are called in order.
                                                REQUIRE(t > cur_time[batch_idx]);

                                                // Ensure the state of ta has
                                                // been propagated until after the
                                                // event.
                                                REQUIRE(ta_.get_time()[batch_idx] > t);

                                                REQUIRE((counter[batch_idx] == 1u || counter[batch_idx] == 6u));

                                                ta_.update_d_output({t, t, t, t});

                                                const auto vel = ta_.get_d_output()[4u + batch_idx];
                                                REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                                ++counter[batch_idx];

                                                cur_time[batch_idx] = t;
                                            },
                                            kw::direction = event_direction::negative)}};

    ta.propagate_until({4., 4., 4., 4.});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(counter[i] == 10u);
    }
}

TEST_CASE("nte multizero negative timestep")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using ev_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;

    auto counter = std::vector{0u, 0u, 0u, 0u};

    auto cur_time = std::vector{0., 0., 0., 0.};

    // In this test, we define two events:
    // - the velocity is smaller in absolute
    //   value than a small limit,
    // - the velocity is exactly zero.
    // It is likely that both events are going to fire
    // in the same timestep, with the first event
    // firing twice. The sequence of events must
    // be 0 1 0 repeated a few times.
    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::nt_events = {ev_t(v * v - 1e-10,
                              [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t < cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta_.get_time()[batch_idx] < t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta_.update_d_output({t, t, t, t});

                                  const auto vel = ta_.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta_, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t < cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta_.get_time()[batch_idx] < t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta_.update_d_output({t, t, t, t});

                             const auto vel = ta_.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                             ++counter[batch_idx];

                             cur_time[batch_idx] = t;
                         })}};

    ta.propagate_until({-4., -4., -4., -4.});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(counter[i] == 12u);
    }
}

TEST_CASE("nte basic")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using ev_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;

    auto counter = std::vector{0u, 0u, 0u, 0u};
    const std::vector periods
        = {2.0149583072955119566777324135479727911105583481363, 2.015602866455777600694040810649276304933055944554756,
           2.0162731039077591887007722648120652760856018525920970125217,
           2.01696906642817313582861191326257261662145101139954930969969};

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {-0.25, -0.26, -0.27, -0.28, 0., 0., 0., 0.},
                                          4,
                                          kw::nt_events
                                          = {ev_t(v, [&counter, &periods](auto &, fp_t t, int, std::uint32_t idx) {
                                                // Check that the first event detection happens at t == 0.
                                                if (counter[idx] == 0u) {
                                                    REQUIRE(t == 0);
                                                }

                                                // Make sure the 3rd event detection corresponds
                                                // to a full period.
                                                if (counter[idx] == 2u) {
                                                    REQUIRE(t == approximately(periods[idx], 1000.));
                                                }

                                                ++counter[idx];
                                            })}};

    for (auto i = 0; i < 20; ++i) {
        ta.step();

        REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                            [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));
    }

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(counter[i] == 3u);
    }
}

TEST_CASE("te basic")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;
    using nt_ev_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;

    // NOTE: test also sub-eps tolerance.
    for (auto cur_tol : {std::numeric_limits<fp_t>::epsilon(), std::numeric_limits<fp_t>::epsilon() / 100}) {
        std::vector<unsigned> counter_nt(4u, 0u), counter_t(4u, 0u);
        std::vector<fp_t> cur_time(4u, 0.);
        bool direction = true;

        auto ta = taylor_adaptive_batch<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
            4,
            kw::tol = cur_tol,
            kw::nt_events = {nt_ev_t(v * v - 1e-10,
                                     [&counter_nt, &cur_time, &direction](auto &ta_, fp_t t, int, std::uint32_t idx) {
                                         // Make sure the callbacks are called in order.
                                         if (direction) {
                                             REQUIRE(t > cur_time[idx]);
                                         } else {
                                             REQUIRE(t < cur_time[idx]);
                                         }

                                         ta_.update_d_output({t, t, t, t});

                                         const auto vel = ta_.get_d_output()[4u + idx];
                                         REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                         ++counter_nt[idx];

                                         cur_time[idx] = t;
                                     })},
            kw::t_events = {t_ev_t(
                v, kw::callback = [&counter_t, &cur_time, &direction](auto &ta_, int, std::uint32_t idx) {
                    const auto &t = ta_.get_time();

                    if (direction) {
                        REQUIRE(t[idx] > cur_time[idx]);
                    } else {
                        REQUIRE(t[idx] < cur_time[idx]);
                    }

                    const auto vel = ta_.get_state()[4u + idx];
                    REQUIRE(abs(vel) < std::numeric_limits<fp_t>::epsilon() * 100);

                    ++counter_t[idx];

                    cur_time[idx] = t[idx];

                    return true;
                })}};

        // Propagate all batches up to the first trigger of the terminal event.
        auto n_trig = 0u;
        std::vector<fp_t> max_delta_t(4u, std::numeric_limits<fp_t>::infinity());
        while (true) {
            ta.step(max_delta_t);
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const auto oc = std::get<0>(ta.get_step_res()[i]);
                if (oc > taylor_outcome::success) {
                    REQUIRE(oc >= taylor_outcome{0});
                    ++n_trig;
                    max_delta_t[i] = 0;
                } else {
                    REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
                }
            }

            if (n_trig >= 4u) {
                break;
            }
        }

        for (std::uint32_t i = 0; i < 4u; ++i) {
            REQUIRE(counter_nt[i] == 1u);
            REQUIRE(counter_t[i] == 1u);
        }

        // Again.
        n_trig = 0;
        max_delta_t = std::vector<fp_t>(4u, std::numeric_limits<fp_t>::infinity());
        while (true) {
            ta.step(max_delta_t);
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const auto oc = std::get<0>(ta.get_step_res()[i]);
                if (oc > taylor_outcome::success) {
                    REQUIRE(oc >= taylor_outcome{0});
                    ++n_trig;
                    max_delta_t[i] = 0;
                } else {
                    REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
                }
            }

            if (n_trig >= 4u) {
                break;
            }
        }

        for (std::uint32_t i = 0; i < 4u; ++i) {
            REQUIRE(counter_nt[i] == 3u);
            REQUIRE(counter_t[i] == 2u);
        }

        // Move backwards.
        direction = false;
        n_trig = 0;
        std::fill(max_delta_t.begin(), max_delta_t.end(), -std::numeric_limits<fp_t>::infinity());
        while (true) {
            ta.step(max_delta_t);
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const auto oc = std::get<0>(ta.get_step_res()[i]);
                if (oc > taylor_outcome::success) {
                    REQUIRE(oc >= taylor_outcome{0});
                    ++n_trig;
                    max_delta_t[i] = 0;
                } else {
                    REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
                }
            }

            if (n_trig >= 4u) {
                break;
            }
        }

        for (std::uint32_t i = 0; i < 4u; ++i) {
            REQUIRE(counter_nt[i] == 5u);
            REQUIRE(counter_t[i] == 3u);
        }

        // Again.
        n_trig = 0;
        std::fill(max_delta_t.begin(), max_delta_t.end(), -std::numeric_limits<fp_t>::infinity());
        while (true) {
            ta.step(max_delta_t);
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const auto oc = std::get<0>(ta.get_step_res()[i]);
                if (oc > taylor_outcome::success) {
                    REQUIRE(oc >= taylor_outcome{0});
                    ++n_trig;
                    max_delta_t[i] = 0;
                } else {
                    REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
                }
            }

            if (n_trig >= 4u) {
                break;
            }
        }

        for (std::uint32_t i = 0; i < 4u; ++i) {
            REQUIRE(counter_nt[i] == 7u);
            REQUIRE(counter_t[i] == 4u);
        }
    }
}

TEST_CASE("nte dir test")
{
    auto [x, v] = make_vars("x", "v");

    bool fwd = true;

    std::vector<std::vector<double>> tlist(4u);

    std::vector<std::vector<double>::reverse_iterator> rit(4u);

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {-0.25, -0.26, -0.27, -0.28, 0., 0., 0., 0.},
                                            4,
                                            kw::nt_events = {nt_event_batch<double>(
                                                v,
                                                [&fwd, &tlist, &rit](auto &, double t, int d_sgn, std::uint32_t idx) {
                                                    REQUIRE(d_sgn == 1);

                                                    if (fwd) {
                                                        tlist[idx].push_back(t);
                                                    } else if (rit[idx] != tlist[idx].rend()) {
                                                        REQUIRE(*rit[idx] == approximately(t));

                                                        ++rit[idx];
                                                    }
                                                },
                                                kw::direction = event_direction::positive)}};

    ta.propagate_until({20, 20, 20, 20});

    fwd = false;
    for (auto i = 0u; i < 4u; ++i) {
        rit[i] = tlist[i].rbegin();
    }

    ta.propagate_until({0, 0, 0, 0});
}

TEST_CASE("nte def ctor")
{
    nt_event_batch<double> nte;

    REQUIRE(nte.get_expression() == 0_dbl);
    REQUIRE(nte.get_callback());
    REQUIRE(nte.get_direction() == event_direction::any);
}

struct s11n_nte_callback {
    template <typename I, typename T>
    void operator()(I &, T, int, std::uint32_t) const
    {
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(s11n_nte_callback, void, taylor_adaptive_batch<double> &, double, int, std::uint32_t)

TEST_CASE("nte s11n")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    nt_event_batch<fp_t> ev(v, s11n_nte_callback{}, kw::direction = event_direction::positive);

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ev;
    }

    ev = nt_event_batch<fp_t>(v + x, [](auto &, fp_t, int, std::uint32_t) {});

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ev;
    }

    REQUIRE(ev.get_expression() == v);
    REQUIRE(ev.get_direction() == event_direction::positive);
    REQUIRE(ev.get_callback().get_type_index() == typeid(s11n_nte_callback));
}

TEST_CASE("te identical")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    t_ev_t ev(v);

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                          4,
                                          kw::t_events = {ev, ev}};

    while (true) {
        ta.step();

        if (std::any_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) > taylor_outcome::success; })) {
            break;
        }
    }

    std::vector<std::int64_t> ev_idx(4u, 0);
    for (std::uint32_t i = 0; i < 4u; ++i) {
        const auto oc = std::get<0>(ta.get_step_res()[i]);
        REQUIRE(oc > taylor_outcome::success);
        auto first_ev = -static_cast<std::int64_t>(oc) - 1;
        REQUIRE((first_ev == 0 || first_ev == 1));
        ev_idx[i] = first_ev;
    }

    // Taking a further step, we might either detect the second event,
    // or it may end up being ignored due to numerics.
    ta.step();

    for (std::uint32_t i = 0; i < 4u; ++i) {
        const auto oc = std::get<0>(ta.get_step_res()[i]);
        if (oc > taylor_outcome::success) {
            auto second_ev = -static_cast<std::int64_t>(oc) - 1;
            REQUIRE((second_ev == 0 || second_ev == 1));
            REQUIRE(second_ev != ev_idx[i]);
        } else {
            REQUIRE(oc == taylor_outcome::success);
        }
    }
}

TEST_CASE("te close")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    t_ev_t ev1(x);
    t_ev_t ev2(
        x - std::numeric_limits<fp_t>::epsilon() * 2, kw::callback = [](auto &, int, std::uint32_t) { return true; });

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {0.1, 0.11, 0.12, 0.13, .25, .26, .27, .28},
                                          4,
                                          kw::t_events = {ev1, ev2}};

    // Propagate all batches up to the first trigger of a terminal event.
    auto n_trig = 0u;
    std::vector<fp_t> max_delta_t(4u, std::numeric_limits<fp_t>::infinity());
    std::vector<std::int64_t> trig_idx(4u, 0);
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(oc >= taylor_outcome{0});
                ++n_trig;
                max_delta_t[i] = 0;
                trig_idx[i] = static_cast<std::int64_t>(oc);
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }

    // The second event must have triggered first.
    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(trig_idx[i] == 1);
    }

    // Next step the first event must trigger.
    n_trig = 0u;
    max_delta_t = std::vector<fp_t>(4u, std::numeric_limits<fp_t>::infinity());
    trig_idx = std::vector<std::int64_t>(4u, 0);
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(oc < taylor_outcome{0});
                ++n_trig;
                max_delta_t[i] = 0;
                trig_idx[i] = static_cast<std::int64_t>(oc);
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }

    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(trig_idx[i] == -1);
    }

    // Next step no event must trigger: event 0 is now on cooldown
    // as it just happened, and event 1 is still close enough to be
    // on cooldown too.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));

    // Go back.
    n_trig = 0u;
    max_delta_t = std::vector<fp_t>(4u, -std::numeric_limits<fp_t>::infinity());
    trig_idx = std::vector<std::int64_t>(4u, 0);
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(oc < taylor_outcome{0});
                ++n_trig;
                max_delta_t[i] = 0;
                trig_idx[i] = static_cast<std::int64_t>(oc);
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }

    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(trig_idx[i] == -1);
    }

    n_trig = 0u;
    max_delta_t = std::vector<fp_t>(4u, -std::numeric_limits<fp_t>::infinity());
    trig_idx = std::vector<std::int64_t>(4u, 0);
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(oc >= taylor_outcome{0});
                ++n_trig;
                max_delta_t[i] = 0;
                trig_idx[i] = static_cast<std::int64_t>(oc);
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }

    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(trig_idx[i] == 1);
    }

    // Taking the step forward will skip event zero as it is still
    // on cooldown.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));
}

TEST_CASE("te retrigger")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    t_ev_t ev(x - (par[0] - std::numeric_limits<fp_t>::epsilon() * 6));

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {1., 1.01, 1.02, 1.03, 0., 0.01, 0.02, 0.03},
                                          4,
                                          kw::t_events = {ev},
                                          kw::pars = std::vector<fp_t>{1., 1.01, 1.02, 1.03}};

    // First timestep triggers the event immediately.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return static_cast<std::int64_t>(std::get<0>(t)) == -1; }));
    REQUIRE(std::all_of(ta.get_time().begin(), ta.get_time().end(), [](const auto &t) { return t != 0; }));

    // Step until re-trigger.
    auto n_trig = 0u;
    auto max_delta_t = std::vector<fp_t>(4u, std::numeric_limits<fp_t>::infinity());
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(static_cast<std::int64_t>(oc) == -1);
                ++n_trig;
                max_delta_t[i] = 0;
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }

    // Another step will immediately retrigger.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return static_cast<std::int64_t>(std::get<0>(t)) == -1; }));
}

TEST_CASE("te dir")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    t_ev_t ev(
        v,
        kw::callback =
            [](auto &, int d_sgn, std::uint32_t) {
                REQUIRE(d_sgn == 1);
                return true;
            },
        kw::direction = event_direction::positive);

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {1., 1.01, 1.02, 1.03, 0., 0., 0., 0.}, 4, kw::t_events = {ev}};

    // First timestep must not trigger the event (derivative is negative).
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));

    // Step until trigger.
    while (true) {
        ta.step();
        if (std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; })) {
            break;
        }
    }

    REQUIRE(ta.get_state()[0] == approximately(-1.));
    REQUIRE(ta.get_state()[1] == approximately(-1.01));
    REQUIRE(ta.get_state()[2] == approximately(-1.02));
    REQUIRE(ta.get_state()[3] == approximately(-1.03));

    // Other direction.
    auto ev1 = t_ev_t(
        v,
        kw::callback =
            [](auto &, int d_sgn, std::uint32_t) {
                REQUIRE(d_sgn == -1);
                return true;
            },
        kw::direction = event_direction::negative);

    ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {1., 1.01, 1.02, 1.03, 0., 0., 0., 0.}, 4, kw::t_events = {ev1}};

    // Check that zero timestep does not detect anything.
    ta.step({0., 0., 0., 0.});
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    // Now it must trigger immediately.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; }));

    // The next timestep must not trigger due to cooldown.
    ta.step();
    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));

    // Step until trigger.
    while (true) {
        ta.step();
        if (std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; })) {
            break;
        }
    }

    REQUIRE(ta.get_state()[0] == approximately(1.));
    REQUIRE(ta.get_state()[1] == approximately(1.01));
    REQUIRE(ta.get_state()[2] == approximately(1.02));
    REQUIRE(ta.get_state()[3] == approximately(1.03));
}

TEST_CASE("te custom cooldown")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    t_ev_t ev(
        v * v - std::numeric_limits<fp_t>::epsilon() * 4,
        kw::callback = [](auto &, int, std::uint32_t) { return true; }, kw::cooldown = fp_t(1e-1));

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.02, 0.03, .25, .26, .27, .28}, 4, kw::t_events = {ev}};

    // Step until trigger.
    auto n_trig = 0u;
    auto max_delta_t = std::vector<fp_t>(4u, std::numeric_limits<fp_t>::infinity());
    while (true) {
        ta.step(max_delta_t);
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto oc = std::get<0>(ta.get_step_res()[i]);
            if (oc > taylor_outcome::success) {
                REQUIRE(static_cast<std::int64_t>(oc) == 0);
                ++n_trig;
                max_delta_t[i] = 0;
            } else {
                REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));
            }
        }

        if (n_trig >= 4u) {
            break;
        }
    }
}

TEST_CASE("te propagate_for")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    std::vector<unsigned> counter(4u, 0u);

    t_ev_t ev(
        v, kw::callback = [&counter](auto &, int, std::uint32_t idx) {
            ++counter[idx];
            return true;
        });

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.02, 0.03, .25, .26, .27, .28}, 4, kw::t_events = {ev}};

    ta.propagate_for({100, 100, 100, 100});

    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));
    REQUIRE(std::all_of(ta.get_time().begin(), ta.get_time().end(), [](const auto &t) { return t == 100.; }));
    REQUIRE(std::all_of(counter.begin(), counter.end(), [](const auto &t) { return t == 100u; }));

    t_ev_t ev1(v);

    ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.02, 0.03, .25, .26, .27, .28}, 4, kw::t_events = {ev1}};

    ta.propagate_for({100, 100, 100, 100});

    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return static_cast<std::int64_t>(std::get<0>(t)) == -1; }));
}

TEST_CASE("te propagate_grid")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    std::vector<unsigned> counter(4u, 0u);

    t_ev_t ev(
        v, kw::callback = [&counter](auto &, int, std::uint32_t idx) {
            ++counter[idx];
            return true;
        });

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.02, 0.03, .25, .26, .27, .28}, 4, kw::t_events = {ev}};

    std::vector<fp_t> grid;
    for (auto i = 0; i < 101; ++i) {
        for (std::uint32_t _ = 0; _ < 4u; ++_) {
            grid.emplace_back(i);
        }
    }

    auto [cb, out] = ta.propagate_grid(grid);

    REQUIRE(!cb);
    REQUIRE(out.size() == 202u * 4u);
    REQUIRE(std::all_of(out.begin() + 1, out.end(), [](const auto &val) { return val != 0; }));

    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(counter[i] == 100u);
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
    }

    t_ev_t ev1(v);

    ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.02, 0.03, .25, .26, .27, .28}, 4, kw::t_events = {ev1}};

    std::tie(cb, out) = ta.propagate_grid(grid);

    REQUIRE(!cb);
    REQUIRE(std::all_of(out.begin() + 8, out.end(), [](const auto &val) { return std::isnan(val); }));

    for (std::uint32_t i = 0; i < 4u; ++i) {
        REQUIRE(static_cast<std::int64_t>(std::get<0>(ta.get_propagate_res()[i])) == -1);
    }
}

// Test for a bug in propagate_grid() in which the
// integration would stop at the first step, in case
// a terminal event triggers immediately.
TEST_CASE("te propagate_grid first step bug")
{
    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive_batch<double>::t_event_t;

    std::vector<double> grid;
    for (auto i = 0; i < 100; ++i) {
        for (std::uint32_t _ = 0; _ < 4u; ++_) {
            grid.emplace_back(5 / 100. * i);
        }
    }

    {
        t_ev_t ev(
            v, kw::callback = [](auto &, int, std::uint32_t) { return true; });

        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                {0.05, 0.051, 0.052, 0.053, 0.025, 0.0251, 0.0252, 0.0253},
                                                4,
                                                kw::t_events = {ev}};

        auto [cb, out] = ta.propagate_grid(grid);

        REQUIRE(!cb);
        REQUIRE(out.size() == 200u * 4u);
        REQUIRE(std::all_of(out.begin(), out.end(), [](const auto &val) { return val != 0; }));
    }

    {
        t_ev_t ev(v);

        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                {0.05, 0.051, 0.052, 0.053, 0.025, 0.0251, 0.0252, 0.0253},
                                                4,
                                                kw::t_events = {ev}};

        auto [cb, out] = ta.propagate_grid(grid);

        REQUIRE(!cb);
        REQUIRE(out.size() == 200u * 4u);
        REQUIRE(std::all_of(out.begin() + 32, out.end(), [](const auto &val) { return std::isnan(val); }));
    }
}

TEST_CASE("te damped pendulum")
{
    using t_ev_t = taylor_adaptive_batch<double>::t_event_t;

    auto [x, v] = make_vars("x", "v");

    std::vector<std::vector<double>> zero_vel_times(4u);

    auto callback = [&zero_vel_times](auto &ta, int, std::uint32_t idx) {
        const auto tm = ta.get_time()[idx];

        if (ta.get_pars()[idx] == 0) {
            ta.get_pars_data()[idx] = 1;
        } else {
            ta.get_pars_data()[idx] = 0;
        }

        zero_vel_times[idx].push_back(tm);

        return true;
    };

    t_ev_t ev(v, kw::callback = callback);

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x) - par[0] * v},
                                            {0.05, 0.051, 0.052, 0.053, 0.025, 0.0251, 0.0252, 0.0253},
                                            4,
                                            kw::t_events = {ev}};

    ta.propagate_until({100, 100, 100, 100});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(zero_vel_times[i].size() == 99u);
    }

    ta.step();

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(zero_vel_times[i].size() == 100u);
    }

    // Mix use of step() and propagate like in the tutorial.
    ta.set_time({0, 0, 0, 0});
    for (auto i = 0u; i < 4u; ++i) {
        zero_vel_times[i].clear();
        ta.get_state_data()[i] = 0.05 + i * 0.001;
        ta.get_state_data()[4u + i] = 0.025 + i * 0.0001;
    }

    do {
        ta.step();
    } while (std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                         [](const auto &t) { return std::get<0>(t) == taylor_outcome::success; }));

    ta.propagate_until({100, 100, 100, 100});

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(zero_vel_times[i].size() == 99u);
    }

    ta.step();

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(zero_vel_times[i].size() == 100u);
    }
}

TEST_CASE("te boolean callback")
{
    using std::abs;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    std::vector<unsigned> counter_t(4u, 0u);
    std::vector<fp_t> cur_time(4u);
    bool direction = true;

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::t_events = {t_ev_t(
            v, kw::callback = [&counter_t, &cur_time, &direction](auto &ta_, int, std::uint32_t idx) {
                const auto &t = ta_.get_time();

                if (direction) {
                    REQUIRE(t[idx] > cur_time[idx]);
                } else {
                    REQUIRE(t[idx] < cur_time[idx]);
                }

                const auto vel = ta_.get_state()[4u + idx];
                REQUIRE(abs(vel) < std::numeric_limits<fp_t>::epsilon() * 100);

                ++counter_t[idx];

                cur_time[idx] = t[idx];

                return counter_t[idx] != 5u;
            })}};

    // First we integrate up to the first
    // occurrence of the terminal event.
    while (true) {
        ta.step();

        if (std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; })) {
            break;
        }
    }

    // Then we propagate for an amount of time large enough
    // to trigger the stopping terminal event.
    ta.propagate_until({1000., 1000., 1000., 1000.});

    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{-1}; }));

    // Reset counter_t and invert direction.
    std::fill(counter_t.begin(), counter_t.end(), 0u);
    direction = false;

    while (true) {
        ta.step_backward();

        if (std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{0}; })) {
            break;
        }
    }

    ta.propagate_until({-1000., -1000., -1000., -1000.});

    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{-1}; }));
}

// Test a terminal event exactly at the end of a timestep.
TEST_CASE("te step end")
{
    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<double>::t_event_t;

    std::vector<unsigned> counter(4u);

    auto ta = taylor_adaptive_batch<double>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::t_events = {t_ev_t(
            heyoka::time - 1., kw::callback = [&counter](auto &ta_, int, std::uint32_t idx) {
                ++counter[idx];
                REQUIRE(ta_.get_time()[idx] == 1.);
                return true;
            })}};

    ta.propagate_until({10., 10., 10., 10.}, kw::max_delta_t = {0.005, 0.005, 0.005, 0.005});

    REQUIRE(std::all_of(counter.begin(), counter.end(), [](auto c) { return c == 1u; }));
}

// Bug: mr always being true for an
// event with zero cooldown.
TEST_CASE("te zero cd mr bug")
{
    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive_batch<double>::t_event_t;

    auto ta = taylor_adaptive_batch<double>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)},
        {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
        4,
        kw::t_events = {
            t_ev_t(
                v, kw::callback = [](auto &, int, std::uint32_t) { return false; }, kw::cooldown = 0),
        }};

    ta.propagate_until({10., 10., 10., 10.});

    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{-1}; }));
}

struct s11n_te_callback {
    template <typename I>
    bool operator()(I &, int, std::uint32_t) const
    {
        return true;
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(s11n_te_callback, bool, taylor_adaptive_batch<double> &, int, std::uint32_t)

TEST_CASE("te s11n")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    t_event_batch<fp_t> ev(v, kw::callback = s11n_te_callback{}, kw::direction = event_direction::positive,
                           kw::cooldown = fp_t(100));

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ev;
    }

    ev = t_event_batch<fp_t>(v + x);

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ev;
    }

    REQUIRE(ev.get_expression() == v);
    REQUIRE(ev.get_direction() == event_direction::positive);
    REQUIRE(ev.get_callback().get_type_index() == typeid(s11n_te_callback));
    REQUIRE(ev.get_cooldown() == fp_t(100));
}

TEST_CASE("te def ctor")
{
    t_event_batch<double> te;

    REQUIRE(te.get_expression() == 0_dbl);
    REQUIRE(!te.get_callback());
    REQUIRE(te.get_direction() == event_direction::any);
    REQUIRE(te.get_cooldown() == -1.);
}
