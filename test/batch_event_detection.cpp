// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <tuple>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("single step nte")
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
        nt_batch_event<fp_t> ev_batch{
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

TEST_CASE("single step te")
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
                             [&trig_times, &v_vals](auto &tint, bool, int) {
                                 trig_times.push_back(tint.get_time());
                                 v_vals.push_back(tint.get_state()[1]);
                                 return true;
                             },
                         kw::direction = event_direction::negative};
        std::vector<std::vector<fp_t>> trig_times_batch(batch_size), v_vals_batch(batch_size);
        t_batch_event<fp_t> ev_batch{
            ev_eq,
            kw::callback =
                [&trig_times_batch, batch_size, &v_vals_batch](auto &tint, bool, int, std::uint32_t batch_idx) {
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
TEST_CASE("linear box")
{
    using ev_t = taylor_adaptive_batch<double>::nt_event_t;

    auto [x] = make_vars("x");

    auto counter = 0u;

    auto ta_ev = taylor_adaptive_batch<double>{
        {prime(x) = par[0]},
        {0., 0., 0., 0.},
        4,
        kw::nt_events = {ev_t(x - 1.,
                              [&counter](taylor_adaptive_batch<double> &tint, double tm, int, std::uint32_t batch_idx) {
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

TEST_CASE("glancing blow test")
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
        kw::nt_events
        = {ev_t(square(x0 - x1) + square(y0 - y1) - 4., [&counter](auto &, fp_t t, int, std::uint32_t batch_idx) {
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
                                     kw::nt_events = {ev_t(square(x0 - x1) + square(y0 - y1) - 4.,
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
                              [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t > cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta.get_time()[batch_idx] > t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta.update_d_output({t, t, t, t});

                                  const auto v = ta.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t > cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta.get_time()[batch_idx] > t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta.update_d_output({t, t, t, t});

                             const auto v = ta.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(v) <= std::numeric_limits<fp_t>::epsilon() * 100);

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
                              [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t > cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta.get_time()[batch_idx] > t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta.update_d_output({t, t, t, t});

                                  const auto v = ta.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t > cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta.get_time()[batch_idx] > t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta.update_d_output({t, t, t, t});

                             const auto v = ta.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(v) <= std::numeric_limits<fp_t>::epsilon() * 100);

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
                                             [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                                 using std::abs;

                                                 // Make sure the callbacks are called in order.
                                                 REQUIRE(t > cur_time[batch_idx]);

                                                 // Ensure the state of ta has
                                                 // been propagated until after the
                                                 // event.
                                                 REQUIRE(ta.get_time()[batch_idx] > t);

                                                 REQUIRE((counter[batch_idx] == 0u
                                                          || (counter[batch_idx] >= 2u && counter[batch_idx] <= 6u)
                                                          || (counter[batch_idx] >= 7u && counter[batch_idx] <= 9u)));

                                                 ta.update_d_output({t, t, t, t});

                                                 const auto v = ta.get_d_output()[4u + batch_idx];
                                                 REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                                 ++counter[batch_idx];

                                                 cur_time[batch_idx] = t;
                                             }),
                                        ev_t(
                                            v,
                                            [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                                using std::abs;

                                                // Make sure the callbacks are called in order.
                                                REQUIRE(t > cur_time[batch_idx]);

                                                // Ensure the state of ta has
                                                // been propagated until after the
                                                // event.
                                                REQUIRE(ta.get_time()[batch_idx] > t);

                                                REQUIRE((counter[batch_idx] == 1u || counter[batch_idx] == 6u));

                                                ta.update_d_output({t, t, t, t});

                                                const auto v = ta.get_d_output()[4u + batch_idx];
                                                REQUIRE(abs(v) <= std::numeric_limits<fp_t>::epsilon() * 100);

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
                                             [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                                 using std::abs;

                                                 // Make sure the callbacks are called in order.
                                                 REQUIRE(t > cur_time[batch_idx]);

                                                 // Ensure the state of ta has
                                                 // been propagated until after the
                                                 // event.
                                                 REQUIRE(ta.get_time()[batch_idx] > t);

                                                 REQUIRE((counter[batch_idx] == 0u
                                                          || (counter[batch_idx] >= 2u && counter[batch_idx] <= 6u)
                                                          || (counter[batch_idx] >= 7u && counter[batch_idx] <= 9u)));

                                                 ta.update_d_output({t, t, t, t});

                                                 const auto v = ta.get_d_output()[4u + batch_idx];
                                                 REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                                 ++counter[batch_idx];

                                                 cur_time[batch_idx] = t;
                                             }),
                                        ev_t(
                                            v,
                                            [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                                using std::abs;

                                                // Make sure the callbacks are called in order.
                                                REQUIRE(t > cur_time[batch_idx]);

                                                // Ensure the state of ta has
                                                // been propagated until after the
                                                // event.
                                                REQUIRE(ta.get_time()[batch_idx] > t);

                                                REQUIRE((counter[batch_idx] == 1u || counter[batch_idx] == 6u));

                                                ta.update_d_output({t, t, t, t});

                                                const auto v = ta.get_d_output()[4u + batch_idx];
                                                REQUIRE(abs(v) <= std::numeric_limits<fp_t>::epsilon() * 100);

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
                              [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                                  using std::abs;

                                  // Make sure the callbacks are called in order.
                                  REQUIRE(t < cur_time[batch_idx]);

                                  // Ensure the state of ta has
                                  // been propagated until after the
                                  // event.
                                  REQUIRE(ta.get_time()[batch_idx] < t);

                                  REQUIRE((counter[batch_idx] % 3u == 0u || counter[batch_idx] % 3u == 2u));

                                  ta.update_d_output({t, t, t, t});

                                  const auto v = ta.get_d_output()[4u + batch_idx];
                                  REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                  ++counter[batch_idx];

                                  cur_time[batch_idx] = t;
                              }),
                         ev_t(v, [&counter, &cur_time](auto &ta, fp_t t, int, std::uint32_t batch_idx) {
                             using std::abs;

                             // Make sure the callbacks are called in order.
                             REQUIRE(t < cur_time[batch_idx]);

                             // Ensure the state of ta has
                             // been propagated until after the
                             // event.
                             REQUIRE(ta.get_time()[batch_idx] < t);

                             REQUIRE((counter[batch_idx] % 3u == 1u));

                             ta.update_d_output({t, t, t, t});

                             const auto v = ta.get_d_output()[4u + batch_idx];
                             REQUIRE(abs(v) <= std::numeric_limits<fp_t>::epsilon() * 100);

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

TEST_CASE("nte dir test")
{
    auto [x, v] = make_vars("x", "v");

    bool fwd = true;

    std::vector<std::vector<double>> tlist(4u);

    std::vector<std::vector<double>::reverse_iterator> rit(4u);

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {-0.25, -0.26, -0.27, -0.28, 0., 0., 0., 0.},
                                            4,
                                            kw::nt_events = {nt_batch_event<double>(
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
    nt_batch_event<double> nte;

    REQUIRE(nte.get_expression() == 0_dbl);
    REQUIRE(nte.get_callback());
    REQUIRE(nte.get_direction() == event_direction::any);
}
