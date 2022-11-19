// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>

#include <fmt/format.h>

#include <mp++/real.hpp>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/real_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "heyoka/kw.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("poly translator 1")
{
    using fp_t = mppp::real;

    auto poly_eval5 = [](const auto &a, const auto &x) {
        return ((((a[5] * x + a[4]) * x + a[3]) * x + a[2]) * x + a[1]) * x + a[0];
    };

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            std::vector<fp_t> output, input;
            output.resize(6u, fp_t{0, prec});
            input.resize(6u, fp_t{0, prec});

            for (auto i = 0u; i < 6u; ++i) {
                mppp::set(input[i], i + 1u);
            }

            llvm_state s{kw::opt_level = opt_level};

            detail::add_poly_translator_1(s, detail::llvm_type_like(s, input[0]), 5, 1);

            s.optimise();

            s.compile();

            auto *pt1 = reinterpret_cast<void (*)(fp_t *, const fp_t *)>(s.jit_lookup("poly_translate_1"));

            pt1(output.data(), input.data());

            REQUIRE(poly_eval5(output, fp_t{"1.1", prec})
                    == approximately(poly_eval5(input, fp_t{"1.1", prec} + fp_t{1, prec})));
        }
    }
}

TEST_CASE("poly csc")
{
    using fp_t = mppp::real;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            std::vector<fp_t> input;
            input.resize(6u, fp_t{0, prec});

            for (auto i = 0u; i < 6u; ++i) {
                mppp::set(input[i], i + 1u);
            }

            llvm_state s{kw::opt_level = opt_level};

            const auto mname = detail::llvm_mangle_type(detail::llvm_type_like(s, input[0]));

            detail::llvm_add_csc(s, detail::llvm_type_like(s, input[0]), 5, 1);

            s.optimise();

            s.compile();

            auto *pt1 = reinterpret_cast<void (*)(std::uint32_t *, const fp_t *)>(
                s.jit_lookup(fmt::format("heyoka_csc_degree_5_{}", mname)));

            std::uint32_t out = 1;
            pt1(&out, input.data());

            REQUIRE(out == 0u);

            mppp::set(input[0], -1);

            pt1(&out, input.data());

            REQUIRE(out == 1u);

            mppp::set(input[1], -2);
            mppp::set(input[3], -1);

            pt1(&out, input.data());

            REQUIRE(out == 3u);
        }
    }
}

// Simple test for the construction of an integrator with events.
TEST_CASE("event construction")
{
    using fp_t = mppp::real;

    using nt_ev_t = nt_event<fp_t>;
    using t_ev_t = t_event<fp_t>;

    auto [x] = make_vars("x");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta_ev = taylor_adaptive<fp_t>{
                {prime(x) = 1_dbl},
                {fp_t{0., prec}},
                kw::opt_level = opt_level,
                kw::t_events = {t_ev_t(
                    x - 1., kw::callback = [](const taylor_adaptive<fp_t> &, bool, int) { return true; })},
                kw::nt_events = {nt_ev_t(x - 1., [](taylor_adaptive<fp_t> &, const fp_t &, int) {})}};
        }
    }
}

// A test case to check that the propagation codepath
// with events produces results identical to the no-events codepath.
TEST_CASE("taylor nte match")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta_ev
                = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                        {fp_t(-0.25, prec), fp_t(0., prec)},
                                        kw::opt_level = opt_level,
                                        kw::compact_mode = true,
                                        kw::nt_events = {ev_t(v, [](taylor_adaptive<fp_t> &, const fp_t &, int) {})}};

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(-0.25, prec), fp_t(0., prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            for (auto i = 0; i < 200; ++i) {
                auto [oc_ev, h_ev] = ta_ev.step();
                auto [oc, h] = ta.step();

                REQUIRE(oc_ev == oc);
                REQUIRE(h_ev == h);

                REQUIRE(ta_ev.get_state()[0] == ta.get_state()[0]);
                REQUIRE(ta_ev.get_state()[1] == ta.get_state()[1]);
                REQUIRE(ta_ev.get_time() == ta.get_time());
            }
        }
    }
}

TEST_CASE("taylor nte glancing blow test")
{
    std::cout << "Starting glancing blow test...\n";

    // NOTE: in this test two spherical particles are
    // "colliding" in a glancing fashion, meaning that
    // the polynomial representing the evolution in time
    // of the mutual distance has a repeated root at
    // t = collision time.

    auto [x0, vx0, x1, vx1] = make_vars("x0", "vx0", "x1", "vx1");
    auto [y0, vy0, y1, vy1] = make_vars("y0", "vy0", "y1", "vy1");

    using fp_t = mppp::real;

    using ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            // First setup: one particle still, the other moving with uniform velocity.
            auto ta = taylor_adaptive<fp_t>{
                {prime(x0) = vx0, prime(y0) = vy0, prime(x1) = vx1, prime(y1) = vy1, prime(vx0) = 0_dbl,
                 prime(vy0) = 0_dbl, prime(vx1) = 0_dbl, prime(vy1) = 0_dbl},
                {fp_t(0., prec), fp_t(0., prec), fp_t(-10., prec), fp_t(2, prec), fp_t(0., prec), fp_t(0., prec),
                 fp_t(1., prec), fp_t(0., prec)},
                kw::nt_events = {ev_t(square(x0 - x1) + square(y0 - y1) - 4.,
                                      [&counter, prec](taylor_adaptive<fp_t> &, fp_t t, int) {
                                          REQUIRE((t - 10.) * (t - 10.) <= detail::eps_from_prec(prec));

                                          ++counter;
                                      })},
                kw::opt_level = opt_level,
                kw::compact_mode = true};

            for (auto i = 0; i < 20; ++i) {
                REQUIRE(std::get<0>(ta.step(fp_t(1.3, prec))) == taylor_outcome::time_limit);
            }

            // Any number of events up to 2 is acceptable here.
            REQUIRE(counter <= 2u);

            counter = 0;

            // Second setup: one particle still, the other accelerating towards positive
            // x direction with constant acceleration.
            ta = taylor_adaptive<fp_t>{
                {prime(x0) = vx0, prime(y0) = vy0, prime(x1) = vx1, prime(y1) = vy1, prime(vx0) = 0_dbl,
                 prime(vy0) = 0_dbl, prime(vx1) = .1_dbl, prime(vy1) = 0_dbl},
                {fp_t(0., prec), fp_t(0., prec), fp_t(-10., prec), fp_t(2, prec), fp_t(0., prec), fp_t(0., prec),
                 fp_t(1., prec), fp_t(0., prec)},
                kw::nt_events = {ev_t(square(x0 - x1) + square(y0 - y1) - 4.,
                                      [&counter](taylor_adaptive<fp_t> &, const fp_t &, int) { ++counter; })},
                kw::opt_level = opt_level,
                kw::compact_mode = true};

            for (auto i = 0; i < 20; ++i) {
                REQUIRE(std::get<0>(ta.step(fp_t(1.3, prec))) == taylor_outcome::time_limit);
            }

            REQUIRE(counter <= 2u);
        }
    }

    std::cout << "Glancing blow test finished\n";
}

TEST_CASE("taylor nte multizero")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            // In this test, we define two events:
            // - the velocity is smaller in absolute
            //   value than a small limit,
            // - the velocity is exactly zero.
            // It is likely that both events are going to fire
            // in the same timestep, with the first event
            // firing twice. The sequence of events must
            // be 0 1 0 repeated a few times.

            fp_t cur_time(0, prec);

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0, prec), fp_t(.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::nt_events
                                            = {ev_t(v * v - 1e-10,
                                                    [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                        using std::abs;

                                                        // Make sure the callbacks are called in order.
                                                        REQUIRE(t > cur_time);

                                                        // Ensure the state of ta has
                                                        // been propagated until after the
                                                        // event.
                                                        REQUIRE(ta_.get_time() > t);

                                                        REQUIRE((counter % 3u == 0u || counter % 3u == 2u));

                                                        ta_.update_d_output(t);

                                                        const auto vel = ta_.get_d_output()[1];
                                                        REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                        ++counter;

                                                        cur_time = t;
                                                    }),
                                               ev_t(v, [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                   using std::abs;

                                                   // Make sure the callbacks are called in order.
                                                   REQUIRE(t > cur_time);

                                                   // Ensure the state of ta has
                                                   // been propagated until after the
                                                   // event.
                                                   REQUIRE(ta_.get_time() > t);

                                                   REQUIRE((counter % 3u == 1u));

                                                   ta_.update_d_output(t);

                                                   const auto vel = ta_.get_d_output()[1];
                                                   REQUIRE(abs(vel) <= detail::eps_from_prec(prec) * 100);

                                                   ++counter;

                                                   cur_time = t;
                                               })}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4, prec))) == taylor_outcome::time_limit);

            REQUIRE(counter == 12u);

            counter = 0;
            cur_time = 0;

            // Run the same test with sub-eps tolerance too.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0, prec), fp_t(.25, prec)},
                                       kw::tol = detail::eps_from_prec(prec) / fp_t(100, prec),
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::nt_events
                                       = {ev_t(v * v - 1e-10,
                                               [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                   using std::abs;

                                                   // Make sure the callbacks are called in order.
                                                   REQUIRE(t > cur_time);

                                                   // Ensure the state of ta has
                                                   // been propagated until after the
                                                   // event.
                                                   REQUIRE(ta_.get_time() > t);

                                                   REQUIRE((counter % 3u == 0u || counter % 3u == 2u));

                                                   ta_.update_d_output(t);

                                                   const auto vel = ta_.get_d_output()[1];
                                                   REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                   ++counter;

                                                   cur_time = t;
                                               }),
                                          ev_t(v, [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                              using std::abs;

                                              // Make sure the callbacks are called in order.
                                              REQUIRE(t > cur_time);

                                              // Ensure the state of ta has
                                              // been propagated until after the
                                              // event.
                                              REQUIRE(ta_.get_time() > t);

                                              REQUIRE((counter % 3u == 1u));

                                              ta_.update_d_output(t);

                                              const auto vel = ta_.get_d_output()[1];
                                              REQUIRE(abs(vel) <= detail::eps_from_prec(prec) * 100);

                                              ++counter;

                                              cur_time = t;
                                          })}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4, prec))) == taylor_outcome::time_limit);

            REQUIRE(counter == 12u);

            counter = 0;
            cur_time = 0;

            // We re-run the test, but this time we want to detect
            // only when the velocity goes from positive to negative.
            // Thus the sequence of events will be:
            // - 0 1 0
            // - 0 0
            // - 0 1 0
            // - 0 0
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0, prec), fp_t(.25, prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::nt_events
                                       = {ev_t(v * v - 1e-10,
                                               [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                   using std::abs;

                                                   // Make sure the callbacks are called in order.
                                                   REQUIRE(t > cur_time);

                                                   // Ensure the state of ta has
                                                   // been propagated until after the
                                                   // event.
                                                   REQUIRE(ta_.get_time() > t);

                                                   REQUIRE((counter == 0u || (counter >= 2u && counter <= 6u)
                                                            || (counter >= 7u && counter <= 9u)));

                                                   ta_.update_d_output(t);

                                                   const auto vel = ta_.get_d_output()[1];
                                                   REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                   ++counter;

                                                   cur_time = t;
                                               }),
                                          ev_t(
                                              v,

                                              [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int d_sgn) {
                                                  using std::abs;

                                                  REQUIRE(d_sgn == -1);

                                                  // Make sure the callbacks are called in order.
                                                  REQUIRE(t > cur_time);

                                                  // Ensure the state of ta has
                                                  // been propagated until after the
                                                  // event.
                                                  REQUIRE(ta_.get_time() > t);

                                                  REQUIRE((counter == 1u || counter == 6u));

                                                  ta_.update_d_output(t);

                                                  const auto vel = ta_.get_d_output()[1];
                                                  REQUIRE(abs(vel) <= detail::eps_from_prec(prec) * 100);

                                                  ++counter;

                                                  cur_time = t;
                                              },
                                              kw::direction = event_direction::negative)}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4, prec))) == taylor_outcome::time_limit);

            REQUIRE(counter == 10u);

            counter = 0;
            cur_time = 0;

            // Sub-eps tolerance too.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0, prec), fp_t(.25, prec)},
                                       kw::tol = detail::eps_from_prec(prec) / fp_t(100, prec),
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::nt_events
                                       = {ev_t(v * v - 1e-10,
                                               [&](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
                                                   using std::abs;

                                                   // Make sure the callbacks are called in order.
                                                   REQUIRE(t > cur_time);

                                                   // Ensure the state of ta has
                                                   // been propagated until after the
                                                   // event.
                                                   REQUIRE(ta_.get_time() > t);

                                                   REQUIRE((counter == 0u || (counter >= 2u && counter <= 6u)
                                                            || (counter >= 7u && counter <= 9u)));

                                                   ta_.update_d_output(t);

                                                   const auto vel = ta_.get_d_output()[1];
                                                   REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                   ++counter;

                                                   cur_time = t;
                                               }),
                                          ev_t(
                                              v,

                                              [&](taylor_adaptive<fp_t> &ta_, fp_t t, int d_sgn) {
                                                  using std::abs;

                                                  REQUIRE(d_sgn == -1);

                                                  // Make sure the callbacks are called in order.
                                                  REQUIRE(t > cur_time);

                                                  // Ensure the state of ta has
                                                  // been propagated until after the
                                                  // event.
                                                  REQUIRE(ta_.get_time() > t);

                                                  REQUIRE((counter == 1u || counter == 6u));

                                                  ta_.update_d_output(t);

                                                  const auto vel = ta_.get_d_output()[1];
                                                  REQUIRE(abs(vel) <= detail::eps_from_prec(prec) * 100);

                                                  ++counter;

                                                  cur_time = t;
                                              },
                                              kw::direction = event_direction::negative)}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4, prec))) == taylor_outcome::time_limit);

            REQUIRE(counter == 10u);
        }
    }
}

TEST_CASE("taylor nte multizero negative timestep")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            fp_t cur_time(0);

            // In this test, we define two events:
            // - the velocity is smaller in absolute
            //   value than a small limit,
            // - the velocity is exactly zero.
            // It is likely that both events are going to fire
            // in the same timestep, with the first event
            // firing twice. The sequence of events must
            // be 0 1 0 repeated a few times.
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0, prec), fp_t(.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::nt_events
                                            = {ev_t(v * v - 1e-10,
                                                    [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                        using std::abs;

                                                        // Make sure the callbacks are called in order.
                                                        REQUIRE(t < cur_time);

                                                        // Ensure the state of ta has
                                                        // been propagated until after the
                                                        // event.
                                                        REQUIRE(ta_.get_time() < t);

                                                        REQUIRE((counter % 3u == 0u || counter % 3u == 2u));

                                                        ta_.update_d_output(t);

                                                        const auto vel = ta_.get_d_output()[1];
                                                        REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                        ++counter;

                                                        cur_time = t;
                                                    }),
                                               ev_t(v, [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                   using std::abs;

                                                   // Make sure the callbacks are called in order.
                                                   REQUIRE(t < cur_time);

                                                   // Ensure the state of ta has
                                                   // been propagated until after the
                                                   // event.
                                                   REQUIRE(ta_.get_time() < t);

                                                   REQUIRE((counter % 3u == 1u));

                                                   ta_.update_d_output(t);

                                                   const auto vel = ta_.get_d_output()[1];
                                                   REQUIRE(abs(vel) <= detail::eps_from_prec(prec) * 100);

                                                   ++counter;

                                                   cur_time = t;
                                               })}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(-4, prec))) == taylor_outcome::time_limit);

            REQUIRE(counter == 12u);
        }
    }
}

// Test for an event triggering exactly at the end of a timestep.
TEST_CASE("nte linear box")
{
    using fp_t = mppp::real;

    using ev_t = taylor_adaptive<mppp::real>::nt_event_t;

    auto [x] = make_vars("x");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            auto ta_ev
                = taylor_adaptive<mppp::real>{{prime(x) = 1_dbl},
                                              {fp_t(0., prec)},
                                              kw::nt_events = {ev_t(x - 1.,
                                                                    [&](auto &, const auto &tm, int) {
                                                                        REQUIRE(tm == approximately(fp_t(1., prec)));
                                                                        ++counter;
                                                                    })},
                                              kw::opt_level = opt_level,
                                              kw::compact_mode = true};

            // Check that the event triggers at the beginning of the second step.
            auto [oc, h] = ta_ev.step(fp_t(1., prec));

            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(h == 1.);
            REQUIRE(counter == 0u);
            REQUIRE(ta_ev.get_state()[0] == 1.);

            std::tie(oc, h) = ta_ev.step(fp_t(1., prec));
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(h == 1.);
            REQUIRE(counter == 1u);
            REQUIRE(ta_ev.get_state()[0] == 2.);
        }
    }
}

// Test for an event triggering exactly at the end of a timestep.
TEST_CASE("te linear box")
{
    using fp_t = mppp::real;

    using ev_t = taylor_adaptive<fp_t>::t_event_t;

    auto [x] = make_vars("x");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            auto ta_ev = taylor_adaptive<fp_t>{{prime(x) = 1_dbl},
                                               {fp_t(0., prec)},
                                               kw::t_events = {ev_t(
                                                   x - 1., kw::callback =
                                                               [&counter](taylor_adaptive<fp_t> &, bool, int) {
                                                                   ++counter;
                                                                   return true;
                                                               })},
                                               kw::opt_level = opt_level,
                                               kw::compact_mode = true};

            // Check that the event triggers at the beginning of the second step.
            auto [oc, h] = ta_ev.step(fp_t(1., prec));

            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(h == 1.);
            REQUIRE(counter == 0u);
            REQUIRE(ta_ev.get_state()[0] == 1.);

            std::tie(oc, h) = ta_ev.step(fp_t(1., prec));
            REQUIRE(oc == taylor_outcome{0});
            REQUIRE(h == approximately(fp_t(0., prec)));
            REQUIRE(counter == 1u);
            REQUIRE(ta_ev.get_state()[0] == 1.);
        }
    }
}

TEST_CASE("taylor te basic")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;
    using nt_ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {

            // NOTE: test also sub-eps tolerance.
            for (auto cur_tol : {detail::eps_from_prec(prec), detail::eps_from_prec(prec) / fp_t(100, prec)}) {
                auto counter_nt = 0u, counter_t = 0u;
                fp_t cur_time(0, prec);
                bool direction = true;

                auto ta = taylor_adaptive<fp_t>{
                    {prime(x) = v, prime(v) = -9.8 * sin(x)},
                    {fp_t(0., prec), fp_t(0.25, prec)},
                    kw::tol = cur_tol,
                    kw::opt_level = opt_level,
                    kw::compact_mode = true,
                    kw::nt_events = {nt_ev_t(v * v - 1e-10,
                                             [&](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
                                                 // Make sure the callbacks are called in order.
                                                 if (direction) {
                                                     REQUIRE(t > cur_time);
                                                 } else {
                                                     REQUIRE(t < cur_time);
                                                 }

                                                 ta_.update_d_output(t);

                                                 const auto vel = ta_.get_d_output()[1];
                                                 REQUIRE(abs(vel * vel - 1e-10) < detail::eps_from_prec(prec));

                                                 ++counter_nt;

                                                 cur_time = t;
                                             })},
                    kw::t_events = {t_ev_t(
                        v, kw::callback = [&](taylor_adaptive<fp_t> &ta_, bool mr, int) {
                            const auto t = ta_.get_time();

                            REQUIRE(!mr);

                            if (direction) {
                                REQUIRE(t > cur_time);
                            } else {
                                REQUIRE(t < cur_time);
                            }

                            const auto vel = ta_.get_state()[1];
                            REQUIRE(abs(vel) < detail::eps_from_prec(prec) * fp_t(100, prec));

                            ++counter_t;

                            cur_time = t;

                            return true;
                        })}};

                taylor_outcome oc;
                while (true) {
                    oc = std::get<0>(ta.step());
                    REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                    if (oc > taylor_outcome::success) {
                        break;
                    }
                    REQUIRE(oc == taylor_outcome::success);
                }

                REQUIRE(static_cast<std::int64_t>(oc) == 0);
                REQUIRE(ta.get_time() < 1);
                REQUIRE(counter_nt == 1u);
                REQUIRE(counter_t == 1u);

                while (true) {
                    oc = std::get<0>(ta.step());
                    REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                    if (oc > taylor_outcome::success) {
                        break;
                    }
                    REQUIRE(oc == taylor_outcome::success);
                }

                REQUIRE(static_cast<std::int64_t>(oc) == 0);
                REQUIRE(ta.get_time() > 1);
                REQUIRE(counter_nt == 3u);
                REQUIRE(counter_t == 2u);

                // Move backwards.
                direction = false;

                while (true) {
                    oc = std::get<0>(ta.step_backward());
                    REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                    if (oc > taylor_outcome::success) {
                        break;
                    }
                    REQUIRE(oc == taylor_outcome::success);
                }

                REQUIRE(static_cast<std::int64_t>(oc) == 0);
                REQUIRE(ta.get_time() < 1);
                REQUIRE(counter_nt == 5u);
                REQUIRE(counter_t == 3u);

                while (true) {
                    oc = std::get<0>(ta.step_backward());
                    REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                    if (oc > taylor_outcome::success) {
                        break;
                    }
                    REQUIRE(oc == taylor_outcome::success);
                }

                REQUIRE(static_cast<std::int64_t>(oc) == 0);
                REQUIRE(ta.get_time() < 0);
                REQUIRE(counter_nt == 7u);
                REQUIRE(counter_t == 4u);
            }
        }
    }
}

TEST_CASE("taylor te identical")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {

            t_ev_t ev(v);

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0., prec), fp_t(0.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev, ev}};

            taylor_outcome oc;
            while (true) {
                oc = std::get<0>(ta.step());
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            // One of the two events must always be detected.
            auto first_ev = -static_cast<std::int64_t>(oc) - 1;
            auto time = ta.get_time();
            REQUIRE((first_ev == 0 || first_ev == 1));

            // Taking a further step, we might either detect the second event,
            // or it may end up being ignored due to numerics.
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                auto second_ev = -static_cast<std::int64_t>(oc) - 1;
                REQUIRE(time == approximately(ta.get_time()));
                REQUIRE((second_ev == 0 || second_ev == 1));
                REQUIRE(second_ev != first_ev);

                // Both events should be in cooldown: taking a step
                // backwards should not run into another event.
                oc = std::get<0>(ta.step_backward());
                REQUIRE(oc == taylor_outcome::success);
            } else {
                REQUIRE(oc == taylor_outcome::success);
            }
        }
    }
}

TEST_CASE("taylor te close")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            t_ev_t ev1(x);
            t_ev_t ev2(
                x - detail::eps_from_prec(prec) * fp_t(2, prec),
                kw::callback = [](taylor_adaptive<fp_t> &, bool mr, int) {
                    REQUIRE(!mr);
                    return true;
                });

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0.1, prec), fp_t(0.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev1, ev2}};

            taylor_outcome oc;
            while (true) {
                oc = std::get<0>(ta.step());
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            // The second event must have triggered first.
            REQUIRE(static_cast<std::int64_t>(oc) == 1);

            // Next step the first event must trigger.
            oc = std::get<0>(ta.step());
            REQUIRE(static_cast<std::int64_t>(oc) == -1);

            // Next step no event must trigger: event 0 is now on cooldown
            // as it just happened, and event 1 is still close enough to be
            // on cooldown too.
            oc = std::get<0>(ta.step());
            REQUIRE(oc == taylor_outcome::success);

            // Go back.
            while (true) {
                oc = std::get<0>(ta.step_backward());
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            REQUIRE(static_cast<std::int64_t>(oc) == -1);

            oc = std::get<0>(ta.step_backward());
            REQUIRE(static_cast<std::int64_t>(oc) == 1);

            // Taking the step forward will skip event zero as it is still
            // on cooldown.
            oc = std::get<0>(ta.step());
            REQUIRE(oc == taylor_outcome::success);
        }
    }
}

TEST_CASE("taylor te retrigger")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            t_ev_t ev(x - (1 - detail::eps_from_prec(prec) * fp_t(6, prec)));

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(1, prec), fp_t(0, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev}};

            // First timestep triggers the event immediately.
            taylor_outcome oc;
            oc = std::get<0>(ta.step());
            REQUIRE(static_cast<std::int64_t>(oc) == -1);
            REQUIRE(ta.get_time() != 0);

            // Step until re-trigger.
            while (true) {
                oc = std::get<0>(ta.step());
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }
            REQUIRE(static_cast<std::int64_t>(oc) == -1);

            auto tm = ta.get_time();
            oc = std::get<0>(ta.step());

            REQUIRE(static_cast<std::int64_t>(oc) == -1);
            REQUIRE(ta.get_time() > tm);
        }
    }
}

TEST_CASE("taylor te custom cooldown")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            t_ev_t ev(
                v * v - detail::eps_from_prec(prec) * fp_t(4, prec),
                kw::callback =
                    [](taylor_adaptive<fp_t> &, bool mr, int) {
                        REQUIRE(mr);
                        return true;
                    },
                kw::cooldown = fp_t(1e-1));

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0, prec), fp_t(0.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev}};

            // Step until trigger.
            taylor_outcome oc;
            while (true) {
                oc = std::get<0>(ta.step());
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }
            REQUIRE(static_cast<std::int64_t>(oc) == 0);
        }
    }
}

TEST_CASE("taylor te propagate_for")
{
    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {

            auto counter = 0u;

            t_ev_t ev(
                v, kw::callback = [&counter](taylor_adaptive<fp_t> &, bool mr, int) {
                    ++counter;
                    REQUIRE(!mr);
                    return true;
                });

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0, prec), fp_t(0.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev}};

            auto oc = std::get<0>(ta.propagate_for(fp_t(100, prec)));
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(ta.get_time() == 100);

            REQUIRE(counter == 100u);

            t_ev_t ev1(v);

            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0, prec), fp_t(0.25, prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::t_events = {ev1}};

            oc = std::get<0>(ta.propagate_for(fp_t(100, prec)));
            REQUIRE(oc > taylor_outcome::success);
            REQUIRE(static_cast<std::int64_t>(oc) == -1);
            REQUIRE(ta.get_time() < 0.502);
        }
    }
}

TEST_CASE("taylor te propagate_grid")
{

    using std::abs;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            t_ev_t ev(
                v, kw::callback = [&counter](taylor_adaptive<fp_t> &, bool mr, int) {
                    ++counter;
                    REQUIRE(!mr);
                    return true;
                });

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0, prec), fp_t(0.25, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::t_events = {ev}};

            std::vector<fp_t> grid;
            for (auto i = 0; i < 101; ++i) {
                grid.emplace_back(i, prec);
            }

            taylor_outcome oc;
            {
                auto [oc_, _1, _2, _3, out] = ta.propagate_grid(grid);
                oc = oc_;
                REQUIRE(out.size() == 202u);
            }
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(ta.get_time() >= 100);

            REQUIRE(counter == 100u);

            t_ev_t ev1(v);

            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0, prec), fp_t(0.25, prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::t_events = {ev1}};

            {
                auto [oc_, _1, _2, _3, out] = ta.propagate_grid(grid);
                oc = oc_;
                REQUIRE(out.size() == 2u);
            }
            REQUIRE(oc > taylor_outcome::time_limit);
            REQUIRE(static_cast<std::int64_t>(oc) == -1);
            REQUIRE(ta.get_time() < 0.502);
        }
    }
}

// Test a terminal event exactly at the end of a timestep.
TEST_CASE("te step end")
{
    auto [x, v] = make_vars("x", "v");

    using fp_t = mppp::real;

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto counter = 0u;

            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0., prec), fp_t(0.25, prec)},
                kw::t_events = {t_ev_t(
                                    heyoka::time - 1., kw::callback =
                                                           [&counter](taylor_adaptive<fp_t> &ta_, bool, int) {
                                                               ++counter;
                                                               REQUIRE(ta_.get_time() == 1.);
                                                               return true;
                                                           }),
                               }, kw::opt_level = opt_level, kw::compact_mode = true};

            ta.propagate_until(fp_t(10., prec), kw::max_delta_t = fp_t(0.005, prec));

            REQUIRE(counter == 1u);
        }
    }
}

// Test to verify an event is not detected
// when it falls exactly at the end of a timestep.
TEST_CASE("te open range")
{
    using std::nextafter;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using ev_t = taylor_adaptive<fp_t>::t_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(-0.25, prec), fp_t(0., prec)},
                                            kw::t_events = {ev_t(heyoka::time - fp_t(97, prec) / fp_t(100000, prec))},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            auto [oc, h] = ta.step(fp_t(97, prec) / fp_t(100000, prec));

            REQUIRE(oc == taylor_outcome::time_limit);

            // Reset the integrator.
            ta.set_time(fp_t(0, prec));
            ta.get_state_data()[0] = fp_t(-0.25, prec);
            ta.get_state_data()[1] = fp_t(0, prec);
            ta.reset_cooldowns();

            // Integrate up to immediately after the event.
            std::tie(oc, h) = ta.step(nextafter(fp_t(97, prec) / fp_t(100000, prec), fp_t(1, prec)));

            REQUIRE(oc == taylor_outcome{-1});

            // Run also a test at the very beginning.
            ta.set_time(fp_t(97, prec) / fp_t(100000, prec));
            ta.get_state_data()[0] = fp_t(-0.25, prec);
            ta.get_state_data()[1] = fp_t(0, prec);
            ta.reset_cooldowns();

            std::tie(oc, h) = ta.step();

            REQUIRE(oc == taylor_outcome{-1});
            REQUIRE(h == 0);

            // And slightly later.
            ta.set_time(nextafter(fp_t(97, prec) / fp_t(100000, prec), fp_t(1)));
            ta.get_state_data()[0] = fp_t(-0.25, prec);
            ta.get_state_data()[1] = fp_t(0, prec);
            ta.reset_cooldowns();

            std::tie(oc, h) = ta.step();

            REQUIRE(oc == taylor_outcome::success);
            REQUIRE(h > 0);
        }
    }
}

// Test event callbacks that change the precision of the internal data.
TEST_CASE("events prec error")
{
    using Catch::Matchers::Message;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<fp_t>::t_event_t;
    using nt_ev_t = taylor_adaptive<fp_t>::nt_event_t;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            // First a couple of ntes.
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(-0.25, prec), fp_t(0., prec)},
                kw::nt_events
                = {nt_ev_t(x, [&](auto &tint, const auto &...) { tint.get_state_data()[0].prec_round(prec - 1); })},
                kw::opt_level = opt_level,
                kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            // Only the first nte changes the precision.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(-0.25, prec), fp_t(0., prec)},
                                       kw::nt_events = {nt_ev_t(x + 1e-5,
                                                                [&](auto &tint, const auto &...) {
                                                                    tint.get_state_data()[0].prec_round(prec - 1);
                                                                }),
                                                        nt_ev_t(x, [&](auto &, const auto &...) {})},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            // Terminal event.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(-0.25, prec), fp_t(0., prec)},
                                       kw::t_events = {t_ev_t(
                                           x, kw::callback =
                                                  [&](auto &tint, const auto &...) {
                                                      tint.get_state_data()[0].prec_round(prec - 1);
                                                      return true;
                                                  })},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            // Mix up te and nte.
            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(-0.25, prec), fp_t(0., prec)},
                kw::t_events = {t_ev_t(x - 0.25)},
                kw::nt_events
                = {nt_ev_t(x, [&](auto &tint, const auto &...) { tint.get_state_data()[0].prec_round(prec - 1); })},
                kw::opt_level = opt_level,
                kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(-0.25, prec), fp_t(0., prec)},
                kw::t_events = {t_ev_t(
                    x, kw::callback =
                           [&](auto &tint, const auto &...) {
                               tint.get_state_data()[0].prec_round(prec - 1);
                               return true;
                           })},
                kw::nt_events = {nt_ev_t(
                    x - 0.25, [&](auto &tint, const auto &...) { tint.get_state_data()[0].prec_round(prec - 1); })},
                kw::opt_level = opt_level,
                kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(fp_t(100, prec)), std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));
        }
    }
}
