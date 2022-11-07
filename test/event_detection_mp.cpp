// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <initializer_list>
#include <iostream>
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

    using ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

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

TEST_CASE("taylor glancing blow test")
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

    using ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

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

// TODO complete when we have propagate_*() support.
#if 0

TEST_CASE("taylor nte multizero")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    using ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

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

            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0, prec), fp_t(.25, prec)},
                kw::opt_level = opt_level,
                kw::compact_mode = true,
                kw::nt_events = {ev_t(v * v - 1e-10,
                                      [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, const fp_t &t, int) {
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
                                          REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                          ++counter;

                                          cur_time = t;
                                      }),
                                 ev_t(v, [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
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
                                     REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                     ++counter;

                                     cur_time = t;
                                 })}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4))) == taylor_outcome::time_limit);

            REQUIRE(counter == 12u);

            counter = 0;
            cur_time = 0;

            // Run the same test with sub-eps tolerance too.
            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0), fp_t(.25)},
                kw::tol = std::numeric_limits<fp_t>::epsilon() / 100,
                kw::opt_level = opt_level,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::nt_events = {ev_t(v * v - 1e-10,
                                      [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
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
                                          REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                          ++counter;

                                          cur_time = t;
                                      }),
                                 ev_t(v, [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
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
                                     REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                     ++counter;

                                     cur_time = t;
                                 })}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4))) == taylor_outcome::time_limit);

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
            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0), fp_t(.25)},
                kw::opt_level = opt_level,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::nt_events = {ev_t(v * v - 1e-10,
                                      [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
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
                                          REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                          ++counter;

                                          cur_time = t;
                                      }),
                                 ev_t(
                                     v,

                                     [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int d_sgn) {
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
                                         REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                         ++counter;

                                         cur_time = t;
                                     },
                                     kw::direction = event_direction::negative)}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4))) == taylor_outcome::time_limit);

            REQUIRE(counter == 10u);

            counter = 0;
            cur_time = 0;

            // Sub-eps tolerance too.
            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0), fp_t(.25)},
                kw::tol = std::numeric_limits<fp_t>::epsilon() / 100,
                kw::opt_level = opt_level,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::nt_events = {ev_t(v * v - 1e-10,
                                      [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
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
                                          REQUIRE(abs(vel * vel - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                          ++counter;

                                          cur_time = t;
                                      }),
                                 ev_t(
                                     v,

                                     [&counter, &cur_time](taylor_adaptive<fp_t> &ta_, fp_t t, int d_sgn) {
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
                                         REQUIRE(abs(vel) <= std::numeric_limits<fp_t>::epsilon() * 100);

                                         ++counter;

                                         cur_time = t;
                                     },
                                     kw::direction = event_direction::negative)}};

            REQUIRE(std::get<0>(ta.propagate_until(fp_t(4))) == taylor_outcome::time_limit);

            REQUIRE(counter == 10u);
        }
    }
}

#endif
