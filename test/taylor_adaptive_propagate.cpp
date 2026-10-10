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
#include <initializer_list>
#include <limits>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

// Repeated and non-monotonic grid points must be rejected regardless of the direction of the grid.
TEST_CASE("grid monotonicity")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

    for (const auto &grid : {std::vector{0., 0., 1.}, std::vector{0., 0., -1.}, std::vector{0., 1., 1.},
                             std::vector{0., -1., -1.}, std::vector{0., 1., 2., 2.}, std::vector{0., -1., -2., -2.},
                             std::vector{0., 1., 0.}, std::vector{0., -1., 0.}}) {
        REQUIRE_THROWS_MATCHES(
            ta.propagate_grid(grid), std::invalid_argument,
            Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));
    }

    // Valid grids in both directions.
    auto out = ta.propagate_grid({0., 1., 2.});
    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<5>(out).size() == 6u);

    out = ta.propagate_grid({2., 1., 0.});
    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<5>(out).size() == 6u);
}

// At the end of a successful propagation, the time coordinate must be exactly the final time.
TEST_CASE("snap")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

    // NOTE: forward and backward targets, the last one bringing the time back to zero.
    const auto targets = std::vector{1.1, 10.3, -0.7, -12.9, 3.3, 0.};

    for (const auto max_delta_t : {std::numeric_limits<double>::infinity(), 0.0123}) {
        // propagate_until().
        for (const auto t : targets) {
            const auto oc = std::get<0>(ta.propagate_until(t, kw::max_delta_t = max_delta_t));
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(ta.get_dtime().first == t);
            REQUIRE(ta.get_dtime().second == 0.);
        }

        // propagate_for(), back and forth from zero.
        for (const auto &[delta_t, t] :
             {std::pair{1.1, 1.1}, std::pair{-1.1, 0.}, std::pair{-12.9, -12.9}, std::pair{12.9, 0.}}) {
            const auto oc = std::get<0>(ta.propagate_for(delta_t, kw::max_delta_t = max_delta_t));
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(ta.get_dtime().first == t);
            REQUIRE(ta.get_dtime().second == 0.);
        }

        // propagate_grid(), forward and backward.
        for (const auto &grid :
             {std::vector{0., 0.3, 5.7, 10.3}, std::vector{10.3, 2.1, -0.7, -12.9}, std::vector{-12.9, -1.1, 0.}}) {
            const auto oc = std::get<0>(ta.propagate_grid(grid, kw::max_delta_t = max_delta_t));
            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(ta.get_dtime().first == grid.back());
            REQUIRE(ta.get_dtime().second == 0.);
        }
    }
}

// Propagating up to the current time results in a single zero-length step.
TEST_CASE("propagate to current time")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}};
    ta.set_time(1.5);

    const auto state = ta.get_state();

    auto n_cb_calls = 0;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    // propagate_until() and propagate_for().
    for (auto i = 0; i < 2; ++i) {
        n_cb_calls = 0;

        const auto res = (i == 0) ? ta.propagate_until(1.5, kw::c_output = true, kw::callback = cb)
                                  : ta.propagate_for(0., kw::c_output = true, kw::callback = cb);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        REQUIRE(std::get<1>(res) == std::numeric_limits<double>::infinity());
        REQUIRE(std::get<2>(res) == 0.);
        REQUIRE(std::get<3>(res) == 1u);
        REQUIRE(std::get<4>(res));
        REQUIRE(std::get<4>(res)->get_n_steps() == 1u);
        REQUIRE(std::get<4>(res)->get_bounds() == std::pair{1.5, 1.5});
        REQUIRE(n_cb_calls == 1);

        // NOTE: in the screen output, a continuous output object whose time range is zero is reported as forward.
        std::ostringstream oss;
        oss << *std::get<4>(res);
        REQUIRE(oss.str().find("Direction   : forward") != std::string::npos);
        REQUIRE(ta.get_dtime() == std::pair{1.5, 0.});
        REQUIRE(ta.get_state() == state);

        // Evaluate the continuous output at the current time and on both sides of it.
        //
        // NOTE: at the current time, the evaluation must return exactly the state. Elsewhere, the result must match
        // the dense output of the integrator, which evaluates the same Taylor polynomial from the same starting time.
        auto c_out = *std::get<4>(res);
        REQUIRE(c_out(1.5) == state);
        for (const auto tm : {0.5, 2.5}) {
            const auto &c_out_res = c_out(tm);
            const auto &d_out_res = ta.update_d_output(tm);
            REQUIRE(c_out_res.size() == d_out_res.size());
            for (std::size_t j = 0; j < c_out_res.size(); ++j) {
                REQUIRE(std::abs(c_out_res[j] - d_out_res[j]) <= 1e-14);
            }
        }
    }

    // propagate_grid() with a single grid point: no steps are reported.
    auto out = ta.propagate_grid({1.5});
    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<1>(out) == std::numeric_limits<double>::infinity());
    REQUIRE(std::get<2>(out) == 0.);
    REQUIRE(std::get<3>(out) == 0u);
    REQUIRE(std::get<5>(out) == state);
    REQUIRE(ta.get_dtime() == std::pair{1.5, 0.});

    // Same, but with a time coordinate whose low part is nonzero. The initial propagation up to the first grid point
    // takes a small step, which is not reported.
    ta.set_dtime(1.5, 1e-17);
    out = ta.propagate_grid({1.5});
    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<1>(out) == std::numeric_limits<double>::infinity());
    REQUIRE(std::get<2>(out) == 0.);
    REQUIRE(std::get<3>(out) == 0u);
    REQUIRE(std::get<5>(out).size() == 2u);
    REQUIRE(ta.get_dtime() == std::pair{1.5, 0.});
}

// This test checks a corner case in which a last step which does not use all the remaining time ends up exactly on the
// final time due to floating-point rounding.
//
// The numerical values in this test were found by a numerical search using double-length arithmetic.
TEST_CASE("exact landing")
{
    auto x = make_vars("x");

    auto ta = taylor_adaptive<double>{{prime(x) = 1_dbl}, {0.}};

    auto n_cb_calls = 0;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    // NOTE: each tuple contains the initial time (hi/lo), delta_t, max_delta_t and the final time (hi/lo).
    for (const auto &[hi, lo, delta_t, max_delta_t, f_hi, f_lo] :
         {std::tuple{1., 1.547948418977138e-17, 1.5852072202659783e-22, 1.5852072202671788e-22, 1.,
                     1.5479642710493408e-17},
          std::tuple{1., -2.4976763061221956e-17, 6.1050033303071096e-21, 6.1050033303066725e-21, 1.,
                     -2.497065805789165e-17}}) {
        // Forward and backward.
        for (const auto sign : {1., -1.}) {
            ta.set_dtime(sign * hi, sign * lo);
            n_cb_calls = 0;

            const auto res = ta.propagate_for(sign * delta_t, kw::max_delta_t = max_delta_t, kw::callback = cb);

            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
            REQUIRE(std::get<3>(res) == 1u);
            REQUIRE(n_cb_calls == 1);
            REQUIRE(ta.get_dtime() == std::pair{sign * f_hi, sign * f_lo});
        }
    }
}

// If max_steps is equal to the number of steps needed to complete the propagation, the outcome must be time_limit.
TEST_CASE("max_steps")
{
    auto x = make_vars("x");

    auto ta = taylor_adaptive<double>{{prime(x) = 1_dbl}, {0.}};

    // NOTE: with x' = 1 and max_delta_t = 0.25, every step has a size of 0.25. Thus, a propagation over a time interval
    // of 1 takes exactly 4 steps.
    for (const auto sign : {1., -1.}) {
        // NOTE: each tuple contains max_steps, the expected outcome, the expected number of steps, the expected final
        // time and the expected size of the output of propagate_grid().
        for (const auto &[max_steps, oc, n_steps, t, out_size] :
             {std::tuple{3u, taylor_outcome::step_limit, 3u, 0.75, 2u},
              std::tuple{4u, taylor_outcome::time_limit, 4u, 1., 3u},
              std::tuple{5u, taylor_outcome::time_limit, 4u, 1., 3u}}) {
            // propagate_until().
            ta.set_time(0.);
            const auto res = ta.propagate_until(sign * 1., kw::max_delta_t = 0.25, kw::max_steps = max_steps);
            REQUIRE(std::get<0>(res) == oc);
            REQUIRE(std::get<3>(res) == n_steps);
            REQUIRE(ta.get_dtime() == std::pair{sign * t, 0.});

            // propagate_grid().
            ta.set_time(0.);
            const auto out
                = ta.propagate_grid({0., sign * 0.5, sign * 1.}, kw::max_delta_t = 0.25, kw::max_steps = max_steps);
            REQUIRE(std::get<0>(out) == oc);
            REQUIRE(std::get<3>(out) == n_steps);
            REQUIRE(std::get<5>(out).size() == out_size);
            REQUIRE(ta.get_dtime() == std::pair{sign * t, 0.});
        }
    }
}

// After an interruption via max_steps, the state of the integrator and the results must be consistent with an
// uninterrupted propagation, and the propagation must be resumable.
TEST_CASE("max_steps interruption")
{
    auto [x, v] = make_vars("x", "v");

    const auto make_ta = [x, v]() { return taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}}; };

    auto n_cb_calls = 0u;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    for (const auto sign : {1., -1.}) {
        // propagate_until().
        {
            // Uninterrupted propagation.
            auto ta_ref = make_ta();
            const auto res_ref = ta_ref.propagate_until(sign * 10.);
            REQUIRE(std::get<0>(res_ref) == taylor_outcome::time_limit);
            // NOTE: make sure the interrupted propagation will be interrupted more than once.
            REQUIRE(std::get<3>(res_ref) > 6u);

            // Interrupted propagation.
            auto ta = make_ta();
            n_cb_calls = 0;
            const auto res = ta.propagate_until(sign * 10., kw::max_steps = 3u, kw::callback = cb, kw::c_output = true);
            REQUIRE(std::get<0>(res) == taylor_outcome::step_limit);
            REQUIRE(std::get<3>(res) == 3u);
            REQUIRE(n_cb_calls == 3u);
            REQUIRE(std::get<4>(res));
            REQUIRE(std::get<4>(res)->get_n_steps() == 3u);
            REQUIRE(std::get<4>(res)->get_bounds() == std::pair{0., ta.get_time()});

            // Resume the propagation, 3 steps at a time.
            auto n_steps = std::get<3>(res);
            while (true) {
                const auto res2 = ta.propagate_until(sign * 10., kw::max_steps = 3u);
                n_steps += std::get<3>(res2);

                if (std::get<0>(res2) == taylor_outcome::time_limit) {
                    break;
                }

                REQUIRE(std::get<0>(res2) == taylor_outcome::step_limit);
                REQUIRE(std::get<3>(res2) == 3u);
            }

            // NOTE: the interrupted propagation takes the same steps as the uninterrupted one.
            REQUIRE(n_steps == std::get<3>(res_ref));
            REQUIRE(ta.get_dtime() == ta_ref.get_dtime());
            REQUIRE(ta.get_state() == ta_ref.get_state());
        }

        // propagate_grid(), interruption in the main loop.
        {
            std::vector<double> grid;
            for (auto i = 0; i <= 1000; ++i) {
                grid.push_back(sign * i / 100.);
            }

            // Uninterrupted propagation.
            auto ta_ref = make_ta();
            const auto out_ref = ta_ref.propagate_grid(grid);
            REQUIRE(std::get<0>(out_ref) == taylor_outcome::time_limit);
            REQUIRE(std::get<3>(out_ref) > 6u);

            // Interrupted propagation.
            auto ta = make_ta();
            n_cb_calls = 0;
            const auto out = ta.propagate_grid(grid, kw::max_steps = 3u, kw::callback = cb);
            REQUIRE(std::get<0>(out) == taylor_outcome::step_limit);
            REQUIRE(std::get<3>(out) == 3u);
            REQUIRE(n_cb_calls == 3u);

            // The output must contain exactly the grid points before the current time.
            const auto [hi, lo] = ta.get_dtime();
            const auto n_before = std::ranges::count_if(grid, [hi, lo, sign](const double t) {
                return sign > 0 ? (t < hi || (t == hi && 0. < lo)) : (t > hi || (t == hi && 0. > lo));
            });
            REQUIRE(n_before > 1);
            REQUIRE(std::get<5>(out).size() == static_cast<std::size_t>(n_before) * 2u);

            // NOTE: the interrupted propagation takes the same steps as the uninterrupted one.
            REQUIRE(std::ranges::equal(std::get<5>(out), std::get<5>(out_ref) | std::views::take(n_before * 2)));
        }

        // propagate_grid(), interruption in the initial propagation up to grid[0]. The low part of the time coordinate
        // is nonzero, and max_delta_t is small enough that several steps are needed to reach grid[0].
        {
            auto ta = make_ta();
            ta.set_dtime(sign * 1., sign * 1e-17);
            n_cb_calls = 0;
            const auto out = ta.propagate_grid({sign * 1., sign * 2.}, kw::max_steps = 3u, kw::max_delta_t = 3e-18,
                                               kw::callback = cb);
            REQUIRE(std::get<0>(out) == taylor_outcome::step_limit);
            REQUIRE(std::get<1>(out) == std::numeric_limits<double>::infinity());
            REQUIRE(std::get<2>(out) == 0.);
            REQUIRE(std::get<3>(out) == 0u);
            REQUIRE(std::get<5>(out).empty());
            REQUIRE(n_cb_calls == 0u);
        }
    }
}

// In the last step, the callback must see the final time exactly. If the callback stops the propagation in the last
// step, the outcome must be cb_stop and, for propagate_grid(), all grid points must be processed.
TEST_CASE("callback last step")
{
    auto [x, v] = make_vars("x", "v");

    const auto make_ta = [x, v]() { return taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}}; };

    for (const auto sign : {1., -1.}) {
        const auto t_final = sign * 10.;

        // Callback which records the time coordinate and stops the propagation when the final time is reached.
        std::vector<std::pair<double, double>> times;
        const auto cb = [&times, t_final](auto &tint) {
            times.push_back(tint.get_dtime());
            return tint.get_dtime() != std::pair{t_final, 0.};
        };

        // propagate_until().
        auto ta = make_ta();
        const auto res = ta.propagate_until(t_final, kw::callback = cb);
        REQUIRE(std::get<0>(res) == taylor_outcome::cb_stop);
        REQUIRE(std::get<3>(res) == times.size());
        REQUIRE(times.back() == std::pair{t_final, 0.});
        REQUIRE(ta.get_dtime() == std::pair{t_final, 0.});

        // propagate_grid().
        std::vector<double> grid;
        for (auto i = 0; i <= 100; ++i) {
            grid.push_back(sign * i / 10.);
        }

        auto ta_ref = make_ta();
        const auto out_ref = ta_ref.propagate_grid(grid);
        REQUIRE(std::get<0>(out_ref) == taylor_outcome::time_limit);

        times.clear();
        ta = make_ta();
        const auto out = ta.propagate_grid(grid, kw::callback = cb);
        REQUIRE(std::get<0>(out) == taylor_outcome::cb_stop);
        REQUIRE(std::get<3>(out) == times.size());
        REQUIRE(times.back() == std::pair{t_final, 0.});
        REQUIRE(ta.get_dtime() == std::pair{t_final, 0.});

        // All grid points must have been processed, with the same results as in the uninterrupted propagation.
        REQUIRE(std::get<5>(out).size() == grid.size() * 2u);
        REQUIRE(std::get<5>(out) == std::get<5>(out_ref));
    }
}

// A grid point which coincides with the end of a step belongs to the next step. Thus, if the propagation is interrupted
// at the end of that step, the grid point is not processed, and the propagation can be resumed with a grid starting from
// that point.
TEST_CASE("grid interruption at step end")
{
    auto x = make_vars("x");

    // NOTE: with x' = 1 and max_delta_t = 0.25, every step has a size of 0.25 and all time coordinates are computed
    // exactly. The value of x is equal to the time coordinate.
    auto ta = taylor_adaptive<double>{{prime(x) = 1_dbl}, {0.}};

    for (const auto sign : {1., -1.}) {
        ta.set_time(0.);
        ta.get_state_data()[0] = 0.;

        // Interrupt after 2 steps, that is, at the grid point sign * 0.5.
        auto out = ta.propagate_grid({0., sign * .25, sign * .5, sign * .75, sign * 1.}, kw::max_delta_t = 0.25,
                                     kw::max_steps = 2u);
        REQUIRE(std::get<0>(out) == taylor_outcome::step_limit);
        REQUIRE(ta.get_dtime() == std::pair{sign * .5, 0.});
        REQUIRE(std::get<5>(out) == std::vector{0., sign * .25});

        // Resume from sign * 0.5.
        out = ta.propagate_grid({sign * .5, sign * .75, sign * 1.}, kw::max_delta_t = 0.25);
        REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
        REQUIRE(std::get<5>(out) == std::vector{sign * .5, sign * .75, sign * 1.});
    }
}

// A stopping terminal event takes precedence over max_steps.
TEST_CASE("terminal event max_steps")
{
    auto x = make_vars("x");

    for (const auto sign : {1., -1.}) {
        // NOTE: with x' = 1, x(0) = 0 and max_delta_t = 0.25, the stopping terminal event triggers at t = sign * 0.6,
        // in the third step.
        const auto make_ta = [x, sign]() {
            return taylor_adaptive<double>{{prime(x) = 1_dbl}, {0.}, kw::t_events = {t_event<double>(x - sign * 0.6)}};
        };

        // NOTE: each tuple contains max_steps, the expected outcome, the expected number of steps and the expected
        // size of the output of propagate_grid().
        for (const auto &[max_steps, oc, n_steps, out_size] : {std::tuple{2u, taylor_outcome::step_limit, 2u, 2u},
                                                               std::tuple{3u, taylor_outcome{-1}, 3u, 4u},
                                                               std::tuple{4u, taylor_outcome{-1}, 3u, 4u}}) {
            // propagate_until().
            auto ta = make_ta();
            const auto res = ta.propagate_until(sign * 1., kw::max_delta_t = 0.25, kw::max_steps = max_steps);
            REQUIRE(std::get<0>(res) == oc);
            REQUIRE(std::get<3>(res) == n_steps);

            // propagate_grid().
            ta = make_ta();
            const auto out = ta.propagate_grid({0., sign * .25, sign * .5, sign * .55, sign * .75, sign * 1.},
                                               kw::max_delta_t = 0.25, kw::max_steps = max_steps);
            REQUIRE(std::get<0>(out) == oc);
            REQUIRE(std::get<3>(out) == n_steps);
            REQUIRE(std::get<5>(out).size() == out_size);
        }
    }
}
