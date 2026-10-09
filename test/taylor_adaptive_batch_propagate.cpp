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

// Repeated and non-monotonic grid points must be rejected regardless of the direction of the grid and of the batch
// element.
TEST_CASE("grid monotonicity")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2};

    for (const auto sign : {1., -1.}) {
        // NOTE: the grids are for a batch size of 2, with the time coordinates of each grid point stored contiguously.
        for (auto grid : {// Repeated first point in batch element 0.
                          std::vector{0., 0., 0., 1., 1., 2.},
                          // Repeated first point in batch element 1.
                          std::vector{0., 0., 1., 0., 2., 1.},
                          // Repeated later point in batch element 0.
                          std::vector{0., 0., 1., 1., 1., 2.},
                          // Repeated later point in batch element 1.
                          std::vector{0., 0., 1., 1., 2., 1.},
                          // Wrong direction in batch element 1, first two points.
                          std::vector{0., 0., 1., -1., 2., -2.},
                          // Wrong direction in batch element 1, later points.
                          std::vector{0., 0., 1., 1., 2., 0.}}) {
            for (auto &val : grid) {
                val *= sign;
            }

            REQUIRE_THROWS_MATCHES(ta.propagate_grid(grid), std::invalid_argument,
                                   Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                           "Taylor integrator in batch mode"));
        }

        // Valid grids, forward and backward with respect to sign.
        auto out = ta.propagate_grid({0., 0., sign * 1., sign * 2., sign * 2., sign * 3.});
        REQUIRE(std::get<1>(out).size() == 12u);
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }

        out = ta.propagate_grid({sign * 2., sign * 3., sign * 1., sign * 2., 0., 0.});
        REQUIRE(std::get<1>(out).size() == 12u);
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }
    }
}

// At the end of a successful propagation, the time coordinate of each batch element must be exactly the final time.
TEST_CASE("snap")
{
    constexpr auto inf = std::numeric_limits<double>::infinity();

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2};

    // Helper to check the outcomes and the final times.
    const auto check = [&ta](const std::vector<double> &t) {
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }
        REQUIRE(ta.get_dtime().first == t);
        REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
    };

    // NOTE: forward and backward targets, in some cases in different directions for the two batch elements. The last
    // targets bring the time coordinates back to zero.
    const auto targets = std::vector<std::vector<double>>{{1.1, -2.3}, {10.3, 7.1}, {-0.7, 7.9},
                                                          {-12.9, -0.3}, {3.3, -5.5}, {0., 0.}};

    for (const auto &max_delta_t : {std::vector{inf, inf}, std::vector{0.0123, 0.0071}}) {
        // propagate_until().
        for (const auto &t : targets) {
            ta.propagate_until(t, kw::max_delta_t = max_delta_t);
            check(t);
        }

        // propagate_for(), back and forth from zero.
        for (const auto &[delta_t, t] : std::vector<std::pair<std::vector<double>, std::vector<double>>>{
                 {{1.1, -12.9}, {1.1, -12.9}},
                 {{-1.1, 12.9}, {0., 0.}},
                 {{-12.9, 1.1}, {-12.9, 1.1}},
                 {{12.9, -1.1}, {0., 0.}}}) {
            ta.propagate_for(delta_t, kw::max_delta_t = max_delta_t);
            check(t);
        }

        // propagate_grid(), forward and backward.
        for (const auto &grid : {std::vector{0., 0., 0.3, 1.7, 5.7, 6.1, 10.3, 9.9},
                                 std::vector{10.3, 9.9, 2.1, 3.3, -0.7, -0.1, -12.9, -11.3},
                                 std::vector{-12.9, -11.3, -1.1, -2.2, 0., 0.}}) {
            ta.propagate_grid(grid, kw::max_delta_t = max_delta_t);
            check({grid[grid.size() - 2u], grid.back()});
        }
    }
}

// Propagating up to the current time results in a single zero-length step for each batch element.
TEST_CASE("propagate to current time")
{
    constexpr auto inf = std::numeric_limits<double>::infinity();

    // The expected propagation results for a single zero-length step.
    const auto zero_step_res = std::tuple{taylor_outcome::time_limit, inf, 0., std::size_t(1)};

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2};
    ta.set_time({1.5, -0.5});

    auto state = ta.get_state();

    auto n_cb_calls = 0;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    // propagate_until() and propagate_for().
    for (auto i = 0; i < 2; ++i) {
        n_cb_calls = 0;

        const auto c_out = (i == 0)
                               ? std::get<0>(ta.propagate_until({1.5, -0.5}, kw::c_output = true, kw::callback = cb))
                               : std::get<0>(ta.propagate_for({0., 0.}, kw::c_output = true, kw::callback = cb));

        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(res == zero_step_res);
        }
        REQUIRE(c_out);
        REQUIRE(c_out->get_n_steps() == 1u);
        REQUIRE(c_out->get_bounds() == std::pair{std::vector{1.5, -0.5}, std::vector{1.5, -0.5}});
        REQUIRE(n_cb_calls == 1);
        REQUIRE(ta.get_dtime().first == std::vector{1.5, -0.5});
        REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
        REQUIRE(ta.get_state() == state);
    }

    // propagate_grid() with a single grid point: no steps are reported.
    auto out = ta.propagate_grid({1.5, -0.5});
    for (const auto &res : ta.get_propagate_res()) {
        REQUIRE(res == std::tuple{taylor_outcome::time_limit, inf, 0., std::size_t(0)});
    }
    REQUIRE(std::get<1>(out) == state);
    REQUIRE(ta.get_dtime().first == std::vector{1.5, -0.5});
    REQUIRE(ta.get_dtime().second == std::vector{0., 0.});

    // Same, but with time coordinates whose low parts are nonzero. The initial propagation up to the first grid point
    // takes small steps, which are not reported.
    ta.set_dtime({1.5, -0.5}, {1e-17, -1e-17});
    out = ta.propagate_grid({1.5, -0.5});
    for (const auto &res : ta.get_propagate_res()) {
        REQUIRE(res == std::tuple{taylor_outcome::time_limit, inf, 0., std::size_t(0)});
    }
    REQUIRE(std::get<1>(out).size() == 4u);
    REQUIRE(ta.get_dtime().first == std::vector{1.5, -0.5});
    REQUIRE(ta.get_dtime().second == std::vector{0., 0.});

    // Batch element 0 is propagated up to its current time, batch element 1 is not. Batch element 0 takes a single
    // zero-length step which is counted, followed by zero-length steps which are not counted while waiting for batch
    // element 1.
    state = ta.get_state();
    ta.propagate_until({1.5, 3.});
    REQUIRE(ta.get_propagate_res()[0] == zero_step_res);
    REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::time_limit);
    REQUIRE(std::get<3>(ta.get_propagate_res()[1]) > 1u);
    REQUIRE(ta.get_dtime().first == std::vector{1.5, 3.});
    REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
    // NOTE: the state of batch element 0 is unchanged.
    REQUIRE(ta.get_state()[0] == state[0]);
    REQUIRE(ta.get_state()[2] == state[2]);
}

// This test checks a corner case in which a last step which does not use all the remaining time ends up exactly on the
// final time due to floating-point rounding.
//
// The numerical values in this test were found by a numerical search using double-length arithmetic (see the scalar
// version of this test).
TEST_CASE("exact landing")
{
    auto x = make_vars("x");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = 1_dbl}, {0., 0.}, 2};

    auto n_cb_calls = 0;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    // NOTE: initial times (hi/lo), delta_t, max_delta_t and final times (hi/lo) for the two batch elements.
    const auto hi = std::vector{1., 1.};
    const auto lo = std::vector{1.547948418977138e-17, -2.4976763061221956e-17};
    const auto delta_t = std::vector{1.5852072202659783e-22, 6.1050033303071096e-21};
    const auto max_delta_t = std::vector{1.5852072202671788e-22, 6.1050033303066725e-21};
    const auto f_hi = std::vector{1., 1.};
    const auto f_lo = std::vector{1.5479642710493408e-17, -2.497065805789165e-17};

    // Forward and backward, also in different directions for the two batch elements.
    for (const auto &[s0, s1] : {std::pair{1., 1.}, std::pair{-1., -1.}, std::pair{1., -1.}, std::pair{-1., 1.}}) {
        ta.set_dtime({s0 * hi[0], s1 * hi[1]}, {s0 * lo[0], s1 * lo[1]});
        n_cb_calls = 0;

        ta.propagate_for({s0 * delta_t[0], s1 * delta_t[1]}, kw::max_delta_t = max_delta_t, kw::callback = cb);

        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
            REQUIRE(std::get<3>(res) == 1u);
        }
        REQUIRE(n_cb_calls == 1);
        REQUIRE(ta.get_dtime().first == std::vector{s0 * f_hi[0], s1 * f_hi[1]});
        REQUIRE(ta.get_dtime().second == std::vector{s0 * f_lo[0], s1 * f_lo[1]});
    }
}

// If max_steps is equal to the number of iterations needed to complete the propagation, the outcome must be time_limit.
TEST_CASE("max_steps")
{
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    auto x = make_vars("x");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = 1_dbl}, {0., 0.}, 2};

    // NOTE: with x' = 1 and max_delta_t = 0.25, every step has a size of 0.25 and x is equal to the time coordinate.
    // Batch element 0 is propagated over a time interval of 1 (4 steps), batch element 1 over a time interval of 0.5
    // (2 steps).
    for (const auto sign : {1., -1.}) {
        // NOTE: each tuple contains max_steps, the expected outcome, the expected number of steps and final time of
        // batch element 0, and the expected output of propagate_grid() (modulo sign).
        for (const auto &[max_steps, oc, n_steps0, t0, grid_out] :
             {std::tuple{3u, taylor_outcome::step_limit, 3u, 0.75, std::vector{0., 0., .5, .25, nan, .5}},
              std::tuple{4u, taylor_outcome::time_limit, 4u, 1., std::vector{0., 0., .5, .25, 1., .5}},
              std::tuple{5u, taylor_outcome::time_limit, 4u, 1., std::vector{0., 0., .5, .25, 1., .5}}}) {
            // Helper to check the propagation results and the final times.
            const auto check = [&]() {
                for (const auto &res : ta.get_propagate_res()) {
                    REQUIRE(std::get<0>(res) == oc);
                }
                REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == n_steps0);
                REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 2u);
                REQUIRE(ta.get_dtime().first == std::vector{sign * t0, sign * .5});
                REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
            };

            // propagate_until().
            ta.set_time(0.);
            ta.propagate_until({sign * 1., sign * .5}, kw::max_delta_t = {.25, .25}, kw::max_steps = max_steps);
            check();

            // propagate_grid().
            ta.set_time(0.);
            ta.get_state_data()[0] = 0.;
            ta.get_state_data()[1] = 0.;
            const auto out = std::get<1>(ta.propagate_grid({0., 0., sign * .5, sign * .25, sign * 1., sign * .5},
                                                           kw::max_delta_t = {.25, .25}, kw::max_steps = max_steps));
            check();
            REQUIRE(out.size() == grid_out.size());
            for (decltype(out.size()) i = 0; i < out.size(); ++i) {
                if (std::isnan(grid_out[i])) {
                    REQUIRE(std::isnan(out[i]));
                } else {
                    REQUIRE(out[i] == sign * grid_out[i]);
                }
            }
        }
    }
}

// After an interruption via max_steps, the state of the integrator and the results must be consistent with an
// uninterrupted propagation, and the propagation must be resumable.
TEST_CASE("max_steps interruption")
{
    constexpr auto inf = std::numeric_limits<double>::infinity();

    auto [x, v] = make_vars("x", "v");

    const auto make_ta = [x, v]() {
        return taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., .5, 1., .7}, 2};
    };

    auto n_cb_calls = 0u;
    const auto cb = [&n_cb_calls](auto &) {
        ++n_cb_calls;
        return true;
    };

    for (const auto sign : {1., -1.}) {
        // propagate_until().
        {
            const auto t_final = std::vector{sign * 12., sign * 9.};

            // Uninterrupted propagation.
            auto ta_ref = make_ta();
            ta_ref.propagate_until(t_final);
            const auto res_ref = ta_ref.get_propagate_res();
            for (const auto &res : res_ref) {
                REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
                // NOTE: make sure the interrupted propagation will be interrupted more than once.
                REQUIRE(std::get<3>(res) > 6u);
            }

            // Interrupted propagation.
            auto ta = make_ta();
            n_cb_calls = 0;
            const auto c_out = std::get<0>(
                ta.propagate_until(t_final, kw::max_steps = 3u, kw::callback = cb, kw::c_output = true));
            for (const auto &res : ta.get_propagate_res()) {
                REQUIRE(std::get<0>(res) == taylor_outcome::step_limit);
                REQUIRE(std::get<3>(res) == 3u);
            }
            REQUIRE(n_cb_calls == 3u);
            REQUIRE(c_out);
            REQUIRE(c_out->get_n_steps() == 3u);
            REQUIRE(c_out->get_bounds() == std::pair{std::vector{0., 0.}, ta.get_dtime().first});

            // Resume the propagation, 3 iterations at a time.
            //
            // NOTE: a batch element which has already reached its final time at the beginning of a call takes a
            // zero-length step which is counted (see the "propagate to current time" test). We keep track of these
            // extra steps in n_extra.
            std::vector<std::size_t> n_steps(2), n_extra(2);
            for (std::size_t i = 0; i < 2u; ++i) {
                n_steps[i] = std::get<3>(ta.get_propagate_res()[i]);
            }
            while (true) {
                for (std::size_t i = 0; i < 2u; ++i) {
                    n_extra[i] += (ta.get_dtime().first[i] == t_final[i] && ta.get_dtime().second[i] == 0.);
                }

                ta.propagate_until(t_final, kw::max_steps = 3u);

                for (std::size_t i = 0; i < 2u; ++i) {
                    n_steps[i] += std::get<3>(ta.get_propagate_res()[i]);
                }

                if (std::ranges::all_of(ta.get_propagate_res(), [](const auto &res) {
                        return std::get<0>(res) == taylor_outcome::time_limit;
                    })) {
                    break;
                }

                for (const auto &res : ta.get_propagate_res()) {
                    REQUIRE(std::get<0>(res) == taylor_outcome::step_limit);
                }
            }

            // NOTE: the interrupted propagation takes the same steps as the uninterrupted one.
            for (std::size_t i = 0; i < 2u; ++i) {
                REQUIRE(n_steps[i] == std::get<3>(res_ref[i]) + n_extra[i]);
            }
            REQUIRE(ta.get_dtime() == ta_ref.get_dtime());
            REQUIRE(ta.get_state() == ta_ref.get_state());
        }

        // propagate_grid(), interruption in the main loop.
        {
            // NOTE: the grid points of batch element 1 are those of batch element 0 multiplied by 0.9.
            std::vector<double> grid;
            for (auto i = 0; i <= 1000; ++i) {
                grid.push_back(sign * i / 100.);
                grid.push_back(sign * i / 100. * 0.9);
            }

            // Uninterrupted propagation.
            auto ta_ref = make_ta();
            const auto out_ref = std::get<1>(ta_ref.propagate_grid(grid));
            for (const auto &res : ta_ref.get_propagate_res()) {
                REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
                REQUIRE(std::get<3>(res) > 6u);
            }

            // Interrupted propagation.
            auto ta = make_ta();
            n_cb_calls = 0;
            const auto out = std::get<1>(ta.propagate_grid(grid, kw::max_steps = 3u, kw::callback = cb));
            for (const auto &res : ta.get_propagate_res()) {
                REQUIRE(std::get<0>(res) == taylor_outcome::step_limit);
                REQUIRE(std::get<3>(res) == 3u);
            }
            REQUIRE(n_cb_calls == 3u);

            // For each batch element, the output must contain exactly the grid points before the current time, with
            // the same values as in the uninterrupted propagation. The other grid points must be NaN.
            //
            // NOTE: the output is stored as [grid point][state variable][batch element].
            const auto [hi, lo] = ta.get_dtime();
            for (std::size_t i = 0; i < 2u; ++i) {
                std::size_t n_before = 0;

                for (std::size_t gidx = 0; gidx < grid.size() / 2u; ++gidx) {
                    const auto t = grid[gidx * 2u + i];
                    const auto before = sign > 0 ? (t < hi[i] || (t == hi[i] && 0. < lo[i]))
                                                 : (t > hi[i] || (t == hi[i] && 0. > lo[i]));
                    n_before += before;

                    for (std::size_t j = 0; j < 2u; ++j) {
                        const auto idx = gidx * 4u + j * 2u + i;
                        if (before) {
                            REQUIRE(out[idx] == out_ref[idx]);
                        } else {
                            REQUIRE(std::isnan(out[idx]));
                        }
                    }
                }

                REQUIRE(n_before > 1u);
            }
        }

        // propagate_grid(), interruption in the initial propagation up to the first grid points. The low parts of the
        // time coordinates are nonzero, and max_delta_t is small enough that several steps are needed to reach the
        // first grid points.
        {
            auto ta = make_ta();
            ta.set_dtime({sign * 1., sign * 1.}, {sign * 1e-17, -sign * 1e-17});
            n_cb_calls = 0;
            const auto out = std::get<1>(ta.propagate_grid({sign * 1., sign * 1., sign * 2., sign * 2.},
                                                           kw::max_steps = 3u, kw::max_delta_t = {3e-18, 3e-18},
                                                           kw::callback = cb));
            for (const auto &res : ta.get_propagate_res()) {
                REQUIRE(res == std::tuple{taylor_outcome::step_limit, inf, 0., std::size_t(0)});
            }
            REQUIRE(out.size() == 8u);
            REQUIRE(std::ranges::all_of(out, [](const double val) { return std::isnan(val); }));
            REQUIRE(n_cb_calls == 0u);
        }
    }
}

// In the last step, the callback must see the final times exactly. If the callback stops the propagation in the last
// step, the outcome must be cb_stop and, for propagate_grid(), all grid points must be processed.
TEST_CASE("callback last step")
{
    auto [x, v] = make_vars("x", "v");

    const auto make_ta = [x, v]() {
        return taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., .5, 1., .7}, 2};
    };

    for (const auto sign : {1., -1.}) {
        std::vector<double> t_final;

        // Callback which records the time coordinates and stops the propagation when all batch elements have reached
        // their final times.
        std::vector<std::vector<double>> times_hi, times_lo;
        const auto cb = [&times_hi, &times_lo, &t_final](auto &tint) {
            times_hi.push_back(tint.get_dtime().first);
            times_lo.push_back(tint.get_dtime().second);
            return !(tint.get_dtime().first == t_final && tint.get_dtime().second == std::vector{0., 0.});
        };

        // Helper to check the propagation results and the final times.
        //
        // NOTE: batch element 0 is propagated over a longer time interval than batch element 1, thus it takes a step
        // in every iteration. Since the callback is invoked once per iteration, the number of steps of batch element 0
        // is equal to the number of callback invocations.
        const auto check = [&times_hi, &times_lo, &t_final](const auto &ta) {
            for (const auto &res : ta.get_propagate_res()) {
                REQUIRE(std::get<0>(res) == taylor_outcome::cb_stop);
            }
            REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == times_hi.size());
            REQUIRE(times_hi.back() == t_final);
            REQUIRE(times_lo.back() == std::vector{0., 0.});
            REQUIRE(ta.get_dtime().first == t_final);
            REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
        };

        // propagate_until().
        t_final = {sign * 10., sign * 8.};
        auto ta = make_ta();
        ta.propagate_until(t_final, kw::callback = cb);
        check(ta);

        // propagate_grid().
        std::vector<double> grid;
        for (auto i = 0; i <= 100; ++i) {
            grid.push_back(sign * i / 10.);
            grid.push_back(sign * i / 10. * 0.8);
        }
        t_final = {grid[grid.size() - 2u], grid.back()};

        auto ta_ref = make_ta();
        const auto out_ref = std::get<1>(ta_ref.propagate_grid(grid));
        for (const auto &res : ta_ref.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }

        times_hi.clear();
        times_lo.clear();
        ta = make_ta();
        const auto out = std::get<1>(ta.propagate_grid(grid, kw::callback = cb));
        check(ta);

        // All grid points must have been processed, with the same results as in the uninterrupted propagation.
        REQUIRE(out == out_ref);
    }
}

// A grid point which coincides with the end of a step belongs to the next step. Thus, if the propagation is interrupted
// at the end of that step, the grid point is not processed, and the propagation can be resumed with a grid starting from
// that point.
TEST_CASE("grid interruption at step end")
{
    auto x = make_vars("x");

    // NOTE: with x' = 1 and max_delta_t = {0.25, 0.125}, every step has a size of 0.25 for batch element 0 and of 0.125
    // for batch element 1. All time coordinates are computed exactly, and the value of x is equal to the time
    // coordinate.
    auto ta = taylor_adaptive_batch<double>{{prime(x) = 1_dbl}, {0., 0.}, 2};

    for (const auto sign : {1., -1.}) {
        ta.set_time(0.);
        ta.get_state_data()[0] = 0.;
        ta.get_state_data()[1] = 0.;

        // Interrupt after 2 steps, that is, at the grid points sign * 0.5 (batch element 0) and sign * 0.25 (batch
        // element 1).
        auto out = std::get<1>(ta.propagate_grid({0., 0., sign * .25, sign * .125, sign * .5, sign * .25, sign * .75,
                                                  sign * .375, sign * 1., sign * .5},
                                                 kw::max_delta_t = {.25, .125}, kw::max_steps = 2u));
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::step_limit);
        }
        REQUIRE(ta.get_dtime().first == std::vector{sign * .5, sign * .25});
        REQUIRE(ta.get_dtime().second == std::vector{0., 0.});
        REQUIRE(out.size() == 10u);
        REQUIRE(std::ranges::equal(out | std::views::take(4), std::vector{0., 0., sign * .25, sign * .125}));
        REQUIRE(std::ranges::all_of(out | std::views::drop(4), [](const double val) { return std::isnan(val); }));

        // Resume from the current time coordinates.
        out = std::get<1>(ta.propagate_grid({sign * .5, sign * .25, sign * .75, sign * .375, sign * 1., sign * .5},
                                            kw::max_delta_t = {.25, .125}));
        for (const auto &res : ta.get_propagate_res()) {
            REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);
        }
        REQUIRE(out == std::vector{sign * .5, sign * .25, sign * .75, sign * .375, sign * 1., sign * .5});
    }
}

// A stopping terminal event takes precedence over max_steps.
TEST_CASE("terminal event max_steps")
{
    auto x = make_vars("x");

    for (const auto sign : {1., -1.}) {
        // NOTE: with x' = 1 and max_delta_t = 0.25, the stopping terminal event triggers at t = sign * 0.6, in the third
        // step, for batch element 0 (x(0) = 0). It never triggers for batch element 1 (x(0) = sign), which is propagated
        // up to t = sign * 1 (4 steps).
        const auto make_ta = [x, sign]() {
            return taylor_adaptive_batch<double>{
                {prime(x) = 1_dbl}, {0., sign}, 2, kw::t_events = {t_event_batch<double>(x - sign * 0.6)}};
        };

        const auto grid = std::vector{0.,         0.,         sign * .25, sign * .25, sign * .5, sign * .5,
                                      sign * .55, sign * .55, sign * .75, sign * .75, sign * 1., sign * 1.};

        // NOTE: each tuple contains max_steps, the expected outcomes of the two batch elements, the expected number of
        // steps and the expected number of processed grid points (the same for both batch elements).
        for (const auto &[max_steps, oc0, oc1, n_steps, n_points] :
             {std::tuple{2u, taylor_outcome::step_limit, taylor_outcome::step_limit, 2u, 2u},
              std::tuple{3u, taylor_outcome{-1}, taylor_outcome::time_limit, 3u, 4u},
              std::tuple{4u, taylor_outcome{-1}, taylor_outcome::time_limit, 3u, 4u}}) {
            // Helper to check the propagation results.
            const auto check = [&](const auto &ta) {
                const auto &res = ta.get_propagate_res();
                REQUIRE(std::get<0>(res[0]) == oc0);
                REQUIRE(std::get<0>(res[1]) == oc1);
                REQUIRE(std::get<3>(res[0]) == n_steps);
                REQUIRE(std::get<3>(res[1]) == n_steps);
            };

            // propagate_until().
            auto ta = make_ta();
            ta.propagate_until(sign * 1., kw::max_delta_t = {.25, .25}, kw::max_steps = max_steps);
            check(ta);

            // propagate_grid().
            ta = make_ta();
            const auto out
                = std::get<1>(ta.propagate_grid(grid, kw::max_delta_t = {.25, .25}, kw::max_steps = max_steps));
            check(ta);
            REQUIRE(out.size() == 12u);
            for (std::size_t gidx = 0; gidx < 6u; ++gidx) {
                for (std::size_t i = 0; i < 2u; ++i) {
                    REQUIRE(std::isnan(out[gidx * 2u + i]) == (gidx >= n_points));
                }
            }
        }
    }
}

// In the screen output of the batch continuous output, a batch element whose time coordinate does not change must be
// reported as forward, consistently with the direction used internally.
TEST_CASE("c_out stream zero direction")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2};
    ta.set_time({0., 1.5});

    // Batch element 0 is propagated backward, batch element 1 is propagated up to its current time.
    const auto c_out = std::get<0>(ta.propagate_until({-1., 1.5}, kw::c_output = true));
    REQUIRE(c_out);

    std::ostringstream oss;
    oss << *c_out;
    REQUIRE(oss.str().find("Directions  : [backward, forward]") != std::string::npos);
}
