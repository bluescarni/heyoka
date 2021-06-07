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
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/math/constants/constants.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

template <typename Out, typename P, typename T>
auto &horner_eval(Out &ret, const P &p, int order, const T &eval)
{
    ret = xt::view(p, xt::all(), order);

    for (--order; order >= 0; --order) {
        ret = xt::view(p, xt::all(), order) + ret * eval;
    }

    return ret;
}

TEST_CASE("batch init outcome")
{
    auto [x, v] = make_vars("x", "v");

    auto ta
        = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, std::vector<double>(46u, 0.), 23u};

    REQUIRE(ta.get_step_res().size() == 23u);
    REQUIRE(ta.get_propagate_res().size() == 23u);

    REQUIRE(std::all_of(ta.get_step_res().begin(), ta.get_step_res().end(), [](const auto &t) {
        auto [oc, h] = t;
        return oc == taylor_outcome::success && h == 0.;
    }));
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(), [](const auto &t) {
        auto [oc, min_h, max_h, steps] = t;
        return oc == taylor_outcome::success && min_h == 0. && max_h == 0. && steps == 0u;
    }));
}

TEST_CASE("propagate grid scalar")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({}), std::invalid_argument,
        Message("Cannot invoke propagate_grid() in an adaptive Taylor integrator if the time grid is empty"));

    ta.set_time(std::numeric_limits<double>::infinity());

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0.}), std::invalid_argument,
        Message("Cannot invoke propagate_grid() in an adaptive Taylor integrator if the current time is not finite"));

    ta.set_time(0.);

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({std::numeric_limits<double>::infinity()}), std::invalid_argument,
        Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1., std::numeric_limits<double>::infinity()}), std::invalid_argument,
        Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1., 2., 1.}), std::invalid_argument,
        Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({-1., -2., 1.}), std::invalid_argument,
        Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 0., 1.}), std::invalid_argument,
        Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 1., 1.}), std::invalid_argument,
        Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 1., 2., 2.}), std::invalid_argument,
        Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

    // Set an infinity in the state.
    ta.get_state_data()[0] = std::numeric_limits<double>::infinity();

    auto out = ta.propagate_grid({.2});
    REQUIRE(std::get<0>(out) == taylor_outcome::err_nf_state);
    REQUIRE(std::get<4>(out).empty());

    // Reset the integrator.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    // Propagate to the initial time.
    out = ta.propagate_grid({0.});
    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<1>(out) == std::numeric_limits<double>::infinity());
    REQUIRE(std::get<2>(out) == 0);
    REQUIRE(std::get<3>(out) == 0u);
    REQUIRE(std::get<4>(out) == std::vector{0.05, 0.025});
    REQUIRE(ta.get_time() == 0.);

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({2., std::numeric_limits<double>::infinity()}), std::invalid_argument,
        Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

    // Switch to the harmonic oscillator.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

    // Integrate forward over a dense grid from 0 to 10.
    std::vector<double> grid;
    for (auto i = 0u; i < 1000u; ++i) {
        grid.push_back(i / 100.);
    }
    out = ta.propagate_grid(grid);

    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<4>(out).size() == 2000u);
    REQUIRE(ta.get_time() == grid.back());

    for (auto i = 0u; i < 1000u; ++i) {
        REQUIRE(std::get<4>(out)[2u * i] == approximately(std::sin(grid[i]), 10000.));
        REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(std::cos(grid[i]), 10000.));
    }

    // Do the same backwards.
    ta.set_time(10.);
    ta.get_state_data()[0] = std::sin(10.);
    ta.get_state_data()[1] = std::cos(10.);
    std::reverse(grid.begin(), grid.end());

    out = ta.propagate_grid(grid);

    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<4>(out).size() == 2000u);
    REQUIRE(ta.get_time() == grid.back());

    for (auto i = 0u; i < 1000u; ++i) {
        REQUIRE(std::get<4>(out)[2u * i] == approximately(std::sin(grid[i]), 10000.));
        REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(std::cos(grid[i]), 10000.));
    }

    // Random testing.
    ta.set_time(0.);
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[1] = 1.;

    std::uniform_real_distribution<double> rdist(0., .1);
    grid[0] = 0;
    for (auto i = 1u; i < 1000u; ++i) {
        grid[i] = grid[i - 1u] + rdist(rng);
    }

    out = ta.propagate_grid(grid);

    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<4>(out).size() == 2000u);
    REQUIRE(ta.get_time() == grid.back());

    for (auto i = 0u; i < 1000u; ++i) {
        REQUIRE(std::get<4>(out)[2u * i] == approximately(std::sin(grid[i]), 100000.));
        REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(std::cos(grid[i]), 100000.));
    }

    // Do it also backwards.
    ta.set_time(0.);
    ta.get_state_data()[0] = 0.;
    ta.get_state_data()[1] = 1.;

    rdist = std::uniform_real_distribution<double>(-.1, 0.);
    grid[0] = 0;
    for (auto i = 1u; i < 1000u; ++i) {
        grid[i] = grid[i - 1u] + rdist(rng);
    }

    out = ta.propagate_grid(grid);

    REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
    REQUIRE(std::get<4>(out).size() == 2000u);
    REQUIRE(ta.get_time() == grid.back());

    for (auto i = 0u; i < 1000u; ++i) {
        REQUIRE(std::get<4>(out)[2u * i] == approximately(std::sin(grid[i]), 100000.));
        REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(std::cos(grid[i]), 100000.));
    }

    // A test with a sparse grid.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

    out = ta.propagate_grid({.1, 10., 100.});

    REQUIRE(std::get<4>(out).size() == 6u);
    REQUIRE(ta.get_time() == 100.);
    REQUIRE(std::get<4>(out)[0] == approximately(std::sin(.1), 100.));
    REQUIRE(std::get<4>(out)[1] == approximately(std::cos(.1), 100.));
    REQUIRE(std::get<4>(out)[2] == approximately(std::sin(10.), 100.));
    REQUIRE(std::get<4>(out)[3] == approximately(std::cos(10.), 100.));
    REQUIRE(std::get<4>(out)[4] == approximately(std::sin(100.), 1000.));
    REQUIRE(std::get<4>(out)[5] == approximately(std::cos(100.), 1000.));

    // A case in which the initial propagate_until() to bring the system
    // to grid[0] interrupts the integration.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 1.}, kw::t_events = {t_event<double>(v - 0.999)}};
    out = ta.propagate_grid({10., 100.});
    REQUIRE(std::get<0>(out) == taylor_outcome{-1});
    REQUIRE(std::get<4>(out).empty());

    ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -x},
        {0., 1.},
        kw::t_events = {t_event<double>(
            v - 0.999, kw::callback = [](taylor_adaptive<double> &, bool, int) { return false; })}};
    out = ta.propagate_grid({10., 100.});
    REQUIRE(std::get<0>(out) == taylor_outcome{-1});
    REQUIRE(std::get<4>(out).empty());
}

TEST_CASE("streaming op")
{
    auto sys = make_nbody_sys(2, kw::masses = {1., 0.});

    std::ostringstream oss;

    {
        auto tad = taylor_adaptive<double>{sys, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}};

        oss << tad;

        REQUIRE(!oss.str().empty());
        REQUIRE(!boost::algorithm::contains(oss.str(), "events"));

        oss.str("");
    }

    using t_ev_t = t_event<double>;
    using nt_ev_t = nt_event<double>;

    {
        auto tad
            = taylor_adaptive<double>{sys, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}, kw::t_events = {t_ev_t("x_0"_var)}};

        oss << tad;

        REQUIRE(!oss.str().empty());
        REQUIRE(boost::algorithm::contains(oss.str(), "N of terminal events"));
        REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
        REQUIRE(!boost::algorithm::contains(oss.str(), "N of non-terminal events"));

        oss.str("");
    }

    {
        auto tad = taylor_adaptive<double>{sys,
                                           {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
                                           kw::nt_events
                                           = {nt_ev_t("x_0"_var, [](taylor_adaptive<double> &, double, int) {})}};

        oss << tad;

        REQUIRE(!oss.str().empty());
        REQUIRE(!boost::algorithm::contains(oss.str(), "N of terminal events"));
        REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
        REQUIRE(boost::algorithm::contains(oss.str(), "N of non-terminal events"));

        oss.str("");
    }

    {
        auto tad = taylor_adaptive<double>{sys,
                                           {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
                                           kw::t_events = {t_ev_t("x_0"_var)},
                                           kw::nt_events
                                           = {nt_ev_t("x_0"_var, [](taylor_adaptive<double> &, double, int) {})}};

        oss << tad;

        REQUIRE(!oss.str().empty());
        REQUIRE(boost::algorithm::contains(oss.str(), "N of terminal events"));
        REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
        REQUIRE(boost::algorithm::contains(oss.str(), "N of non-terminal events"));

        oss.str("");
    }
}

TEST_CASE("param scalar")
{
    using std::sqrt;
    const auto pi = boost::math::constants::pi<double>();

    const auto init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0};

    auto tad = taylor_adaptive<double>{make_nbody_par_sys(2, kw::n_massive = 1), init_state, kw::tol = 1e-18};

    REQUIRE(tad.get_pars().size() == 1u);

    // Set the central mass to 1.
    tad.get_pars_data()[0] = 1.;

    tad.propagate_for(2 * pi);

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[1] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[3] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[5] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(1.));
    REQUIRE(tad.get_state()[7] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[9] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(1.));
    REQUIRE(tad.get_state()[11] == approximately(0.));

    // Set the central mass to 8
    tad.get_pars_data()[0] = 8.;
    // Set the initial speed so that the orbit stays circular.
    std::copy(init_state.begin(), init_state.end(), tad.get_state_data());
    tad.get_state_data()[10] = sqrt(8.);

    tad.propagate_for(2 * pi * sqrt(1 / 8.));

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[1] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[3] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[5] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(1.));
    REQUIRE(tad.get_state()[7] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[9] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(sqrt(8.)));
    REQUIRE(tad.get_state()[11] == approximately(0.));
}

TEST_CASE("param batch")
{
    using std::cbrt;
    using std::sqrt;
    const auto pi = boost::math::constants::pi<double>();

    auto a0 = 1.;
    auto v0 = 1.;
    auto a1 = cbrt(2);
    auto v1 = sqrt(2 / a1);

    auto init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a0, a1, 0, 0, 0, 0, 0, 0, v0, v1, 0, 0};

    auto tad = taylor_adaptive_batch<double>{make_nbody_par_sys(2, kw::n_massive = 1), init_state, 2, kw::tol = 1e-18};

    REQUIRE(tad.get_pars().size() == 2u);

    // Set the central masses to 1 and 2.
    tad.get_pars_data()[0] = 1.;
    tad.get_pars_data()[1] = 2.;

    tad.propagate_for({2 * pi, 2 * pi});

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(0.));
    REQUIRE(tad.get_state()[12] == approximately(a0));
    REQUIRE(tad.get_state()[14] == approximately(0.));
    REQUIRE(tad.get_state()[16] == approximately(0.));
    REQUIRE(tad.get_state()[18] == approximately(0.));
    REQUIRE(tad.get_state()[20] == approximately(v0));
    REQUIRE(tad.get_state()[22] == approximately(0.));

    REQUIRE(tad.get_state()[0 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[2 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[4 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[6 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[8 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[10 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[12 + 1] == approximately(a1));
    REQUIRE(tad.get_state()[14 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[16 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[18 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[20 + 1] == approximately(v1));
    REQUIRE(tad.get_state()[22 + 1] == approximately(0.));

    // Flip around.
    init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a1, a0, 0, 0, 0, 0, 0, 0, v1, v0, 0, 0};
    std::copy(init_state.begin(), init_state.end(), tad.get_state_data());
    tad.get_pars_data()[0] = 2.;
    tad.get_pars_data()[1] = 1.;

    tad.propagate_for({2 * pi, 2 * pi});

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(0.));
    REQUIRE(tad.get_state()[12] == approximately(a1));
    REQUIRE(tad.get_state()[14] == approximately(0.));
    REQUIRE(tad.get_state()[16] == approximately(0.));
    REQUIRE(tad.get_state()[18] == approximately(0.));
    REQUIRE(tad.get_state()[20] == approximately(v1));
    REQUIRE(tad.get_state()[22] == approximately(0.));

    REQUIRE(tad.get_state()[0 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[2 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[4 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[6 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[8 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[10 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[12 + 1] == approximately(a0));
    REQUIRE(tad.get_state()[14 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[16 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[18 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[20 + 1] == approximately(v0));
    REQUIRE(tad.get_state()[22 + 1] == approximately(0.));
}

// Make sure the last timestep is properly recorded when using the
// step/propagate functions.
TEST_CASE("last h")
{
    auto [x, v] = make_vars("x", "v");

    // Scalar test.
    {
        auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

        REQUIRE(ta.get_last_h() == 0.);
        auto lh = ta.get_last_h();

        ta.step();

        REQUIRE(ta.get_last_h() != 0.);
        lh = ta.get_last_h();

        ta.step(1e-4);

        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.step_backward();

        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.propagate_for(1.23);
        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.propagate_until(0.);
        REQUIRE(ta.get_last_h() != lh);
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        std::vector<double> init_state;
        for (auto i = 0u; i < batch_size; ++i) {
            init_state.push_back(0.05 + i / 100.);
        }
        for (auto i = 0u; i < batch_size; ++i) {
            init_state.push_back(0.025 + i / 1000.);
        }

        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, init_state, batch_size};

        REQUIRE(std::all_of(ta.get_last_h().begin(), ta.get_last_h().end(), [](auto x) { return x == 0.; }));
        auto lh = ta.get_last_h();

        ta.step();
        REQUIRE(std::all_of(ta.get_last_h().begin(), ta.get_last_h().end(), [](auto x) { return x != 0.; }));
        lh = ta.get_last_h();

        ta.step(std::vector<double>(batch_size, 1e-4));
        for (auto i = 0u; i < batch_size; ++i) {
            REQUIRE(lh[i] != ta.get_last_h()[i]);
        }
    }
}

TEST_CASE("dense output")
{
    auto [x, v] = make_vars("x", "v");

    // Scalar test.
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                  {0.05, 0.025},
                                                  kw::high_accuracy = ha,
                                                  kw::compact_mode = cm,
                                                  kw::opt_level = opt_level};

                auto d_out = xt::adapt(ta.get_d_output());

                // Take a first step.
                ta.step(true);

                // The d_out at t = 0 must be the same
                // as the IC.
                ta.update_d_output(0);
                REQUIRE(d_out[0] == approximately(0.05, 100.));
                REQUIRE(d_out[1] == approximately(0.025, 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Store the state at the end of the first step.
                auto old_state1 = ta.get_state();

                // Take a second step.
                ta.step(true);

                // The d_out at the beginning of the timestep
                // must be equal to the state at the end of the
                // previous timestep.
                ta.update_d_output(ta.get_time() - ta.get_last_h());
                REQUIRE(d_out[0] == approximately(old_state1[0], 100.));
                REQUIRE(d_out[1] == approximately(old_state1[1], 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Store the state at the end of the second timestep.
                auto old_state2 = ta.get_state();

                // Take a third timestep.
                ta.step(true);

                // The d_out at the beginning of the timestep
                // must be equal to the state at the end of the
                // previous timestep.
                ta.update_d_output(ta.get_time() - ta.get_last_h());
                REQUIRE(d_out[0] == approximately(old_state2[0], 100.));
                REQUIRE(d_out[1] == approximately(old_state2[1], 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Do it a few more times.
                for (auto i = 0; i < 100; ++i) {
                    old_state2 = ta.get_state();

                    auto [oc, _] = ta.step(true);

                    REQUIRE(oc == taylor_outcome::success);

                    ta.update_d_output(-ta.get_last_h(), true);
                    REQUIRE(d_out[0] == approximately(old_state2[0], 1000.));
                    REQUIRE(d_out[1] == approximately(old_state2[1], 1000.));

                    ta.update_d_output(0., true);
                    REQUIRE(d_out[0] == approximately(ta.get_state()[0], 1000.));
                    REQUIRE(d_out[1] == approximately(ta.get_state()[1], 1000.));
                }
            }
        }
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            for (auto cm : {false, true}) {
                for (auto ha : {false, true}) {
                    std::vector<double> init_state;
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.05 + i / 100.);
                    }
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.025 + i / 1000.);
                    }

                    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                            init_state,
                                                            batch_size,
                                                            kw::high_accuracy = ha,
                                                            kw::compact_mode = cm,
                                                            kw::opt_level = opt_level};

                    auto sa = xt::adapt(ta.get_state_data(), {2u, batch_size});
                    auto isa = xt::adapt(init_state.data(), {2u, batch_size});
                    auto coa = xt::adapt(ta.get_d_output(), {2u, batch_size});

                    ta.step(true);

                    ta.update_d_output(std::vector<double>(batch_size, 0.));
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(isa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(isa(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    auto old_state1 = ta.get_state();
                    auto ost1a = xt::adapt(old_state1, {2u, batch_size});
                    auto old_time1 = ta.get_time();

                    ta.step(true);

                    ta.update_d_output(old_time1);
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(ost1a(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(ost1a(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    auto old_state2 = ta.get_state();
                    auto ost2a = xt::adapt(old_state2, {2u, batch_size});
                    auto old_time2 = ta.get_time();

                    ta.step(true);

                    ta.update_d_output(old_time2);
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(ost2a(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(ost2a(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    for (auto _ = 0; _ < 100; ++_) {
                        auto old_state = ta.get_state();
                        auto osta = xt::adapt(old_state, {2u, batch_size});
                        auto old_time = ta.get_time();

                        ta.step(true);

                        auto neg_last_h = ta.get_last_h();
                        for (auto &tmp : neg_last_h) {
                            tmp = -tmp;
                        }

                        ta.update_d_output(neg_last_h, true);
                        for (auto i = 0u; i < batch_size; ++i) {
                            REQUIRE(coa(0u, i) == approximately(osta(0u, i), 1000.));
                            REQUIRE(coa(1u, i) == approximately(osta(1u, i), 1000.));
                        }

                        std::vector<double> zero_vec(batch_size, 0.);

                        ta.update_d_output(zero_vec, true);
                        for (auto i = 0u; i < batch_size; ++i) {
                            REQUIRE(coa(0u, i) == approximately(sa(0u, i), 1000.));
                            REQUIRE(coa(1u, i) == approximately(sa(1u, i), 1000.));
                        }
                    }

                    using Catch::Matchers::Message;

                    REQUIRE_THROWS_MATCHES(
                        ta.update_d_output({}), std::invalid_argument,
                        Message("Invalid number of time coordinates specified for the dense output in a Taylor "
                                "integrator in batch "
                                "mode: the batch size is "
                                + std::to_string(batch_size) + ", but the number of time coordinates is 0"));
                    REQUIRE_THROWS_MATCHES(
                        ta.update_d_output(std::vector<double>(123u)), std::invalid_argument,
                        Message("Invalid number of time coordinates specified for the dense output in a Taylor "
                                "integrator in batch "
                                "mode: the batch size is "
                                + std::to_string(batch_size) + ", but the number of time coordinates is 123"));
                }
            }
        }
    }
}

TEST_CASE("taylor tc basic")
{
    // Scalar test.
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, v] = make_vars("x", "v");

                auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                  {0.05, 0.025},
                                                  kw::high_accuracy = ha,
                                                  kw::compact_mode = cm,
                                                  kw::opt_level = opt_level};

                REQUIRE(ta.get_tc().size() == 2u * (ta.get_order() + 1u));

                auto tca = xt::adapt(ta.get_tc().data(), {2u, ta.get_order() + 1u});

                auto [oc, h] = ta.step(true);

                auto ret = xt::eval(xt::zeros<double>({2}));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(0.05, 10.));
                REQUIRE(ret[1] == approximately(0.025, 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));

                auto old_state = ta.get_state();

                std::tie(oc, h) = ta.step_backward(true);

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(old_state[0], 10.));
                REQUIRE(ret[1] == approximately(old_state[1], 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));

                old_state = ta.get_state();

                std::tie(oc, h) = ta.step(1e-3, true);

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), 0);
                REQUIRE(ret[0] == approximately(old_state[0], 10.));
                REQUIRE(ret[1] == approximately(old_state[1], 10.));

                horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
                REQUIRE(ret[0] == approximately(ta.get_state()[0], 10.));
                REQUIRE(ret[1] == approximately(ta.get_state()[1], 10.));
            }
        }
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            for (auto cm : {false, true}) {
                for (auto ha : {false, true}) {
                    auto [x, v] = make_vars("x", "v");

                    std::vector<double> init_state;
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.05 + i / 100.);
                    }
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.025 + i / 1000.);
                    }

                    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                            init_state,
                                                            batch_size,
                                                            kw::high_accuracy = ha,
                                                            kw::compact_mode = cm,
                                                            kw::opt_level = opt_level};

                    REQUIRE(ta.get_tc().size() == 2u * (ta.get_order() + 1u) * batch_size);

                    auto sa = xt::adapt(ta.get_state_data(), {2u, batch_size});
                    auto isa = xt::adapt(init_state.data(), {2u, batch_size});

                    {
                        ta.step(true);
                        auto &oc = ta.get_step_res();

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            auto tca = xt::adapt(ta.get_tc(), {2u, ta.get_order() + 1u, batch_size});

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), 0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }

                    init_state = ta.get_state();

                    {
                        ta.step_backward(true);
                        auto &oc = ta.get_step_res();

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            auto tca = xt::adapt(ta.get_tc(), {2u, ta.get_order() + 1u, batch_size});

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), 0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }

                    init_state = ta.get_state();

                    {
                        std::vector<double> max_delta_t;
                        for (auto i = 0u; i < batch_size; ++i) {
                            max_delta_t.push_back(1e-5 + i * 1e-5);
                        }

                        ta.step(max_delta_t, true);
                        auto &oc = ta.get_step_res();

                        for (auto i = 0u; i < batch_size; ++i) {
                            auto ret = xt::eval(xt::zeros<double>({2u}));

                            auto tca = xt::adapt(ta.get_tc(), {2u, ta.get_order() + 1u, batch_size});

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), 0);
                            REQUIRE(ret[0] == approximately(xt::view(isa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(isa, 1, i)[0], 10.));

                            horner_eval(ret, xt::view(std::as_const(tca), xt::all(), xt::all(), i),
                                        static_cast<int>(ta.get_order()), std::get<1>(oc[i]));
                            REQUIRE(ret[0] == approximately(xt::view(sa, 0, i)[0], 10.));
                            REQUIRE(ret[1] == approximately(xt::view(sa, 1, i)[0], 10.));
                        }
                    }
                }
            }
        }
    }
}

// A test to check that vector data passed to the constructor
// is moved into the integrator (and not just copied)
TEST_CASE("taylor scalar move")
{
    auto [x, v] = make_vars("x", "v");

    auto init_state = std::vector{-1., 0.};
    auto pars = std::vector{9.8};
    auto tes = std::vector{t_event<double>(v)};
    auto ntes = std::vector{nt_event<double>(v, [](taylor_adaptive<double> &, double, int) {})};

    auto s_data = init_state.data();
    auto p_data = pars.data();
    auto tes_data = tes.data();
    auto ntes_data = ntes.data();

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -par[0] * sin(x)},
                                      std::move(init_state),
                                      kw::pars = std::move(pars),
                                      kw::t_events = std::move(tes),
                                      kw::nt_events = std::move(ntes)};

    REQUIRE(s_data == ta.get_state().data());
    REQUIRE(p_data == ta.get_pars().data());
    REQUIRE(tes_data == ta.get_t_events().data());
    REQUIRE(ntes_data == ta.get_nt_events().data());
}

// A test to make sure the propagate functions deal correctly
// with trivial dynamics.
TEST_CASE("propagate trivial")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = 1_dbl}, {0, 0}};

    REQUIRE(std::get<0>(ta.propagate_for(1.2)) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta.propagate_until(2.3)) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta.propagate_grid({3, 4, 5, 6, 7.})) == taylor_outcome::time_limit);
}

TEST_CASE("propagate for_until")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};
    auto ta_copy = ta;

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        ta.propagate_until(std::numeric_limits<double>::infinity()), std::invalid_argument,
        Message("A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_until(10., kw::max_delta_t = std::numeric_limits<double>::quiet_NaN()), std::invalid_argument,
        Message("A nan max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator"));

    REQUIRE_THROWS_MATCHES(ta.propagate_until(10., kw::max_delta_t = -1), std::invalid_argument,
                           Message("A non-positive max_delta_t was passed to the propagate_until() function of an "
                                   "adaptive Taylor integrator"));

    ta.set_time(std::numeric_limits<double>::lowest());

    REQUIRE_THROWS_MATCHES(ta.propagate_until(std::numeric_limits<double>::max()), std::invalid_argument,
                           Message("The final time passed to the propagate_until() function of an adaptive Taylor "
                                   "integrator results in an overflow condition"));

    ta.set_time(0.);

    // Propagate forward in time limiting the timestep size and passing in a callback.
    auto counter = 0ul;

    auto oc = std::get<0>(ta.propagate_until(
        10., kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        }));
    auto oc_copy = std::get<0>(ta_copy.propagate_until(10.));

    REQUIRE(ta.get_time() == 10.);
    REQUIRE(counter == 100000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 10.);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));

    // Do propagate_for() too.
    oc = std::get<0>(ta.propagate_for(
        10., kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        }));
    oc_copy = std::get<0>(ta_copy.propagate_for(10.));

    REQUIRE(ta.get_time() == 20.);
    REQUIRE(counter == 200000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 20.);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));

    // Do backwards in time too.
    oc = std::get<0>(ta.propagate_for(
        -10., kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        }));
    oc_copy = std::get<0>(ta_copy.propagate_for(-10.));

    REQUIRE(ta.get_time() == 10.);
    REQUIRE(counter == 300000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 10.);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));

    oc = std::get<0>(ta.propagate_until(
        0., kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        }));
    oc_copy = std::get<0>(ta_copy.propagate_until(0.));

    REQUIRE(ta.get_time() == 0.);
    REQUIRE(counter == 400000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 0.);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));
}

TEST_CASE("propagate for_until write_tc")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    ta.propagate_until(
        10, kw::callback = [](auto &t) {
            REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta.propagate_until(
        20, kw::write_tc = true, kw::callback = [](auto &t) {
            REQUIRE(!std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    ta.propagate_for(
        10, kw::callback = [](auto &t) {
            REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta.propagate_for(
        20, kw::write_tc = true, kw::callback = [](auto &t) {
            REQUIRE(!std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });
}

TEST_CASE("propagate grid")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};
    auto ta_copy = ta;

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1.}, kw::max_delta_t = std::numeric_limits<double>::quiet_NaN()), std::invalid_argument,
        Message("A nan max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1.}, kw::max_delta_t = -1.), std::invalid_argument,
        Message(
            "A non-positive max_delta_t was passed to the propagate_grid() function of an adaptive Taylor integrator"));

    ta.set_time(std::numeric_limits<double>::lowest());

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()}),
        std::invalid_argument,
        Message("The final time passed to the propagate_grid() function of an adaptive Taylor "
                "integrator results in an overflow condition"));

    ta.set_time(0.);

    // Propagate forward in time limiting the timestep size and passing in a callback.
    auto counter = 0ul;

    auto [oc, _1, _2, _3, out] = ta.propagate_grid(
        {1., 5., 10.}, kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        });
    auto [oc_copy, _4, _5, _6, out_copy] = ta_copy.propagate_grid({1., 5., 10.});

    REQUIRE(ta.get_time() == 10.);
    REQUIRE(counter == 90000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 10.);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    for (auto i = 0u; i < 3u; ++i) {
        REQUIRE(out[2u * i] == approximately(out_copy[2u * i], 1000.));
        REQUIRE(out[2u * i + 1u] == approximately(out_copy[2u * i + 1u], 1000.));
    }

    // Backwards.
    std::tie(oc, _1, _2, _3, out) = ta.propagate_grid(
        {10., 5., 1.}, kw::max_delta_t = 1e-4, kw::callback = [&counter](taylor_adaptive<double> &) {
            ++counter;
            return true;
        });
    std::tie(oc_copy, _4, _5, _6, out_copy) = ta_copy.propagate_grid({10., 5., 1.});

    REQUIRE(ta.get_time() == 1);
    REQUIRE(counter == 180000ul);
    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta_copy.get_time() == 1);
    REQUIRE(oc_copy == taylor_outcome::time_limit);

    for (auto i = 0u; i < 3u; ++i) {
        REQUIRE(out[2u * i] == approximately(out_copy[2u * i], 1000.));
        REQUIRE(out[2u * i + 1u] == approximately(out_copy[2u * i + 1u], 1000.));
    }
}

// Test the stream operator of the outcome enum.
TEST_CASE("outcome stream")
{
    {
        std::ostringstream oss;

        oss << taylor_outcome::success;

        REQUIRE(oss.str() == "taylor_outcome::success");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome::step_limit;

        REQUIRE(oss.str() == "taylor_outcome::step_limit");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome::time_limit;

        REQUIRE(oss.str() == "taylor_outcome::time_limit");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome::err_nf_state;

        REQUIRE(oss.str() == "taylor_outcome::err_nf_state");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome::cb_stop;

        REQUIRE(oss.str() == "taylor_outcome::cb_stop");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome{0};

        REQUIRE(oss.str() == "taylor_outcome::terminal_event_0 (continuing)");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome{42};

        REQUIRE(oss.str() == "taylor_outcome::terminal_event_42 (continuing)");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome{-1};

        REQUIRE(oss.str() == "taylor_outcome::terminal_event_0 (stopping)");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome{-43};

        REQUIRE(oss.str() == "taylor_outcome::terminal_event_42 (stopping)");
    }

    {
        std::ostringstream oss;

        oss << taylor_outcome{static_cast<std::int64_t>(taylor_outcome::cb_stop) - 1};

        REQUIRE(oss.str() == "taylor_outcome::??");
    }
}

// Test the stream operator of the event direction enum.
TEST_CASE("event direction stream")
{
    {
        std::ostringstream oss;

        oss << event_direction{-1};

        REQUIRE(oss.str() == "event_direction::negative");
    }

    {
        std::ostringstream oss;

        oss << event_direction{0};

        REQUIRE(oss.str() == "event_direction::any");
    }

    {
        std::ostringstream oss;

        oss << event_direction{1};

        REQUIRE(oss.str() == "event_direction::positive");
    }

    {
        std::ostringstream oss;

        oss << event_direction{-2};

        REQUIRE(oss.str() == "event_direction::??");
    }

    {
        std::ostringstream oss;

        oss << event_direction{2};

        REQUIRE(oss.str() == "event_direction::??");
    }
}
