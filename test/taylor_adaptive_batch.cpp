// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

std::mt19937 rng;

using namespace heyoka;
namespace hy = heyoka;
using namespace heyoka_test;

TEST_CASE("batch consistency")
{
    auto [x, v] = make_vars("x", "v");

    const auto batch_size = 4;

    std::vector<double> state(2 * batch_size), pars(1 * batch_size);

    auto s_arr = xt::adapt(state.data(), {2, batch_size});
    auto p_arr = xt::adapt(pars.data(), {1, batch_size});

    xt::view(s_arr, 0, xt::all()) = xt::xarray<double>{0.01, 0.02, 0.03, 0.04};
    xt::view(s_arr, 1, xt::all()) = xt::xarray<double>{1.85, 1.86, 1.87, 1.88};
    xt::view(p_arr, 0, xt::all()) = xt::xarray<double>{0.10, 0.11, 0.12, 0.13};

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = cos(hy::time) - par[0] * v - sin(x)},
                                            std::move(state),
                                            batch_size,
                                            kw::pars = std::move(pars)};

    auto t_arr = xt::adapt(ta.get_time_data(), {batch_size});
    ta.set_time({0.1, 0.2, 0.3, 0.4});

    std::vector<taylor_adaptive<double>> t_scal;
    for (auto i = 0u; i < batch_size; ++i) {
        t_scal.push_back(taylor_adaptive<double>({prime(x) = v, prime(v) = cos(hy::time) - par[0] * v - sin(x)},
                                                 {s_arr(0, i), s_arr(1, i)}, kw::pars = {p_arr(0, i)}));

        t_scal.back().set_time((i + 1) / 10.);
    }

    ta.propagate_until({20, 21, 22, 23});

    for (auto i = 0u; i < batch_size; ++i) {
        t_scal[i].propagate_until(20 + i);

        REQUIRE(t_scal[i].get_state()[0] == approximately(s_arr(0, i), 1000.));
    }
}

TEST_CASE("propagate grid")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025, 0.051, 0.0251, 0.052, 0.0252, 0.053, 0.0253}, 4u};

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({}), std::invalid_argument,
        Message(
            "Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if the time grid is empty"));

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1.}), std::invalid_argument,
        Message("Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
                "the grid has a size of 1, which is not a multiple of the batch size (4)"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1., 2.}), std::invalid_argument,
        Message("Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
                "the grid has a size of 2, which is not a multiple of the batch size (4)"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({1., 2., 3., 4., 5.}), std::invalid_argument,
        Message("Invalid grid size detected in propagate_grid() for an adaptive Taylor integrator in batch mode: "
                "the grid has a size of 5, which is not a multiple of the batch size (4)"));

    ta.set_time({0., 0., std::numeric_limits<double>::infinity(), 0.});

    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0.}), std::invalid_argument,
                           Message("Cannot invoke propagate_grid() in an adaptive Taylor integrator in batch mode if "
                                   "the current time is not finite"));

    ta.set_time({0., 0., 0., 0.});

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 0., std::numeric_limits<double>::infinity(), 0.}), std::invalid_argument,
        Message(
            "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 0., 0., 0., 0., std::numeric_limits<double>::infinity(), 0., 0.}), std::invalid_argument,
        Message(
            "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 1., 1., -1., 1.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., std::numeric_limits<double>::infinity()}),
        std::invalid_argument,
        Message(
            "A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 1., 1., 1., 1., 2., 0., 0., 2.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 0., 1., 1., 1., 2., 2., 2., 2.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 1., 0., 1., 1., 2., 2., 2., 2.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 1., 1., 1., 0., 2., 2., 2., 2.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 1., 2.}), std::invalid_argument,
                           Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive "
                                   "Taylor integrator in batch mode"));

    // Set an infinity in the state.
    ta.get_state_data()[0] = std::numeric_limits<double>::infinity();

    auto ret = ta.propagate_grid({.2, .2, .2, .2});
    REQUIRE(ret.size() == 8u);
    REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::err_nf_state);
    REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta.get_propagate_res()[2]) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta.get_propagate_res()[3]) == taylor_outcome::time_limit);

    // Reset the integrator.
    ta = taylor_adaptive_batch<double>{
        {prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025, 0.051, 0.0251, 0.052, 0.0252, 0.053, 0.0253}, 4u};

    // Propagate to the initial time.
    ret = ta.propagate_grid({0., 0., 0., 0.});
    REQUIRE(ret.size() == 8u);
    REQUIRE(ret == std::vector{0.05, 0.025, 0.051, 0.0251, 0.052, 0.0252, 0.053, 0.0253});
    for (auto i = 0u; i < 4u; ++i) {
        auto [oc, min_h, max_h, nsteps] = ta.get_propagate_res()[i];

        REQUIRE(oc == taylor_outcome::time_limit);
        REQUIRE(min_h == std::numeric_limits<double>::infinity());
        REQUIRE(max_h == 0);
        REQUIRE(nsteps == 0u);
    }

    // Switch to the harmonic oscillator.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 0., 0., 1., 1.1, 1.2, 1.3}, 4u};

    // Integrate forward over a dense grid from ~0 to ~10.
    std::vector<double> grid;
    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0; j < 4; ++j) {
            grid.push_back(i / 100.);
            grid.back() += j / 10.;
        }
    }

    ret = ta.propagate_grid(grid);

    REQUIRE(ret.size() == 8000ull);

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(ta.get_time()[i] == grid[3996u + i]);
    }

    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            REQUIRE(ret[8u * i + j] == approximately((1 + j / 10.) * std::sin(grid[i * 4u + j]), 10000.));
            REQUIRE(ret[8u * i + j + 4u] == approximately((1 + j / 10.) * std::cos(grid[i * 4u + j]), 10000.));
        }
    }

    // Do the same backwards.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 0., 0., 1., 1.1, 1.2, 1.3}, 4u};
    grid.clear();
    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0; j < 4; ++j) {
            grid.push_back(i / -100.);
            grid.back() += j / -10.;
        }
    }

    ret = ta.propagate_grid(grid);

    REQUIRE(ret.size() == 8000ull);

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(ta.get_time()[i] == grid[3996u + i]);
    }

    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            REQUIRE(ret[8u * i + j] == approximately((1 + j / 10.) * std::sin(grid[i * 4u + j]), 10000.));
            REQUIRE(ret[8u * i + j + 4u] == approximately((1 + j / 10.) * std::cos(grid[i * 4u + j]), 10000.));
        }
    }

    // Random testing.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 0., 0., 1., 1.1, 1.2, 1.3}, 4u};
    std::fill(grid.begin(), grid.begin() + 4, 0.);
    std::uniform_real_distribution<double> rdist(0., .1);
    for (auto i = 1u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            grid[i * 4u + j] = grid[(i - 1u) * 4u + j] + rdist(rng);
        }
    }

    ret = ta.propagate_grid(grid);

    REQUIRE(ret.size() == 8000ull);

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(ta.get_time()[i] == grid[3996u + i]);
    }

    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            REQUIRE(ret[8u * i + j] == approximately((1 + j / 10.) * std::sin(grid[i * 4u + j]), 400000.));
            REQUIRE(ret[8u * i + j + 4u] == approximately((1 + j / 10.) * std::cos(grid[i * 4u + j]), 400000.));
        }
    }

    // Do it backwards too.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0., 0., 0., 1., 1.1, 1.2, 1.3}, 4u};
    std::fill(grid.begin(), grid.begin() + 4, 0.);
    rdist = std::uniform_real_distribution<double>(-.1, 0.);
    for (auto i = 1u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            grid[i * 4u + j] = grid[(i - 1u) * 4u + j] + rdist(rng);
        }
    }

    ret = ta.propagate_grid(grid);

    REQUIRE(ret.size() == 8000ull);

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(std::get<0>(ta.get_propagate_res()[i]) == taylor_outcome::time_limit);
        REQUIRE(ta.get_time()[i] == grid[3996u + i]);
    }

    for (auto i = 0u; i < 1000u; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            REQUIRE(ret[8u * i + j] == approximately((1 + j / 10.) * std::sin(grid[i * 4u + j]), 800000.));
            REQUIRE(ret[8u * i + j + 4u] == approximately((1 + j / 10.) * std::cos(grid[i * 4u + j]), 800000.));
        }
    }
}

// A test to make sure the propagate functions deal correctly
// with trivial dynamics.
TEST_CASE("propagate trivial")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = 1_dbl}, {0, 0, 0.1, 0.1}, 2};

    ta.propagate_for({1.2, 1.3});
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    ta.propagate_until({2.3, 4.5});
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    ta.propagate_grid({5, 6, 7, 8.});
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));
}

TEST_CASE("set time")
{
    using Catch::Matchers::Message;
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = 1_dbl}, {0, 0, 0.1, 0.1}, 2};

    REQUIRE_THROWS_MATCHES(
        ta.set_time({}), std::invalid_argument,
        Message("Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified times is 0"));
    REQUIRE_THROWS_MATCHES(
        ta.set_time({1, 2, 3}), std::invalid_argument,
        Message("Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified times is 3"));

    REQUIRE(ta.get_time() == std::vector{0., 0.});

    ta.set_time({1, -2});

    REQUIRE(ta.get_time() == std::vector{1., -2.});
}
