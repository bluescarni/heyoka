// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

std::mt19937 rng;

using namespace heyoka;
namespace hy = heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

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

TEST_CASE("propagate for_until")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.06, 0.025, 0.026}, 2u};
    auto ta_copy = ta;

    // Error modes.
    REQUIRE_THROWS_MATCHES(ta.propagate_until({0., std::numeric_limits<double>::infinity()}), std::invalid_argument,
                           Message("A non-finite time was passed to the propagate_until() function of an adaptive "
                                   "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_until({10., 11.}, kw::max_delta_t = std::vector<double>{1}), std::invalid_argument,
        Message("Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified timesteps is 1"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_until({10., 11.}, kw::max_delta_t = {1., 2., 3.}), std::invalid_argument,
        Message("Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified timesteps is 3"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_until({10., 11.}, kw::max_delta_t = {1., std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument,
        Message("A nan max_delta_t was passed to the propagate_until() function of an adaptive "
                "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_until({10., 11.}, kw::max_delta_t = {1., -1.}), std::invalid_argument,
                           Message("A non-positive max_delta_t was passed to the propagate_until() function of an "
                                   "adaptive Taylor integrator in batch mode"));

    ta.set_time({0., std::numeric_limits<double>::lowest()});

    REQUIRE_THROWS_MATCHES(
        ta.propagate_until({10., std::numeric_limits<double>::max()}, kw::max_delta_t = std::vector<double>{}),
        std::invalid_argument,
        Message("The final time passed to the propagate_until() function of an adaptive Taylor "
                "integrator in batch mode results in an overflow condition"));

    ta.set_time({0., 0.});

    // Propagate forward in time limiting the timestep size and passing in a callback.
    auto counter0 = 0ul, counter1 = counter0;

    auto cb = [&counter0, &counter1](taylor_adaptive_batch<double> &t) {
        if (t.get_last_h()[0] != 0) {
            ++counter0;
        }
        if (t.get_last_h()[1] != 0) {
            ++counter1;
        }

        return true;
    };

    ta.propagate_until({10., 11.}, kw::max_delta_t = {1e-4, 5e-5}, kw::callback = cb);
    ta_copy.propagate_until({10., 11.});

    REQUIRE(ta.get_time() == std::vector{10., 11.});
    REQUIRE(counter0 == 100000ul);
    REQUIRE(counter1 == 220000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{10., 11.});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));
    REQUIRE(ta.get_state()[2] == approximately(ta_copy.get_state()[2], 1000.));
    REQUIRE(ta.get_state()[3] == approximately(ta_copy.get_state()[3], 1000.));

    // Do propagate_for() too.
    ta.propagate_for({10., 11.}, kw::max_delta_t = std::vector{1e-4, 5e-5}, kw::callback = cb);
    ta_copy.propagate_for({10., 11.});

    REQUIRE(ta.get_time() == std::vector{20., 22.});
    REQUIRE(counter0 == 200000ul);
    REQUIRE(counter1 == 440000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{20., 22.});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));
    REQUIRE(ta.get_state()[2] == approximately(ta_copy.get_state()[2], 1000.));
    REQUIRE(ta.get_state()[3] == approximately(ta_copy.get_state()[3], 1000.));

    // Do backwards in time too.
    ta.propagate_for({-10., -11.}, kw::max_delta_t = std::vector{1e-4, 5e-5}, kw::callback = cb);
    ta_copy.propagate_for({-10., -11.});

    REQUIRE(ta.get_time() == std::vector{10., 11.});
    REQUIRE(counter0 == 300000ul);
    REQUIRE(counter1 == 660000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{10., 11.});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));
    REQUIRE(ta.get_state()[2] == approximately(ta_copy.get_state()[2], 1000.));
    REQUIRE(ta.get_state()[3] == approximately(ta_copy.get_state()[3], 1000.));

    ta.propagate_until({0., 0.}, kw::max_delta_t = {1e-4, 5e-5}, kw::callback = cb);
    ta_copy.propagate_until({0., 0.});

    REQUIRE(ta.get_time() == std::vector{0., 0.});
    REQUIRE(counter0 == 400000ul);
    REQUIRE(counter1 == 880000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{0., 0.});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], 1000.));
    REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], 1000.));
    REQUIRE(ta.get_state()[2] == approximately(ta_copy.get_state()[2], 1000.));
    REQUIRE(ta.get_state()[3] == approximately(ta_copy.get_state()[3], 1000.));
}

TEST_CASE("propagate for_until write_tc")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.06, 0.025, 0.026}, 2};

    ta.propagate_until(
        {10., 11.}, kw::callback = [](auto &t) {
            REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta.propagate_until(
        {20., 21.}, kw::write_tc = true, kw::callback = [](auto &t) {
            REQUIRE(!std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.06, 0.025, 0.026}, 2};

    ta.propagate_for(
        {10., 11.}, kw::callback = [](auto &t) {
            REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });

    ta.propagate_for(
        {20., 21.}, kw::write_tc = true, kw::callback = [](auto &t) {
            REQUIRE(!std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &x) { return x == 0.; }));
            return true;
        });
}

TEST_CASE("propagate grid 2")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.06, 0.025, 0.026}, 2u};
    auto ta_copy = ta;

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({10., 11.}, kw::max_delta_t = std::vector<double>{1}), std::invalid_argument,
        Message("Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified timesteps is 1"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({10., 11.}, kw::max_delta_t = {1., 2., 3.}), std::invalid_argument,
        Message("Invalid number of max timesteps specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified timesteps is 3"));
    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({10., 11.}, kw::max_delta_t = {1., std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument,
        Message("A nan max_delta_t was passed to the propagate_grid() function of an adaptive "
                "Taylor integrator in batch mode"));
    REQUIRE_THROWS_MATCHES(ta.propagate_grid({10., 11.}, kw::max_delta_t = {1., -1.}), std::invalid_argument,
                           Message("A non-positive max_delta_t was passed to the propagate_grid() function of an "
                                   "adaptive Taylor integrator in batch mode"));

    ta.set_time({0., std::numeric_limits<double>::lowest()});

    REQUIRE_THROWS_MATCHES(
        ta.propagate_grid({0., std::numeric_limits<double>::lowest(), 1., std::numeric_limits<double>::max()},
                          kw::max_delta_t = std::vector<double>{}),
        std::invalid_argument,
        Message("The final time passed to the propagate_grid() function of an adaptive Taylor "
                "integrator in batch mode results in an overflow condition"));

    ta.set_time({0., 0.});

    // Propagate forward in time limiting the timestep size and passing in a callback.
    auto counter0 = 0ul, counter1 = counter0;

    auto cb = [&counter0, &counter1](taylor_adaptive_batch<double> &t) {
        if (t.get_last_h()[0] != 0) {
            ++counter0;
        }
        if (t.get_last_h()[1] != 0) {
            ++counter1;
        }

        return true;
    };

    auto out
        = ta.propagate_grid({1., 1.5, 5., 5.6, 10., 11.}, kw::max_delta_t = std::vector{1e-4, 5e-5}, kw::callback = cb);
    auto out_copy = ta_copy.propagate_grid({1., 1.5, 5., 5.6, 10., 11.});

    REQUIRE(ta.get_time() == std::vector{10., 11.});
    REQUIRE(counter0 == 90000ul);
    REQUIRE(counter1 == 190000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{10., 11.});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    for (auto i = 0u; i < 3u; ++i) {
        REQUIRE(out[4u * i + 0u] == approximately(out_copy[4u * i + 0u], 1000.));
        REQUIRE(out[4u * i + 1u] == approximately(out_copy[4u * i + 1u], 1000.));
        REQUIRE(out[4u * i + 2u] == approximately(out_copy[4u * i + 2u], 1000.));
        REQUIRE(out[4u * i + 3u] == approximately(out_copy[4u * i + 3u], 1000.));
    }

    // Do backward in time too.
    out = ta.propagate_grid({10., 11., 5., 5.6, 1., 1.5}, kw::max_delta_t = std::vector{1e-4, 5e-5}, kw::callback = cb);
    out_copy = ta_copy.propagate_grid({10., 11., 5., 5.6, 1., 1.5});

    REQUIRE(ta.get_time() == std::vector{1., 1.5});
    REQUIRE(counter0 == 180000ul);
    REQUIRE(counter1 == 380000ul);
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    REQUIRE(ta_copy.get_time() == std::vector{1., 1.5});
    REQUIRE(std::all_of(ta_copy.get_propagate_res().begin(), ta_copy.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome::time_limit; }));

    for (auto i = 0u; i < 3u; ++i) {
        REQUIRE(out[4u * i + 0u] == approximately(out_copy[4u * i + 0u], 1000.));
        REQUIRE(out[4u * i + 1u] == approximately(out_copy[4u * i + 1u], 1000.));
        REQUIRE(out[4u * i + 2u] == approximately(out_copy[4u * i + 2u], 1000.));
        REQUIRE(out[4u * i + 3u] == approximately(out_copy[4u * i + 3u], 1000.));
    }
}

// Test the interruption of the propagate_*() functions via callback.
TEST_CASE("cb interrupt")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.06, 0.025, 0.026}, 2u};

    // propagate_for/until().
    {
        ta.propagate_until(
            {1., 1.1}, kw::callback = [](auto &) { return false; });

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::cb_stop);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::cb_stop);

        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1u);
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1u);

        REQUIRE(ta.get_time()[0] < 1.);
        REQUIRE(ta.get_time()[1] < 1.);

        auto counter = 0u;
        ta.propagate_for(
            {10., 10.1}, kw::callback = [&counter](auto &) { return counter++ != 5u; });

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::cb_stop);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::cb_stop);

        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 6u);
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 6u);

        REQUIRE(ta.get_time()[0] < 10.);
        REQUIRE(ta.get_time()[1] < 10.);
    }

    // propagate_grid().
    {
        auto res = ta.propagate_grid(
            {10., 10.1, 11., 11.1, 12., 12.1}, kw::callback = [](auto &) { return false; });

        REQUIRE(std::all_of(res.begin() + 4, res.end(), [](double x) { return x == 0; }));

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::cb_stop);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::cb_stop);

        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1u);
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1u);

        REQUIRE(ta.get_time()[0] < 11.);
        REQUIRE(ta.get_time()[1] < 11.);

        auto counter = 0u;
        res = ta.propagate_grid(
            {20., 20.1, 21., 21.1, 32., 32.1}, kw::callback = [&counter](auto &) { return counter++ != 5u; });

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::cb_stop);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::cb_stop);

        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 6u);
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 6u);

        REQUIRE(ta.get_time()[0] < 32.);
        REQUIRE(ta.get_time()[1] < 32.);
    }
}

// Test excessive number of params provided upon construction
// of the integrator.
TEST_CASE("param too many")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    REQUIRE_THROWS_MATCHES(
        (void)(taylor_adaptive_batch<double>{{prime(x) = v + par[0], prime(v) = -9.8 * sin(x)},
                                             {0.05, 0.06, 0.025, 0.026},
                                             2u,
                                             kw::pars = std::vector{1., 2., 3.}}),
        std::invalid_argument,
        Message("Excessive number of parameter values passed to the constructor of an adaptive "
                "Taylor integrator: 3 parameter values were passed, but the ODE system contains only 1 parameters "
                "(in batches of 2)"));
}

template <typename Oa, typename Ia>
void s11n_test_impl()
{
    auto [x, v] = make_vars("x", "v");

    // Test without events.
    {
        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x + par[0])},
                                                {0., 0.01, 0.5, 0.51},
                                                2u,
                                                kw::pars = std::vector<double>{-1e-4, -1.1e-4}};

        REQUIRE(ta.get_tol() == std::numeric_limits<double>::epsilon());

        ta.propagate_until({10., 10.1});

        std::stringstream ss;

        {
            Oa oa(ss);

            oa << ta;
        }

        auto ta_copy = ta;
        ta = taylor_adaptive_batch<double>{{prime(x) = x}, {0.123, 0.1231}, 2u, kw::tol = 1e-3};

        {
            Ia ia(ss);

            ia >> ta;
        }

        REQUIRE(ta.get_llvm_state().get_ir() == ta_copy.get_llvm_state().get_ir());
        REQUIRE(ta.get_decomposition() == ta_copy.get_decomposition());
        REQUIRE(ta.get_order() == ta_copy.get_order());
        REQUIRE(ta.get_tol() == ta_copy.get_tol());
        REQUIRE(ta.get_dim() == ta_copy.get_dim());
        REQUIRE(ta.get_time() == ta_copy.get_time());
        REQUIRE(ta.get_state() == ta_copy.get_state());
        REQUIRE(ta.get_pars() == ta_copy.get_pars());
        REQUIRE(ta.get_tc() == ta_copy.get_tc());
        REQUIRE(ta.get_last_h() == ta_copy.get_last_h());
        REQUIRE(ta.get_d_output() == ta_copy.get_d_output());

        REQUIRE(ta.get_step_res() == ta_copy.get_step_res());
        REQUIRE(ta.get_propagate_res() == ta_copy.get_propagate_res());

        // Take a step in ta and in ta_copy.
        ta.step(true);
        ta_copy.step(true);

        REQUIRE(ta.get_time() == ta_copy.get_time());
        REQUIRE(ta.get_state() == ta_copy.get_state());
        REQUIRE(ta.get_tc() == ta_copy.get_tc());
        REQUIRE(ta.get_last_h() == ta_copy.get_last_h());
        REQUIRE(ta.get_step_res() == ta_copy.get_step_res());

        ta.update_d_output({-.1, -.11}, true);
        ta_copy.update_d_output({-.1, -.11}, true);

        REQUIRE(ta.get_d_output() == ta_copy.get_d_output());
    }
}

TEST_CASE("s11n")
{
    s11n_test_impl<boost::archive::binary_oarchive, boost::archive::binary_iarchive>();
}

TEST_CASE("def ctor")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        taylor_adaptive_batch<fp_t> ta;

        REQUIRE(ta.get_state() == std::vector{fp_t(0)});
        REQUIRE(ta.get_time() == std::vector{fp_t(0)});
        REQUIRE(ta.get_batch_size() == 1u);
    };

    tuple_for_each(fp_types, tester);
}

#if defined(HEYOKA_ARCH_PPC)

TEST_CASE("ppc long double")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    REQUIRE_THROWS_MATCHES((taylor_adaptive_batch<long double>{
                               {prime(x) = v, prime(v) = -9.8l * sin(x)}, {0.05l, 0.06l, 0.025l, 0.026l}, 2u}),
                           not_implemented_error, Message("'long double' computations are not supported on PowerPC"));
}

#endif
