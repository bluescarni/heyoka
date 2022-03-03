// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/nbody.hpp>
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

TEST_CASE("dc deep copy")
{
    auto [x, v] = make_vars("x", "v");

    auto ta
        = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, std::vector<double>(46u, 0.), 23u};

    auto ta2 = ta;

    for (unsigned i = 2; i < ta.get_decomposition().size() - 2u; ++i) {
        REQUIRE(std::get<func>(ta.get_decomposition()[i].first.value()).get_ptr()
                != std::get<func>(ta2.get_decomposition()[i].first.value()).get_ptr());
    }
}

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

struct cb_functor_grid {
    cb_functor_grid() = default;
    cb_functor_grid(cb_functor_grid &&) noexcept = default;
    cb_functor_grid(const cb_functor_grid &)
    {
        ++n_copies;
    }
    bool operator()(taylor_adaptive_batch<double> &) const
    {
        REQUIRE(n_copies == n_copies_after);

        return true;
    }
    inline static unsigned n_copies = 0;
    inline static unsigned n_copies_after = 0;
};

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

    // A case in which the initial propagate_until() to bring the system
    // to grid[0] interrupts the integration.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x},
                                       {0., 0.01, 0.02, 0.03, 1., 1.01, 1.02, 1.03},
                                       4,
                                       kw::t_events = {t_event_batch<double>(v - 0.999)}};
    auto out = ta.propagate_grid({10., 10., 10., 10., 100., 100., 100., 100.});
    REQUIRE(out.size() == 16u);
    REQUIRE(std::all_of(out.begin(), out.end(), [](const auto &v) { return std::isnan(v); }));
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{-1}; }));

    ta = taylor_adaptive_batch<double>{
        {prime(x) = v, prime(v) = -x},
        {0., 0.01, 0.02, 0.03, 1., 1.01, 1.02, 1.03},
        4,
        kw::t_events = {t_event_batch<double>(
            v - 0.999, kw::callback = [](auto &, bool, int, std::uint32_t) { return false; })},
    };
    out = ta.propagate_grid({10., 10., 10., 10., 100., 100., 100., 100.});
    REQUIRE(out.size() == 16u);
    REQUIRE(std::all_of(out.begin(), out.end(), [](const auto &v) { return std::isnan(v); }));
    REQUIRE(std::all_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                        [](const auto &t) { return std::get<0>(t) == taylor_outcome{-1}; }));

    // Test the callback is not copied if it is already a std::function.
    ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -x}, {0., 0.01, 0.02, 0.03, 1., 1.01, 1.02, 1.03}, 4};
    std::function<bool(taylor_adaptive_batch<double> &)> f_cb_grid(cb_functor_grid{});
    f_cb_grid.target<cb_functor_grid>()->n_copies_after = f_cb_grid.target<cb_functor_grid>()->n_copies;
    ta.propagate_grid({10., 10., 10., 10., 100., 100., 100., 100.}, kw::callback = f_cb_grid);
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
        ta.set_time(std::vector<double>{}), std::invalid_argument,
        Message("Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified times is 0"));
    REQUIRE_THROWS_MATCHES(
        ta.set_time({1, 2, 3}), std::invalid_argument,
        Message("Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified times is 3"));

    REQUIRE(ta.get_time() == std::vector{0., 0.});

    ta.set_time({1, -2});

    REQUIRE(ta.get_time() == std::vector{1., -2.});

    ta.set_time(-1);
    REQUIRE(ta.get_time() == std::vector{-1., -1.});
    ta.set_time(1);
    REQUIRE(ta.get_time() == std::vector{1., 1.});
}

struct cb_functor_until {
    cb_functor_until() = default;
    cb_functor_until(cb_functor_until &&) noexcept = default;
    cb_functor_until(const cb_functor_until &)
    {
        ++n_copies;
    }
    bool operator()(taylor_adaptive_batch<double> &) const
    {
        REQUIRE(n_copies == n_copies_after);

        return true;
    }
    inline static unsigned n_copies = 0;
    inline static unsigned n_copies_after = 0;
};

struct cb_functor_for {
    cb_functor_for() = default;
    cb_functor_for(cb_functor_for &&) noexcept = default;
    cb_functor_for(const cb_functor_for &)
    {
        ++n_copies;
    }
    bool operator()(taylor_adaptive_batch<double> &) const
    {
        REQUIRE(n_copies == n_copies_after);

        return true;
    }
    inline static unsigned n_copies = 0;
    inline static unsigned n_copies_after = 0;
};

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

    // Scalar input time.
    auto ta_copy2 = ta, ta_copy3 = ta;
    ta_copy2.propagate_until(20.);
    ta_copy3.propagate_until({20., 20.});
    REQUIRE(ta_copy2.get_state() == ta_copy3.get_state());

    // Try also with max_delta_t.
    ta_copy2.propagate_until(30., kw::max_delta_t = std::vector{1e-4, 5e-5});
    ta_copy3.propagate_until({30., 30.}, kw::max_delta_t = std::vector{1e-4, 5e-5});
    REQUIRE(ta_copy2.get_state() == ta_copy3.get_state());

    // Do propagate_for() too.
    ta.propagate_for({10., 11.}, kw::max_delta_t = std::vector{1e-4, 5e-5}, kw::callback = cb);
    ta_copy.propagate_for({10., 11.});

    // Scalar input time.
    ta_copy2.propagate_for(20.);
    ta_copy3.propagate_for({20., 20.});
    REQUIRE(ta_copy2.get_state() == ta_copy3.get_state());

    // Try also with max_delta_t.
    ta_copy2.propagate_for(30., kw::max_delta_t = std::vector{1e-4, 5e-5});
    ta_copy3.propagate_for({30., 30.}, kw::max_delta_t = std::vector{1e-4, 5e-5});
    REQUIRE(ta_copy2.get_state() == ta_copy3.get_state());

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

    // Try with scalar max_delta_t.
    ta_copy = ta;
    ta.propagate_until({10., 11.}, kw::max_delta_t = {1e-4, 1e-4});
    ta_copy.propagate_until({10., 11.}, kw::max_delta_t = 1e-4);
    REQUIRE(ta.get_propagate_res() == ta_copy.get_propagate_res());

    ta.propagate_for({10., 11.}, kw::max_delta_t = {1e-4, 1e-4});
    ta_copy.propagate_for({10., 11.}, kw::max_delta_t = 1e-4);
    REQUIRE(ta.get_propagate_res() == ta_copy.get_propagate_res());

    // Test the callback is not copied if it is already a std::function.
    std::function<bool(taylor_adaptive_batch<double> &)> f_cb_until(cb_functor_until{});
    f_cb_until.target<cb_functor_until>()->n_copies_after = f_cb_until.target<cb_functor_until>()->n_copies;
    ta.propagate_until(20., kw::callback = f_cb_until);

    std::function<bool(taylor_adaptive_batch<double> &)> f_cb_for(cb_functor_for{});
    f_cb_for.target<cb_functor_for>()->n_copies_after = f_cb_for.target<cb_functor_for>()->n_copies;
    ta.propagate_for(10., kw::callback = f_cb_for);
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

    // Test also with scalar max_delta_t.
    ta_copy = ta;
    out = ta.propagate_grid({1., 1.5, 5., 5.6, 10., 11.}, kw::max_delta_t = std::vector{1e-4, 1e-4});
    out_copy = ta_copy.propagate_grid({1., 1.5, 5., 5.6, 10., 11.}, kw::max_delta_t = 1e-4);

    REQUIRE(out == out_copy);
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

        REQUIRE(std::all_of(res.begin() + 4, res.end(), [](double x) { return std::isnan(x); }));

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

    // Check that stopping via cb still processes the grid points
    // within the last taken step.
    {
        auto res = ta.propagate_grid(
            {10., 10.1, 10. + 1e-6, 21.1, 10. + 2e-6, 32.1}, kw::callback = [](auto &) { return false; });

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::cb_stop);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::cb_stop);

        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1u);
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1u);

        REQUIRE(res.size() == 12u);
        REQUIRE(!std::isnan(res[0]));
        REQUIRE(!std::isnan(res[1]));
        REQUIRE(!std::isnan(res[2]));
        REQUIRE(!std::isnan(res[3]));
        REQUIRE(!std::isnan(res[4]));
        REQUIRE(std::isnan(res[5]));
        REQUIRE(!std::isnan(res[6]));
        REQUIRE(std::isnan(res[7]));
        REQUIRE(!std::isnan(res[8]));
        REQUIRE(std::isnan(res[9]));
        REQUIRE(!std::isnan(res[10]));
        REQUIRE(std::isnan(res[11]));
    }
}

// Test excessive number of params provided upon construction
// of the integrator.
TEST_CASE("param too many")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    REQUIRE_THROWS_MATCHES((void)(taylor_adaptive_batch<double>{{prime(x) = v + par[0], prime(v) = -9.8 * sin(x)},
                                                                {0.05, 0.06, 0.025, 0.026},
                                                                2u,
                                                                kw::pars = std::vector{1., 2., 3.}}),
                           std::invalid_argument,
                           Message("Excessive number of parameter values passed to the constructor of an adaptive "
                                   "Taylor integrator in batch mode: 3 parameter values were passed, but the ODE "
                                   "system contains only 1 parameters "
                                   "(in batches of 2)"));

    REQUIRE_THROWS_MATCHES((void)(taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                                {0.05, 0.06, 0.025, 0.026},
                                                                2u,
                                                                kw::pars = std::vector{1., 2., 3.},
                                                                kw::t_events = {t_event_batch<double>(v - par[0])}}),
                           std::invalid_argument,
                           Message("Excessive number of parameter values passed to the constructor of an adaptive "
                                   "Taylor integrator in batch mode: 3 parameter values were passed, but the ODE "
                                   "system contains only 1 parameters "
                                   "(in batches of 2)"));
}

// Test case for bug: parameters in event equations are ignored
// in the determination of the total number of params in a system.
TEST_CASE("param deduction from events")
{
    auto [x, v] = make_vars("x", "v");

    // Terminal events.
    {
        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                {0.05, 0.06, 0.025, 0.026},
                                                2,
                                                kw::t_events = {t_event_batch<double>(v - par[0])}};

        REQUIRE(ta.get_pars().size() == 2u);
    }

    // Non-terminal events.
    {
        auto ta = taylor_adaptive_batch<double>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {0.05, 0.06, 0.025, 0.026},
            2,
            kw::nt_events = {nt_event_batch<double>(v - par[1], [](auto &, double, int, std::uint32_t) {})}};

        REQUIRE(ta.get_pars().size() == 4u);
    }

    // Both.
    {
        auto ta = taylor_adaptive_batch<double>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {0.05, 0.06, 0.025, 0.026},
            2,
            kw::t_events = {t_event_batch<double>(v - par[10])},
            kw::nt_events = {nt_event_batch<double>(v - par[1], [](auto &, double, int, std::uint32_t) {})}};

        REQUIRE(ta.get_pars().size() == 22u);
    }
}

struct s11n_nt_cb {
    template <typename I>
    void operator()(I &, double, int, std::uint32_t) const
    {
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(s11n_nt_cb, void, taylor_adaptive_batch<double> &, double, int, std::uint32_t);

struct s11n_t_cb {
    template <typename T>
    bool operator()(T &, bool, int, std::uint32_t) const
    {
        return false;
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(s11n_t_cb, bool, taylor_adaptive_batch<double> &, bool, int, std::uint32_t);

template <typename Oa, typename Ia>
void s11n_test_impl()
{
    auto [x, v] = make_vars("x", "v");

    // Test without events.
    {
        auto ta = taylor_adaptive_batch<double>{
            {prime(x) = v, prime(v) = -9.8 * sin(x + par[0])}, {0., 0.01, 0.5, 0.51},    2u,
            kw::pars = std::vector<double>{-1e-4, -1.1e-4},    kw::high_accuracy = true, kw::compact_mode = true};

        REQUIRE(ta.get_tol() == std::numeric_limits<double>::epsilon());
        REQUIRE(ta.get_high_accuracy());
        REQUIRE(ta.get_compact_mode());

        ta.propagate_until({10., 10.1});

        std::stringstream ss;

        {
            Oa oa(ss);

            oa << ta;
        }

        auto ta_copy = ta;
        ta = taylor_adaptive_batch<double>{{prime(x) = x}, {0.123, 0.1231}, 2u, kw::tol = 1e-3};
        REQUIRE(!ta.get_high_accuracy());
        REQUIRE(!ta.get_compact_mode());

        {
            Ia ia(ss);

            ia >> ta;
        }

        REQUIRE(ta.get_llvm_state().get_ir() == ta_copy.get_llvm_state().get_ir());
        REQUIRE(ta.get_decomposition() == ta_copy.get_decomposition());
        REQUIRE(ta.get_order() == ta_copy.get_order());
        REQUIRE(ta.get_tol() == ta_copy.get_tol());
        REQUIRE(ta.get_high_accuracy() == ta_copy.get_high_accuracy());
        REQUIRE(ta.get_compact_mode() == ta_copy.get_compact_mode());
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

    // A test with events.
    {
        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                {0., 0.01, 0.5, 0.51},
                                                2,
                                                kw::t_events = {t_event_batch<double>(v, kw::callback = s11n_t_cb{})},
                                                kw::nt_events = {nt_event_batch<double>(v - par[0], s11n_nt_cb{})},
                                                kw::pars = std::vector<double>{-1e-4, -1e-5}};

        REQUIRE(ta.get_tol() == std::numeric_limits<double>::epsilon());

        // Perform a few steps.
        ta.step();
        ta.step();
        ta.step();
        ta.step();

        std::stringstream ss;

        {
            Oa oa(ss);

            oa << ta;
        }

        auto ta_copy = ta;
        ta = taylor_adaptive_batch<double>{{prime(x) = x},
                                           {0.123, 0.124},
                                           2,
                                           kw::tol = 1e-3,
                                           kw::t_events = {t_event_batch<double>(x, kw::callback = s11n_t_cb{})}};

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

        REQUIRE(ta.get_t_events()[0].get_callback().get_type_index()
                == ta_copy.get_t_events()[0].get_callback().get_type_index());
        REQUIRE(ta.get_t_events()[0].get_cooldown() == ta_copy.get_t_events()[0].get_cooldown());
        REQUIRE(ta.get_te_cooldowns() == ta_copy.get_te_cooldowns());

        REQUIRE(ta.get_nt_events()[0].get_callback().get_type_index()
                == ta_copy.get_nt_events()[0].get_callback().get_type_index());

        // Take a step in ta and in ta_copy.
        ta.step(true);
        ta_copy.step(true);

        REQUIRE(ta.get_time() == ta_copy.get_time());
        REQUIRE(ta.get_state() == ta_copy.get_state());
        REQUIRE(ta.get_tc() == ta_copy.get_tc());
        REQUIRE(ta.get_last_h() == ta_copy.get_last_h());

        ta.update_d_output({-.1, -.1}, true);
        ta_copy.update_d_output({-.1, -.1}, true);

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
        REQUIRE(ta.get_high_accuracy() == false);
        REQUIRE(ta.get_compact_mode() == false);
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("stream output")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v - par[1], prime(v) = -9.8 * sin(x + par[0])},
                                                  {0., 0.01, 0.5, 0.51},
                                                  2u,
                                                  kw::pars = std::vector<fp_t>{-1e-4, -1.1e-4}};

            std::ostringstream oss;

            oss << ta;

            REQUIRE(boost::algorithm::contains(oss.str(), "Tolerance"));
            REQUIRE(boost::algorithm::contains(oss.str(), "Dimension"));
            REQUIRE(boost::algorithm::contains(oss.str(), "Batch size"));
            REQUIRE(boost::algorithm::contains(oss.str(), "Parameters"));
            REQUIRE(!boost::algorithm::contains(oss.str(), "events"));
            REQUIRE(boost::algorithm::contains(oss.str(), "High accuracy"));
            REQUIRE(boost::algorithm::contains(oss.str(), "Compact mode"));
        }

        using t_ev_t = t_event_batch<fp_t>;
        using nt_ev_t = nt_event_batch<fp_t>;

        {
            auto tad = taylor_adaptive_batch<fp_t>{{prime(x) = v - par[1], prime(v) = -9.8 * sin(x + par[0])},
                                                   {0., 0.01, 0.5, 0.51},
                                                   2u,
                                                   kw::t_events = {t_ev_t(x)}};

            std::ostringstream oss;

            oss << tad;

            REQUIRE(!oss.str().empty());
            REQUIRE(boost::algorithm::contains(oss.str(), "N of terminal events"));
            REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
            REQUIRE(!boost::algorithm::contains(oss.str(), "N of non-terminal events"));
        }

        {
            auto tad
                = taylor_adaptive_batch<fp_t>{{prime(x) = v - par[1], prime(v) = -9.8 * sin(x + par[0])},
                                              {0., 0.01, 0.5, 0.51},
                                              2u,
                                              kw::nt_events = {nt_ev_t(x, [](auto &, fp_t, int, std::uint32_t) {})}};

            std::ostringstream oss;

            oss << tad;

            REQUIRE(!oss.str().empty());
            REQUIRE(!boost::algorithm::contains(oss.str(), "N of terminal events"));
            REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
            REQUIRE(boost::algorithm::contains(oss.str(), "N of non-terminal events"));
        }

        {
            auto tad
                = taylor_adaptive_batch<fp_t>{{prime(x) = v - par[1], prime(v) = -9.8 * sin(x + par[0])},
                                              {0., 0.01, 0.5, 0.51},
                                              2u,
                                              kw::t_events = {t_ev_t(x)},
                                              kw::nt_events = {nt_ev_t(x, [](auto &, fp_t, int, std::uint32_t) {})}};

            std::ostringstream oss;

            oss << tad;

            REQUIRE(!oss.str().empty());
            REQUIRE(boost::algorithm::contains(oss.str(), "N of terminal events"));
            REQUIRE(boost::algorithm::contains(oss.str(), ": 1"));
            REQUIRE(boost::algorithm::contains(oss.str(), "N of non-terminal events"));

            std::cout << tad << '\n';
        }
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

// A test to check that vector data passed to the constructor
// is moved into the integrator (and not just copied)
TEST_CASE("taylor move")
{
    auto [x, v] = make_vars("x", "v");

    auto init_state = std::vector{-1., -1.1, 0., 0.1};
    auto pars = std::vector{9.8, 9.9};
    auto tes = std::vector{t_event_batch<double>(v)};
    auto ntes = std::vector{nt_event_batch<double>(v, [](auto &, double, int, std::uint32_t) {})};

    auto s_data = init_state.data();
    auto p_data = pars.data();
    auto tes_data = tes.data();
    auto ntes_data = ntes.data();

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -par[0] * sin(x)},
                                            std::move(init_state),
                                            2,
                                            kw::pars = std::move(pars),
                                            kw::t_events = std::move(tes),
                                            kw::nt_events = std::move(ntes)};

    REQUIRE(s_data == ta.get_state().data());
    REQUIRE(p_data == ta.get_pars().data());
    REQUIRE(tes_data == ta.get_t_events().data());
    REQUIRE(ntes_data == ta.get_nt_events().data());
}

TEST_CASE("events error")
{
    using Catch::Matchers::Message;

    auto sys = make_nbody_sys(2, kw::masses = {1., 0.});

    using t_ev_t = t_event_batch<double>;
    using nt_ev_t = nt_event_batch<double>;

    {
        auto tad
            = taylor_adaptive_batch<double>{sys,
                                            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                            2,
                                            kw::t_events = {t_ev_t("x_0"_var)}};

        REQUIRE(tad.with_events());
        REQUIRE_THROWS_MATCHES(
            tad.reset_cooldowns(2), std::invalid_argument,
            Message("Cannot reset the cooldowns at batch index 2: the batch size for this integrator is only 2"));
    }

    // Check reset cooldowns works when there are no terminal events defined.
    {
        auto tad = taylor_adaptive_batch<double>{
            sys,
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
            2,
            kw::nt_events = {nt_ev_t("x_0"_var, [](auto &, double, int, std::uint32_t) {})}};

        REQUIRE(tad.with_events());
        REQUIRE(std::all_of(tad.get_te_cooldowns().begin(), tad.get_te_cooldowns().end(),
                            [](const auto &v) { return v.empty(); }));
        tad.reset_cooldowns();
        tad.reset_cooldowns(0);
        tad.reset_cooldowns(1);
        REQUIRE_THROWS_MATCHES(
            tad.reset_cooldowns(2), std::invalid_argument,
            Message("Cannot reset the cooldowns at batch index 2: the batch size for this integrator is only 2"));
    }

    {
        auto tad = taylor_adaptive_batch<double>{
            sys, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0}, 2};

        REQUIRE(!tad.with_events());
        REQUIRE_THROWS_MATCHES(tad.get_t_events(), std::invalid_argument,
                               Message("No events were defined for this integrator"));
        REQUIRE_THROWS_MATCHES(tad.get_te_cooldowns(), std::invalid_argument,
                               Message("No events were defined for this integrator"));
        REQUIRE_THROWS_MATCHES(tad.get_te_cooldowns(), std::invalid_argument,
                               Message("No events were defined for this integrator"));
        REQUIRE_THROWS_MATCHES(tad.get_nt_events(), std::invalid_argument,
                               Message("No events were defined for this integrator"));
        REQUIRE_THROWS_MATCHES(tad.reset_cooldowns(), std::invalid_argument,
                               Message("No events were defined for this integrator"));
        REQUIRE_THROWS_MATCHES(tad.reset_cooldowns(2), std::invalid_argument,
                               Message("No events were defined for this integrator"));
    }
}

TEST_CASE("ev inf state")
{
    auto [x] = make_vars("x");

    auto ta = taylor_adaptive_batch<double>{
        {prime(x) = 1_dbl}, {0., 0., 0., 0.}, 4, kw::t_events = {t_event_batch<double>(x - 5.)}};

    ta.get_state_data()[2] = std::numeric_limits<double>::infinity();

    ta.step({10., 10., 10., 10.});

    REQUIRE(std::get<0>(ta.get_step_res()[0]) == taylor_outcome{-1});
    REQUIRE(std::get<0>(ta.get_step_res()[1]) == taylor_outcome{-1});
    REQUIRE(std::get<0>(ta.get_step_res()[2]) == taylor_outcome::err_nf_state);
    REQUIRE(std::get<0>(ta.get_step_res()[3]) == taylor_outcome{-1});
}

TEST_CASE("ev exception callback")
{
    using Catch::Matchers::Message;

    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using nte_t = typename taylor_adaptive_batch<fp_t>::nt_event_t;
    using te_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    {
        auto ta = taylor_adaptive_batch<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
            4,
            kw::nt_events = {nte_t(
                v * v - 1e-10, [](auto &, fp_t, int, std::uint32_t) { throw std::invalid_argument("hello world 0"); })},
            kw::t_events = {te_t(
                v, kw::callback = [](auto &, bool, int, std::uint32_t) -> bool {
                    throw std::invalid_argument("hello world 1");
                })}};

        REQUIRE_THROWS_MATCHES(ta.propagate_until({4., 4., 4., 4.}), std::invalid_argument, Message("hello world 0"));
    }

    {
        auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                              {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                              4,
                                              kw::nt_events
                                              = {nte_t(v * v - 1e-10, [](auto &, fp_t, int, std::uint32_t) {})},
                                              kw::t_events = {te_t(
                                                  v, kw::callback = [](auto &, bool, int, std::uint32_t) -> bool {
                                                      throw std::invalid_argument("hello world 1");
                                                  })}};

        REQUIRE_THROWS_MATCHES(ta.propagate_until({4., 4., 4., 4.}), std::invalid_argument, Message("hello world 1"));
    }
}

TEST_CASE("reset cooldowns")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    using te_t = typename taylor_adaptive_batch<fp_t>::t_event_t;

    auto ta
        = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                      {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                      4,
                                      kw::t_events = {te_t(
                                          v, kw::callback = [](auto &, bool, int, std::uint32_t) { return false; })}};

    ta.propagate_until({100., 100., 100., 100.});

    REQUIRE(std::any_of(ta.get_te_cooldowns().begin(), ta.get_te_cooldowns().end(), [](const auto &v) {
        return std::any_of(v.begin(), v.end(), [](const auto &x) { return static_cast<bool>(x); });
    }));

    ta.reset_cooldowns();

    REQUIRE(std::all_of(ta.get_te_cooldowns().begin(), ta.get_te_cooldowns().end(), [](const auto &v) {
        return std::all_of(v.begin(), v.end(), [](const auto &x) { return !static_cast<bool>(x); });
    }));
}

TEST_CASE("copy semantics")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {0., 0., 0.5, 0.5},
                                          2,
                                          kw::t_events = {t_event_batch<fp_t>(v, kw::callback = s11n_t_cb{})},
                                          kw::nt_events = {nt_event_batch<fp_t>(v - par[0], s11n_nt_cb{})},
                                          kw::pars = std::vector<fp_t>{-1e-4, -1e-4},
                                          kw::high_accuracy = true,
                                          kw::compact_mode = true,
                                          kw::tol = 1e-11};

    auto ta_copy = ta;

    REQUIRE(ta_copy.get_nt_events().size() == 1u);
    REQUIRE(ta_copy.get_t_events().size() == 1u);
    REQUIRE(ta_copy.get_tol() == ta.get_tol());
    REQUIRE(ta_copy.get_high_accuracy() == ta.get_high_accuracy());
    REQUIRE(ta_copy.get_compact_mode() == ta.get_compact_mode());

    ta_copy = taylor_adaptive_batch<fp_t>{};
    ta_copy = ta;

    REQUIRE(ta_copy.get_nt_events().size() == 1u);
    REQUIRE(ta_copy.get_t_events().size() == 1u);
    REQUIRE(ta_copy.get_tol() == ta.get_tol());
    REQUIRE(ta_copy.get_high_accuracy() == ta.get_high_accuracy());
    REQUIRE(ta_copy.get_compact_mode() == ta.get_compact_mode());
}

// Test case for the propagate_*() functions not considering
// the last step in the step count if the integration is stopped
// by a terminal event.
TEST_CASE("propagate step count te stop bug")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {0., 0., 0.5, 0.5001},
                                          2,
                                          kw::t_events = {t_event_batch<fp_t>(x - 1e-6)}};

    ta.propagate_until({10., 10.});

    REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1u);
    REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1u);

    ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                     {0., 0., 0.5, 0.5001},
                                     2,
                                     kw::t_events = {t_event_batch<fp_t>(x - 1e-6)}};

    ta.propagate_grid({0., 0., 1., 1., 2., 2.});

    REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1u);
    REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1u);
}

// Bug: set_time() used to use std::copy(), which results in UB for self-copy.
TEST_CASE("set_time alias bug")
{
    using fp_t = double;

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          {0., 0., 0.5, 0.5001},
                                          2,
                                          kw::t_events = {t_event_batch<fp_t>(x - 1e-6)}};

    ta.set_time(ta.get_time());

    REQUIRE(ta.get_time()[0] == 0.);
    REQUIRE(ta.get_time()[1] == 0.);
}

TEST_CASE("get_set_dtime")
{
    using Catch::Matchers::Message;
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0, 0.01, 0.1, 0.11}, 2};

    ta.step();

    REQUIRE(ta.get_dtime().first[0] != 0);
    REQUIRE(ta.get_dtime().first[1] != 0);
    REQUIRE(ta.get_dtime().second[0] == 0);
    REQUIRE(ta.get_dtime().second[1] == 0);

    REQUIRE(std::is_reference_v<decltype(ta.get_dtime().first)>);
    REQUIRE(std::is_reference_v<decltype(ta.get_dtime().second)>);

    for (auto i = 0; i < 1000; ++i) {
        ta.step();
    }

    // REQUIRE(ta.get_dtime_data().first[0] != 0);
    // REQUIRE(ta.get_dtime_data().first[1] != 0);
    // REQUIRE(ta.get_dtime_data().second[0] != 0);
    // REQUIRE(ta.get_dtime_data().second[1] != 0);

    REQUIRE_THROWS_MATCHES(
        ta.set_dtime(std::vector<double>{}, std::vector<double>{1.}), std::invalid_argument,
        Message("Invalid number of new times specified in a Taylor integrator in batch mode: the batch size is 2, "
                "but the number of specified times is (0, 1)"));

    auto dtm = ta.get_dtime();
    ta.set_dtime(dtm.first, dtm.second);
    REQUIRE(ta.get_dtime() == dtm);

    ta.set_dtime({3., -1}, {4., 5.});
    REQUIRE(ta.get_dtime().first[0] == 7);
    REQUIRE(ta.get_dtime().first[1] == 4);
    REQUIRE(ta.get_dtime().second[0] == 0);
    REQUIRE(ta.get_dtime().second[1] == 0);

    ta.set_dtime({3., -3}, {std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::epsilon()});
    REQUIRE(ta.get_dtime().first[0] == 3);
    REQUIRE(ta.get_dtime().first[1] == -3);
    REQUIRE(ta.get_dtime().second[0] == std::numeric_limits<double>::epsilon());
    REQUIRE(ta.get_dtime().second[1] == std::numeric_limits<double>::epsilon());

    ta.set_dtime(3., 4.);
    REQUIRE(ta.get_dtime().first[0] == 7);
    REQUIRE(ta.get_dtime().first[1] == 7);
    REQUIRE(ta.get_dtime().second[0] == 0);
    REQUIRE(ta.get_dtime().second[1] == 0);

    ta.set_dtime(3., std::numeric_limits<double>::epsilon());
    REQUIRE(ta.get_dtime().first[0] == 3);
    REQUIRE(ta.get_dtime().first[1] == 3);
    REQUIRE(ta.get_dtime().second[0] == std::numeric_limits<double>::epsilon());
    REQUIRE(ta.get_dtime().second[1] == std::numeric_limits<double>::epsilon());
}

// Check the callback is invoked when the integration is stopped
// by a terminal event.
TEST_CASE("callback ste")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using ev_t = typename taylor_adaptive_batch<fp_t>::t_event_t;
        auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                              {-1, -0.0001, -1, -1, 0.025, 0.026, 0.027, 0.028},
                                              4,
                                              kw::t_events = {ev_t(x)}};

        int n_invoked = 0;
        auto pcb = [&n_invoked](auto &) {
            ++n_invoked;

            return true;
        };

        ta.propagate_until(10, kw::callback = pcb);

        REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::success);
        REQUIRE(std::get<3>(ta.get_propagate_res()[0]) == 1);
        REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome{-1});
        REQUIRE(std::get<3>(ta.get_propagate_res()[1]) == 1);
        REQUIRE(std::get<0>(ta.get_propagate_res()[2]) == taylor_outcome::success);
        REQUIRE(std::get<3>(ta.get_propagate_res()[2]) == 1);
        REQUIRE(std::get<0>(ta.get_propagate_res()[3]) == taylor_outcome::success);
        REQUIRE(std::get<3>(ta.get_propagate_res()[3]) == 1);

        REQUIRE(n_invoked == 1);
    };

    tuple_for_each(fp_types, tester);
}

// Test for an issue arising when propagate_grid()
// would not write the TCs necessary to the computation
// of the dense output for some grid points.
TEST_CASE("propagate_grid tc issue")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive_batch<double>({prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2);

    std::vector t_grid = {-.5, -.5, -.1, -.4999, .1, .1, .2, .2};

    auto out = ta.propagate_grid(t_grid);

    REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome::time_limit);

    for (auto i = 0u; i < 4u; ++i) {
        REQUIRE(out[i * 4u] == approximately(std::sin(t_grid[2u * i])));
        REQUIRE(out[i * 4u + 1u] == approximately(std::sin(t_grid[2u * i + 1u])));
        REQUIRE(out[i * 4u + 2u] == approximately(std::cos(t_grid[2u * i])));
        REQUIRE(out[i * 4u + 3u] == approximately(std::cos(t_grid[2u * i + 1u])));
    }
}

// Test that when propagate_grid() runs into a stopping terminal
// event all the grid points within the last taken step are
// processed.
TEST_CASE("propagate_grid ste")
{
    auto [x, v] = make_vars("x", "v");

    using ev_t = typename taylor_adaptive_batch<double>::t_event_t;

    auto ta = taylor_adaptive_batch<double>({prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, 2,
                                            kw::t_events = {ev_t(heyoka::time - .1)});

    std::vector t_grid = {0., 0., .1 - 2e-6, 10., .1 - 1e-6, 20., .1 + 1e-6, 30.};

    auto res = ta.propagate_grid(t_grid);

    REQUIRE(res.size() == 16u);

    REQUIRE(std::get<0>(ta.get_propagate_res()[0]) == taylor_outcome{-1});
    REQUIRE(std::get<0>(ta.get_propagate_res()[1]) == taylor_outcome{-1});

    REQUIRE(!std::isnan(res[0]));
    REQUIRE(!std::isnan(res[1]));
    REQUIRE(!std::isnan(res[2]));
    REQUIRE(!std::isnan(res[3]));
    REQUIRE(!std::isnan(res[4]));
    REQUIRE(std::isnan(res[5]));
    REQUIRE(!std::isnan(res[6]));
    REQUIRE(std::isnan(res[7]));
    REQUIRE(!std::isnan(res[8]));
    REQUIRE(std::isnan(res[9]));
    REQUIRE(!std::isnan(res[10]));
    REQUIRE(std::isnan(res[11]));
    REQUIRE(std::isnan(res[12]));
    REQUIRE(std::isnan(res[13]));
    REQUIRE(std::isnan(res[14]));
    REQUIRE(std::isnan(res[15]));
}
