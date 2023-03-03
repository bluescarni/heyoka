// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

const int ntrials = 100;

using namespace heyoka;
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

// NOTE: these tests are designed to check the increased time accuracy
// after the representation of the time coordinate was switched to
// double-length arithmetic.

TEST_CASE("scalar test")
{
    auto tester = [](auto fp_x) {
        using std::abs;
        using std::sin;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>({prime(x) = v, prime(v) = -x}, {0, 1}, kw::compact_mode = true);

        const auto final_time = fp_t(10000.);

        fp_t err = 0;

        std::uniform_real_distribution rdist(-1e-9, 1e-9);

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);

            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = v0;

            if (i % 2) {
                ta.propagate_until(final_time);
            } else {
                ta.propagate_for(final_time);
            }

            const auto exact = v0 * sin(final_time);
            err += abs((exact - ta.get_state()[0]) / exact);
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(1000)));

        // Backwards.
        err = 0;

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);

            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = v0;

            if (i % 2) {
                ta.propagate_until(-final_time);
            } else {
                ta.propagate_for(-final_time);
            }

            const auto exact = v0 * sin(-final_time);
            err += abs((exact - ta.get_state()[0]) / exact);
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(1000)));

        // Some testing for propagate_grid() too.
        err = 0;

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);

            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = v0;

            const auto grid = std::vector{fp_t{1.}, fp_t{10.}, final_time};

            const auto out = std::get<4>(ta.propagate_grid(grid));

            for (auto j = 0u; j < 3u; ++j) {
                auto t = grid[j];

                const auto exact = v0 * sin(t);
                err += abs((exact - out[2u * j]) / exact);
            }
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(3000)));

        // Backwards.
        err = 0;

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);

            ta.set_time(0);
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = v0;

            const auto grid = std::vector{fp_t{-1.}, fp_t{-10.}, -final_time};

            const auto out = std::get<4>(ta.propagate_grid(grid));

            for (auto j = 0u; j < 3u; ++j) {
                auto t = grid[j];

                const auto exact = v0 * sin(t);
                err += abs((exact - out[2u * j]) / exact);
            }
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(3000)));
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x); });
}

TEST_CASE("batch test")
{
    auto tester = [](auto fp_x) {
        using std::abs;
        using std::sin;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive_batch<fp_t>({prime(x) = v, prime(v) = -x}, {0, 0, 1, 1}, 2u, kw::compact_mode = true);

        auto final_time = std::vector{fp_t(10000.), fp_t(11000.)};

        fp_t err = 0;

        std::uniform_real_distribution rdist(-1e-9, 1e-9);

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);
            const auto v1 = fp_t(1) + rdist(rng);

            ta.set_time({fp_t(0), 0});
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = v0;
            ta.get_state_data()[3] = v1;

            if (i % 2) {
                ta.propagate_until(final_time);
            } else {
                ta.propagate_for(final_time);
            }

            const auto exact0 = v0 * sin(final_time[0]);
            const auto exact1 = v1 * sin(final_time[1]);

            err += abs((exact0 - ta.get_state()[0]) / exact0);
            err += abs((exact1 - ta.get_state()[1]) / exact1);
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(1000)));

        // Backwards.
        err = 0;

        final_time = std::vector{fp_t(-10000.), fp_t(-11000.)};

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);
            const auto v1 = fp_t(1) + rdist(rng);

            ta.set_time({fp_t(0), 0});
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = v0;
            ta.get_state_data()[3] = v1;

            if (i % 2) {
                ta.propagate_until(final_time);
            } else {
                ta.propagate_for(final_time);
            }

            const auto exact0 = v0 * sin(final_time[0]);
            const auto exact1 = v1 * sin(final_time[1]);

            err += abs((exact0 - ta.get_state()[0]) / exact0);
            err += abs((exact1 - ta.get_state()[1]) / exact1);
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(1000)));

        // Some testing for propagate_grid() too.
        err = 0;

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);
            const auto v1 = fp_t(1) + rdist(rng);

            ta.set_time({fp_t(0), 0});
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = v0;
            ta.get_state_data()[3] = v1;

            const auto grid = std::vector{fp_t{1.}, fp_t{2.}, fp_t{10.}, fp_t{20.}, fp_t(10000.), fp_t(11000.)};

            const auto out = ta.propagate_grid(grid);

            for (auto j = 0u; j < 3u; ++j) {
                auto t0 = grid[2u * j];
                auto t1 = grid[2u * j + 1u];

                const auto exact0 = v0 * sin(t0);
                const auto exact1 = v1 * sin(t1);
                err += abs((exact0 - out[4u * j]) / exact0);
                err += abs((exact1 - out[4u * j + 1u]) / exact1);
            }
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(3000)));

        // Backwards.
        err = 0;

        for (auto i = 0; i < ntrials; ++i) {
            const auto v0 = fp_t(1) + rdist(rng);
            const auto v1 = fp_t(1) + rdist(rng);

            ta.set_time({fp_t(0), 0});
            ta.get_state_data()[0] = 0;
            ta.get_state_data()[1] = 0;
            ta.get_state_data()[2] = v0;
            ta.get_state_data()[3] = v1;

            const auto grid = std::vector{fp_t{-1.}, fp_t{-2.}, fp_t{-10.}, fp_t{-20.}, fp_t(-10000.), fp_t(-11000.)};

            const auto out = ta.propagate_grid(grid);

            for (auto j = 0u; j < 3u; ++j) {
                auto t0 = grid[2u * j];
                auto t1 = grid[2u * j + 1u];

                const auto exact0 = v0 * sin(t0);
                const auto exact1 = v1 * sin(t1);
                err += abs((exact0 - out[4u * j]) / exact0);
                err += abs((exact1 - out[4u * j + 1u]) / exact1);
            }
        }

        REQUIRE(err / ntrials == approximately(fp_t(0), fp_t(3000)));
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x); });
}

TEST_CASE("dfloat s11n")
{
    using df_t = detail::dfloat<double>;

    auto df = df_t(1.1) + df_t(1.3);

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << df;
    }

    df = df_t(0.);

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> df;
    }

    REQUIRE(df == df_t(1.1) + df_t(1.3));
}

TEST_CASE("dfloat add test")
{
    using std::abs;

    std::uniform_real_distribution<double> rdist(-1e3, 1e3);

    for (auto i = 0; i < ntrials * 100; ++i) {
        auto a_hi = rdist(rng);
        auto a_lo = rdist(rng);
        if (abs(a_hi) < abs(a_lo)) {
            std::swap(a_hi, a_lo);
        }

        auto b_hi = rdist(rng);
        auto b_lo = rdist(rng);
        if (abs(b_hi) < abs(b_lo)) {
            std::swap(b_hi, b_lo);
        }

        detail::dfloat<double> x(a_hi, a_lo), y(b_hi, b_lo);

        x = detail::normalise(x);
        y = detail::normalise(y);

        // Test commutativity.
        auto res1 = x + y;
        auto res2 = y + x;

        REQUIRE(res1.hi == res2.hi);
        REQUIRE(res1.lo == res2.lo);

        // Test smallness of the low part.
        REQUIRE(res1.hi == res1.hi + res1.lo);
    }
}
