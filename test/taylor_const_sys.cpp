// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <tuple>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Helper to ease the removal of mul() in the test code.
auto mul(const expression &a, const expression &b)
{
    return expression{func{detail::prod_impl({a, b})}};
}

TEST_CASE("taylor const sys")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y, z] = make_vars("x", "y", "z");

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expression{number{fp_t{1}}},
                                             prime(y) = expression{number{fp_t{-2}}},
                                             prime(z) = expression{number{fp_t{0}}}},
                                            {fp_t{2}, fp_t{3}, fp_t{4}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = par[0], prime(y) = expression{number{fp_t{-2}}}, prime(z) = expression{number{fp_t{0}}}},
                {fp_t{2}, fp_t{3}, fp_t{4}},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{1}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expression{number{fp_t{1}}},
                                                   prime(y) = expression{number{fp_t{-2}}},
                                                   prime(z) = expression{number{fp_t{0}}}},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -4);

            REQUIRE(jet[6] == 1);
            REQUIRE(jet[7] == 1);

            REQUIRE(jet[8] == -2);
            REQUIRE(jet[9] == -2);

            REQUIRE(jet[10] == 0);
            REQUIRE(jet[11] == 0);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = expression{number{fp_t{1}}}, prime(y) = par[1], prime(z) = expression{number{fp_t{0}}}},
                {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{1}, fp_t{1}, fp_t{-2}, fp_t{-2}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -4);

            REQUIRE(jet[6] == 1);
            REQUIRE(jet[7] == 1);

            REQUIRE(jet[8] == -2);
            REQUIRE(jet[9] == -2);

            REQUIRE(jet[10] == 0);
            REQUIRE(jet[11] == 0);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expression{number{fp_t{1}}},
                                             prime(y) = expression{number{fp_t{-2}}},
                                             prime(z) = expression{number{fp_t{0}}}},
                                            {fp_t{2}, fp_t{3}, fp_t{4}},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
            REQUIRE(jet[6] == 0);
            REQUIRE(jet[7] == 0);
            REQUIRE(jet[8] == 0);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expression{number{fp_t{1}}},
                                                   prime(y) = expression{number{fp_t{-2}}},
                                                   prime(z) = expression{number{fp_t{0}}}},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -4);

            REQUIRE(jet[6] == 1);
            REQUIRE(jet[7] == 1);

            REQUIRE(jet[8] == -2);
            REQUIRE(jet[9] == -2);

            REQUIRE(jet[10] == 0);
            REQUIRE(jet[11] == 0);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);

            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == 0);

            REQUIRE(jet[16] == 0);
            REQUIRE(jet[17] == 0);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                 prime(z) = expression{number{fp_t{0}}}},
                {fp_t{2}, fp_t{-2}, fp_t{0}, fp_t{3}, fp_t{-3}, fp_t{0}, fp_t{4}, fp_t{-4}, fp_t{0}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == 4);
            REQUIRE(jet[7] == -4);
            REQUIRE(jet[8] == 0);

            REQUIRE(jet[9] == 1);
            REQUIRE(jet[10] == 1);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == -2);
            REQUIRE(jet[13] == -2);
            REQUIRE(jet[14] == -2);

            REQUIRE(jet[15] == 0);
            REQUIRE(jet[16] == 0);
            REQUIRE(jet[17] == 0);

            for (auto i = 18u; i < 36u; ++i) {
                REQUIRE(jet[i] == 0);
            }
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = par[0], prime(y) = par[1], prime(z) = par[2]},
                {fp_t{2}, fp_t{-2}, fp_t{0}, fp_t{3}, fp_t{-3}, fp_t{0}, fp_t{4}, fp_t{-4}, fp_t{0}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{1}, fp_t{1}, fp_t{1}, fp_t{-2}, fp_t{-2}, fp_t{-2}, fp_t{0}, fp_t{0}, fp_t{0}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == 4);
            REQUIRE(jet[7] == -4);
            REQUIRE(jet[8] == 0);

            REQUIRE(jet[9] == 1);
            REQUIRE(jet[10] == 1);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == -2);
            REQUIRE(jet[13] == -2);
            REQUIRE(jet[14] == -2);

            REQUIRE(jet[15] == 0);
            REQUIRE(jet[16] == 0);
            REQUIRE(jet[17] == 0);

            for (auto i = 18u; i < 36u; ++i) {
                REQUIRE(jet[i] == 0);
            }
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                    prime(z) = expression{number{fp_t{0}}}},
                                   opt_level, high_accuracy, compact_mode, rng, -10.f, 10.f);
    };

    for (auto cm : {true, false}) {
        for (auto ha : {true, false}) {
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 0, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 1, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 2, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 3, ha, cm); });
        }
    }
}

// A test in which equations have an expression without variables
// at the end.
TEST_CASE("taylor end novars")
{
    using std::cos;
    using std::sin;

    using fp_t = double;

    auto no_vars = expression{mul(2_dbl, 3_dbl)};

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(y) + cos(x) + sin(x) + cos(y) + no_vars,
                                     prime(y) = sin(y) + cos(x) + sin(x) + cos(y) + no_vars},
                                    {2., 3.},
                                    kw::tol = .5};

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1]) + 6));
    REQUIRE(jet[3] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1]) + 6));
    REQUIRE(jet[4]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
    REQUIRE(jet[5]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
}
