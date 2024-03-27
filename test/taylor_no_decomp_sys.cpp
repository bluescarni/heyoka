// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>
#include <random>
#include <tuple>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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

TEST_CASE("taylor const sys")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = y, prime(y) = x},
                                            {fp_t(2), fp_t(-3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -3);
            REQUIRE(jet[2] == jet[1]);
            REQUIRE(jet[3] == jet[0]);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = y, prime(y) = x},
                                                  {fp_t{2}, fp_t{1}, fp_t{-3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == jet[2]);
            REQUIRE(jet[5] == jet[3]);

            REQUIRE(jet[6] == jet[0]);
            REQUIRE(jet[7] == jet[1]);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = y, prime(y) = x},
                                            {fp_t(2), fp_t(-3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -3);
            REQUIRE(jet[2] == jet[1]);
            REQUIRE(jet[3] == jet[0]);
            REQUIRE(jet[4] == jet[3] / 2);
            REQUIRE(jet[5] == jet[2] / 2);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = y, prime(y) = x},
                                                  {fp_t{2}, fp_t{1}, fp_t{-3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == jet[2]);
            REQUIRE(jet[5] == jet[3]);

            REQUIRE(jet[6] == jet[0]);
            REQUIRE(jet[7] == jet[1]);

            REQUIRE(jet[8] == jet[6] / 2);
            REQUIRE(jet[9] == jet[7] / 2);

            REQUIRE(jet[10] == jet[4] / 2);
            REQUIRE(jet[11] == jet[5] / 2);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = y, prime(y) = x},
                                                  {fp_t{2}, fp_t{1}, fp_t{0}, fp_t{-3}, fp_t{5}, fp_t{4}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 4);

            REQUIRE(jet[6] == jet[3]);
            REQUIRE(jet[7] == jet[4]);
            REQUIRE(jet[8] == jet[5]);

            REQUIRE(jet[9] == jet[0]);
            REQUIRE(jet[10] == jet[1]);
            REQUIRE(jet[11] == jet[2]);

            REQUIRE(jet[12] == jet[9] / 2);
            REQUIRE(jet[13] == jet[10] / 2);
            REQUIRE(jet[14] == jet[11] / 2);

            REQUIRE(jet[15] == jet[6] / 2);
            REQUIRE(jet[16] == jet[7] / 2);
            REQUIRE(jet[17] == jet[8] / 2);

            REQUIRE(jet[18] == fp_t{1} / 6 * jet[15] * 2);
            REQUIRE(jet[19] == approximately(fp_t{1} / 6 * jet[16] * 2));
            REQUIRE(jet[20] == fp_t{1} / 6 * jet[17] * 2);

            REQUIRE(jet[21] == fp_t{1} / 6 * jet[12] * 2);
            REQUIRE(jet[22] == fp_t{1} / 6 * jet[13] * 2);
            REQUIRE(jet[23] == fp_t{1} / 6 * jet[14] * 2);
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = y, prime(y) = x}, opt_level, high_accuracy, compact_mode, rng, -10.f,
                                   10.f);
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
