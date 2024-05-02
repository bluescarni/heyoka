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
#include <heyoka/math/sum.hpp>
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

TEST_CASE("taylor sum")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto [x, y] = make_vars("x", "y");

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sum({2_dbl, x, par[0], y}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t{2}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{2 + 2 + 2 + 3}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sum({2_dbl, x, par[0], y}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{-1}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 2 + 2 + 2 + 3);
            REQUIRE(jet[5] == 2 - 2 - 1 - 3);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sum({2_dbl, x, par[0], y}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t{2}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{2 + 2 + 2 + 3}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
            REQUIRE(jet[4] == approximately(fp_t(.5) * (jet[2] + jet[3])));
            REQUIRE(jet[5] == approximately(fp_t(.5) * (jet[2] + jet[3])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sum({2_dbl, x, par[0], y}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{-1}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 2 + 2 + 2 + 3);
            REQUIRE(jet[5] == 2 - 2 - 1 - 3);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == approximately(fp_t(.5) * (jet[4] + jet[6])));
            REQUIRE(jet[9] == approximately(fp_t(.5) * (jet[5] + jet[7])));

            REQUIRE(jet[10] == approximately(fp_t(.5) * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t(.5) * (jet[5] + jet[7])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sum({2_dbl, x, par[0], y}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{2}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t(2), fp_t(-1), fp_t(3)}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 2);

            REQUIRE(jet[6] == 2 + 2 + 2 + 3);
            REQUIRE(jet[7] == 2 - 2 - 1 - 3);
            REQUIRE(jet[8] == 2 + 1 + 3 + 2);

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 3);

            REQUIRE(jet[12] == approximately(fp_t(.5) * (jet[6] + jet[9])));
            REQUIRE(jet[13] == approximately(fp_t(.5) * (jet[7] + jet[10])));
            REQUIRE(jet[14] == approximately(fp_t(.5) * (jet[8] + jet[11])));

            REQUIRE(jet[15] == approximately(fp_t(.5) * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t(.5) * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t(.5) * (jet[8] + jet[11])));

            REQUIRE(jet[18] == approximately((jet[12] + jet[15]) / 3));
            REQUIRE(jet[19] == approximately((jet[13] + jet[16]) / 3));
            REQUIRE(jet[20] == approximately((jet[14] + jet[17]) / 3));

            REQUIRE(jet[21] == approximately((jet[12] + jet[15]) / 3));
            REQUIRE(jet[22] == approximately((jet[13] + jet[16]) / 3));
            REQUIRE(jet[23] == approximately((jet[14] + jet[17]) / 3));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = sum({2_dbl, x, -3_dbl, y}), prime(y) = x + y}, opt_level, high_accuracy,
                                   compact_mode, rng, -10.f, 10.f);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}
