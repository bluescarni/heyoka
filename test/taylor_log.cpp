// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
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

TEST_CASE("ode test")
{
    using std::abs;
    using std::log;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, s] = make_vars("x", "s");

                taylor_adaptive<double> ta0({prime(x) = log(x) + x}, {1.5}, kw::high_accuracy = ha,
                                            kw::compact_mode = cm, kw::opt_level = opt_level);
                taylor_adaptive<double> ta1({prime(x) = s + x, prime(s) = (s + x) / x}, {1.5, log(1.5)},
                                            kw::high_accuracy = ha, kw::compact_mode = cm, kw::opt_level = opt_level);

                ta0.propagate_until(15.);
                ta1.propagate_until(15.);

                REQUIRE(abs((ta0.get_state()[0] - ta1.get_state()[0]) / ta0.get_state()[0]) < 1e-14);

                const auto v0 = log(ta0.get_state()[0]);
                const auto v1 = ta1.get_state()[1];

                REQUIRE(abs((v0 - v1) / v0) < 1e-14);
            }
        }
    }
}

TEST_CASE("taylor log")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::log;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(log(fp_t{2})));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = log(par[0]), prime(y) = x + y},
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
            REQUIRE(jet[2] == approximately(log(fp_t{2})));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
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

            REQUIRE(jet[4] == approximately(log(fp_t{2})));
            REQUIRE(jet[5] == approximately(log(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, fp_t{2}, fp_t{2}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == approximately(log(fp_t{2})));
            REQUIRE(jet[5] == approximately(log(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(log(fp_t{2})));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + log(fp_t{2}))));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
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

            REQUIRE(jet[4] == approximately(log(fp_t{2})));
            REQUIRE(jet[5] == approximately(log(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + log(fp_t{2}))));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + log(fp_t{2}))));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == approximately(log(fp_t{2})));
            REQUIRE(jet[7] == approximately(log(fp_t{2})));
            REQUIRE(jet[8] == approximately(log(fp_t{2})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + log(fp_t{2}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + log(fp_t{2}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + log(fp_t{2}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(par[0]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == approximately(log(fp_t{2})));
            REQUIRE(jet[7] == approximately(log(fp_t{2})));
            REQUIRE(jet[8] == approximately(log(fp_t{2})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + log(fp_t{2}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + log(fp_t{2}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + log(fp_t{2}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = log(expression{number{fp_t(2)}}), prime(y) = x + y}, opt_level,
                                   high_accuracy, compact_mode, rng, .1f, 20.f);

        // Variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = log(y), prime(y) = log(x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(log(fp_t{3})));
            REQUIRE(jet[3] == approximately(log(fp_t{2})));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(y), prime(y) = log(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(log(fp_t{3})));
            REQUIRE(jet[5] == approximately(log(fp_t{5})));

            REQUIRE(jet[6] == approximately(log(fp_t{2})));
            REQUIRE(jet[7] == approximately(log(fp_t{4})));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = log(y), prime(y) = log(x)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(log(fp_t{3})));
            REQUIRE(jet[3] == approximately(log(fp_t{2})));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * jet[3] / jet[1]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * jet[2] / jet[0]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(y), prime(y) = log(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(log(fp_t{3})));
            REQUIRE(jet[5] == approximately(log(fp_t{5})));

            REQUIRE(jet[6] == approximately(log(fp_t{2})));
            REQUIRE(jet[7] == approximately(log(fp_t{4})));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[6] / jet[2]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[7] / jet[3]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * jet[4] / jet[0]));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * jet[5] / jet[1]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = log(y), prime(y) = log(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{3}, fp_t{5}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);
            REQUIRE(jet[2] == 3);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(log(fp_t{3})));
            REQUIRE(jet[7] == approximately(log(fp_t{5})));
            REQUIRE(jet[8] == approximately(log(fp_t{6})));

            REQUIRE(jet[9] == approximately(log(fp_t{2})));
            REQUIRE(jet[10] == approximately(log(fp_t{4})));
            REQUIRE(jet[11] == approximately(log(fp_t{3})));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[9] / jet[3]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[10] / jet[4]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[11] / jet[5]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * jet[6] / jet[0]));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * jet[7] / jet[1]));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * jet[8] / jet[2]));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6 * (2 * jet[15] * jet[3] - jet[9] * jet[9]) / (jet[3] * jet[3])));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6 * (2 * jet[16] * jet[4] - jet[10] * jet[10]) / (jet[4] * jet[4])));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6 * (2 * jet[17] * jet[5] - jet[11] * jet[11]) / (jet[5] * jet[5])));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6 * (2 * jet[12] * jet[0] - jet[6] * jet[6]) / (jet[0] * jet[0])));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6 * (2 * jet[13] * jet[1] - jet[7] * jet[7]) / (jet[1] * jet[1])));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6 * (2 * jet[14] * jet[2] - jet[8] * jet[8]) / (jet[2] * jet[2])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = log(y), prime(y) = log(x)}, opt_level, high_accuracy, compact_mode, rng,
                                   .1f, 20.f);
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
