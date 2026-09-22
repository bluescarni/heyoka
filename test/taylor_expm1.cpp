// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
    using std::exp;
    using std::expm1;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, s] = make_vars("x", "s");

                taylor_adaptive<double> ta0({prime(x) = expm1(x) + x}, {-1.}, kw::high_accuracy = ha,
                                            kw::compact_mode = cm, kw::opt_level = opt_level);
                // NOTE: s stands for expm1(x), hence s' = exp(x)*x' = exp(x)*(s + x). We write the factor as exp(x)
                // rather than s + 1: the latter is mathematically equivalent but cancels badly for strongly negative x,
                // which would limit the accuracy of the reference solution.
                taylor_adaptive<double> ta1({prime(x) = s + x, prime(s) = exp(x) * (s + x)}, {-1., expm1(-1.)},
                                            kw::high_accuracy = ha, kw::compact_mode = cm, kw::opt_level = opt_level);

                ta0.propagate_until(.5);
                ta1.propagate_until(.5);

                REQUIRE(abs((ta0.get_state()[0] - ta1.get_state()[0]) / ta0.get_state()[0]) < 1e-14);

                const auto v0 = expm1(ta0.get_state()[0]);
                const auto v1 = ta1.get_state()[1];

                REQUIRE(abs((v0 - v1) / v0) < 1e-14);
            }
        }
    }
}

// Test the decomposition path in which the hidden dependency exp(arg) constant-folds into a number and is subsequently
// replaced by a num_identity() in taylor_decompose_replace_numbers().
TEST_CASE("taylor expm1 decompose number")
{
    using std::expm1;

    auto x = "x"_var;

    const auto ex = expression{func{detail::expm1_impl{1_dbl}}};

    for (auto cm : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            auto ta = taylor_adaptive<double>{
                {prime(x) = ex - x}, {0.}, kw::tol = 1., kw::compact_mode = cm, kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 0.);
            REQUIRE(jet[1] == approximately(expm1(1.) - jet[0]));
        }
    }
}

// Test that the hidden dependency exp(arg) introduced by the decomposition of expm1() is CSE-ed together with an
// explicit exp() on the same argument, rather than being duplicated.
TEST_CASE("taylor expm1 cse")
{
    auto [x, y] = make_vars("x", "y");

    auto ta_m1
        = taylor_adaptive<double>{{prime(x) = expm1(x + y), prime(y) = x}, {2., 3.}, kw::opt_level = 0, kw::tol = 1.};

    auto ta_both = taylor_adaptive<double>{
        {prime(x) = exp(x + y) + expm1(x + y), prime(y) = x}, {2., 3.}, kw::opt_level = 0, kw::tol = 1.};

    // NOTE: adding the explicit exp(x + y) introduces exactly one new u variable - the outer sum. The exp() itself must
    // be shared with the hidden dependency of expm1(). Without the sharing the difference would be 2 rather than 1.
    REQUIRE(ta_both.get_decomposition().size() == ta_m1.get_decomposition().size() + 1u);
}

TEST_CASE("taylor expm1")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::exp;
        using std::expm1;

        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expm1(par[0]), prime(y) = x + y},
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
            REQUIRE(jet[2] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[5] == approximately(expm1(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(par[1]), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[5] == approximately(expm1(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + expm1(fp_t{2}))));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[5] == approximately(expm1(fp_t{2})));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + expm1(fp_t{2}))));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + expm1(fp_t{2}))));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[6] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[7] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[8] == approximately(expm1(fp_t{2})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + expm1(fp_t{2}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + expm1(fp_t{2}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + expm1(fp_t{2}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(par[0]), prime(y) = x + y},
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

            REQUIRE(jet[6] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[7] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[8] == approximately(expm1(fp_t{2})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + expm1(fp_t{2}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + expm1(fp_t{2}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + expm1(fp_t{2}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = expm1(expression{number{fp_t(2)}}), prime(y) = x + y}, opt_level,
                                   high_accuracy, compact_mode, rng, .1f, 20.f);

        // Variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expm1(y), prime(y) = expm1(x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(expm1(fp_t{3})));
            REQUIRE(jet[3] == approximately(expm1(fp_t{2})));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(y), prime(y) = expm1(x)},
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

            REQUIRE(jet[4] == approximately(expm1(fp_t{3})));
            REQUIRE(jet[5] == approximately(expm1(fp_t{5})));

            REQUIRE(jet[6] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[7] == approximately(expm1(fp_t{4})));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = expm1(y), prime(y) = expm1(x)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(expm1(fp_t{3})));
            REQUIRE(jet[3] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * exp(fp_t{3}) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * exp(fp_t{2}) * jet[2]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(y), prime(y) = expm1(x)},
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

            REQUIRE(jet[4] == approximately(expm1(fp_t{3})));
            REQUIRE(jet[5] == approximately(expm1(fp_t{5})));

            REQUIRE(jet[6] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[7] == approximately(expm1(fp_t{4})));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * exp(fp_t{3}) * jet[6]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * exp(fp_t{5}) * jet[7]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * exp(fp_t{2}) * jet[4]));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * exp(fp_t{4}) * jet[5]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = expm1(y), prime(y) = expm1(x)},
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

            REQUIRE(jet[6] == approximately(expm1(fp_t{3})));
            REQUIRE(jet[7] == approximately(expm1(fp_t{5})));
            REQUIRE(jet[8] == approximately(expm1(fp_t{6})));

            REQUIRE(jet[9] == approximately(expm1(fp_t{2})));
            REQUIRE(jet[10] == approximately(expm1(fp_t{4})));
            REQUIRE(jet[11] == approximately(expm1(fp_t{3})));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * exp(fp_t{3}) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * exp(fp_t{5}) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * exp(fp_t{6}) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * exp(fp_t{2}) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * exp(fp_t{4}) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * exp(fp_t{3}) * jet[8]));

            REQUIRE(jet[18] == approximately(fp_t{1} / 6 * exp(fp_t{3}) * (jet[9] * jet[9] + 2 * jet[15])));
            REQUIRE(jet[19] == approximately(fp_t{1} / 6 * exp(fp_t{5}) * (jet[10] * jet[10] + 2 * jet[16])));
            REQUIRE(jet[20] == approximately(fp_t{1} / 6 * exp(fp_t{6}) * (jet[11] * jet[11] + 2 * jet[17])));

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * exp(fp_t{2}) * (jet[6] * jet[6] + 2 * jet[12])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * exp(fp_t{4}) * (jet[7] * jet[7] + 2 * jet[13])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * exp(fp_t{3}) * (jet[8] * jet[8] + 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = expm1(y), prime(y) = expm1(x)}, opt_level, high_accuracy, compact_mode,
                                   rng, .1f, 20.f);
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
