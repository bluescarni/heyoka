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

#include <heyoka/detail/sub.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

auto sub_wrapper(expression a, expression b)
{
    return detail::sub(std::move(a), std::move(b));
}

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

// Test issue when two different c_diff functions would end up mangled with the same name.
TEST_CASE("multimangle")
{
    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{
        {prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y}, {2., 3.}, kw::opt_level = 0, kw::tol = 1.};

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(double{-1}));
    REQUIRE(jet[3] == approximately(double{5}));
}

TEST_CASE("taylor sub")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{-1}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(par[0], 3_dbl), prime(y) = x + y},
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
            REQUIRE(jet[2] == approximately(fp_t{-1}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y},
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
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == approximately(fp_t{5}));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] + jet[3])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y},
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
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == approximately(fp_t{5}));
            REQUIRE(jet[7] == approximately(-fp_t{5}));
            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);
            REQUIRE(jet[10] == approximately(fp_t(.5) * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t(.5) * (jet[5] + jet[7])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{-1}, fp_t{3}, fp_t{2}, fp_t{4}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 2);
            REQUIRE(jet[5] == 4);
            REQUIRE(jet[6] == -1);
            REQUIRE(jet[7] == -1);
            REQUIRE(jet[8] == -1);
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(par[0], par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{-1}, fp_t{3}, fp_t{2}, fp_t{4}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 2);
            REQUIRE(jet[5] == 4);
            REQUIRE(jet[6] == -1);
            REQUIRE(jet[7] == -1);
            REQUIRE(jet[8] == -1);
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = sub_wrapper(2_dbl, 3_dbl), prime(y) = x + y}, opt_level, high_accuracy,
                                   compact_mode, rng, -10.f, 10.f);

        // Variable-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)},
                                            {fp_t(2.), fp_t(3.)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(y, par[0]), prime(y) = sub_wrapper(x, -4_dbl)},
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
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)},
                                                  {fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));

            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, par[1])},
                                                  {fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, fp_t{-4}, fp_t{-4}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));

            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
            REQUIRE(jet[4] == approximately(fp_t(.5) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t(.5) * jet[2]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));
            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
            REQUIRE(jet[8] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[9] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[10] == approximately(fp_t(.5) * jet[4]));
            REQUIRE(jet[11] == approximately(fp_t(.5) * jet[5]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(jet[3] - 2));
            REQUIRE(jet[7] == approximately(jet[4] - 2));
            REQUIRE(jet[8] == approximately(jet[5] - 2));

            REQUIRE(jet[9] == approximately(jet[0] + 4));
            REQUIRE(jet[10] == approximately(jet[1] + 4));
            REQUIRE(jet[11] == approximately(jet[2] + 4));

            REQUIRE(jet[12] == approximately(fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * jet[14]));
        }

        {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(y, par[0]), prime(y) = sub_wrapper(x, par[1])},
                                              {fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}},
                                              3,
                                              kw::tol = .1,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}, fp_t{-4}, fp_t{-4}, fp_t{-4}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(jet[3] - 2));
            REQUIRE(jet[7] == approximately(jet[4] - 2));
            REQUIRE(jet[8] == approximately(jet[5] - 2));

            REQUIRE(jet[9] == approximately(jet[0] + 4));
            REQUIRE(jet[10] == approximately(jet[1] + 4));
            REQUIRE(jet[11] == approximately(jet[2] + 4));

            REQUIRE(jet[12] == approximately(fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * jet[14]));
        }

        compare_batch_scalar<fp_t>({prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(x, -4_dbl)}, opt_level,
                                   high_accuracy, compact_mode, rng, -10.f, 10.f);

        // Number/variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(2 - jet[1]));
            REQUIRE(jet[3] == approximately(-4 - jet[0]));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(par[0], y), prime(y) = sub_wrapper(-4_dbl, x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t(2)}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(2 - jet[1]));
            REQUIRE(jet[3] == approximately(-4 - jet[0]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));

            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(par[1], x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, fp_t{-4}, fp_t{-4}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));

            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(-jet[1] + 2));
            REQUIRE(jet[3] == approximately(-jet[0] - 4));
            REQUIRE(jet[4] == approximately(-fp_t(.5) * jet[3]));
            REQUIRE(jet[5] == approximately(-fp_t(.5) * jet[2]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));
            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
            REQUIRE(jet[8] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[9] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[10] == approximately(-fp_t(.5) * jet[4]));
            REQUIRE(jet[11] == approximately(-fp_t(.5) * jet[5]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(-jet[3] + 2));
            REQUIRE(jet[7] == approximately(-jet[4] + 2));
            REQUIRE(jet[8] == approximately(-jet[5] + 2));

            REQUIRE(jet[9] == approximately(-jet[0] - 4));
            REQUIRE(jet[10] == approximately(-jet[1] - 4));
            REQUIRE(jet[11] == approximately(-jet[2] - 4));

            REQUIRE(jet[12] == approximately(-fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(-fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(-fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(-fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(-1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(-1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(-1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{3} * jet[14]));
        }

        {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(par[0], y), prime(y) = sub_wrapper(par[1], x)},
                                              {fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}},
                                              3,
                                              kw::tol = .1,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}, fp_t{-4}, fp_t{-4}, fp_t{-4}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(-jet[3] + 2));
            REQUIRE(jet[7] == approximately(-jet[4] + 2));
            REQUIRE(jet[8] == approximately(-jet[5] + 2));

            REQUIRE(jet[9] == approximately(-jet[0] - 4));
            REQUIRE(jet[10] == approximately(-jet[1] - 4));
            REQUIRE(jet[11] == approximately(-jet[2] - 4));

            REQUIRE(jet[12] == approximately(-fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(-fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(-fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(-fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(-1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(-1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(-1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{3} * jet[14]));
        }

        compare_batch_scalar<fp_t>({prime(x) = sub_wrapper(2_dbl, y), prime(y) = sub_wrapper(-4_dbl, x)}, opt_level,
                                   high_accuracy, compact_mode, rng, -10.f, 10.f);

        // Variable/variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == jet[0] - jet[1]);
            REQUIRE(jet[3] == -jet[0] + jet[1]);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                                  {fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == jet[0] - jet[2]);
            REQUIRE(jet[5] == jet[1] - jet[3]);

            REQUIRE(jet[6] == -jet[0] + jet[2]);
            REQUIRE(jet[7] == -jet[1] + jet[3]);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == jet[0] - jet[1]);
            REQUIRE(jet[3] == -jet[0] + jet[1]);
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * (jet[2] - jet[3])));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (-jet[2] + jet[3])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                                  {fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == jet[0] - jet[2]);
            REQUIRE(jet[5] == jet[1] - jet[3]);

            REQUIRE(jet[6] == -jet[0] + jet[2]);
            REQUIRE(jet[7] == -jet[1] + jet[3]);

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * (jet[4] - jet[6])));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * (jet[5] - jet[7])));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (-jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (-jet[5] + jet[7])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                                  {fp_t{2}, fp_t{1}, fp_t{3}, fp_t{3}, fp_t{-4}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);
            REQUIRE(jet[2] == 3);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == jet[0] - jet[3]);
            REQUIRE(jet[7] == jet[1] - jet[4]);
            REQUIRE(jet[8] == jet[2] - jet[5]);

            REQUIRE(jet[9] == -jet[0] + jet[3]);
            REQUIRE(jet[10] == -jet[1] + jet[4]);
            REQUIRE(jet[11] == -jet[2] + jet[5]);

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * (jet[6] - jet[9])));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * (jet[7] - jet[10])));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * (jet[8] - jet[11])));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (-jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (-jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (-jet[8] + jet[11])));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * (jet[12] - jet[15])));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * (jet[13] - jet[16])));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * (jet[14] - jet[17])));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * (-jet[12] + jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * (-jet[13] + jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * (-jet[14] + jet[17])));
        }

        compare_batch_scalar<fp_t>({prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)}, opt_level,
                                   high_accuracy, compact_mode, rng, -10.f, 10.f);
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
