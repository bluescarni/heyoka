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
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/pow.hpp>
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

auto square_wrapper(expression e)
{
    return pow(std::move(e), 2.);
}

TEST_CASE("taylor square")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{4}));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = square_wrapper(par[0]), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t{2}}};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_pos_small_int_2.par_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{4}));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(fp_t{4}));
            REQUIRE(jet[5] == approximately(fp_t{4}));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = square_wrapper(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, fp_t{2}, fp_t{2}}};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_pos_small_int_2.par_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == approximately(fp_t{4}));
            REQUIRE(jet[5] == approximately(fp_t{4}));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{4}));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + fp_t{4})));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(fp_t{4}));
            REQUIRE(jet[5] == approximately(fp_t{4}));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + fp_t{4})));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + fp_t{4})));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
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

            REQUIRE(jet[6] == approximately((fp_t{4})));
            REQUIRE(jet[7] == approximately((fp_t{4})));
            REQUIRE(jet[8] == approximately((fp_t{4})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + (fp_t{4}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + (fp_t{4}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + (fp_t{4}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = square_wrapper(par[0]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}}};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_pos_small_int_2.par_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == approximately((fp_t{4})));
            REQUIRE(jet[7] == approximately((fp_t{4})));
            REQUIRE(jet[8] == approximately((fp_t{4})));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + (fp_t{4}))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + (fp_t{4}))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + (fp_t{4}))));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = square_wrapper(expression{number{fp_t(2)}}), prime(y) = x + y},
                                   opt_level, high_accuracy, compact_mode, rng, .1f, 20.f);

        // Variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_square.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately((fp_t{9})));
            REQUIRE(jet[3] == approximately((fp_t{4})));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_square.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately((fp_t{9})));
            REQUIRE(jet[5] == approximately((fp_t{25})));

            REQUIRE(jet[6] == approximately((fp_t{4})));
            REQUIRE(jet[7] == approximately((fp_t{16})));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_square.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately((fp_t{9})));
            REQUIRE(jet[3] == approximately((fp_t{4})));
            REQUIRE(jet[4] == approximately(jet[1] * jet[3]));
            REQUIRE(jet[5] == approximately(jet[0] * jet[2]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_square.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately((fp_t{9})));
            REQUIRE(jet[5] == approximately((fp_t{25})));

            REQUIRE(jet[6] == approximately((fp_t{4})));
            REQUIRE(jet[7] == approximately((fp_t{16})));

            REQUIRE(jet[8] == approximately(jet[2] * jet[6]));
            REQUIRE(jet[9] == approximately(jet[3] * jet[7]));

            REQUIRE(jet[10] == approximately(jet[0] * jet[4]));
            REQUIRE(jet[11] == approximately(jet[1] * jet[5]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)},
                                                  {fp_t{2}, fp_t{4}, fp_t{3}, fp_t{3}, fp_t{5}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            if (compact_mode && opt_level == 0u) {
                REQUIRE(ir_contains(ta, "pow_square.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);
            REQUIRE(jet[2] == 3);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately((fp_t{9})));
            REQUIRE(jet[7] == approximately((fp_t{25})));
            REQUIRE(jet[8] == approximately((fp_t{36})));

            REQUIRE(jet[9] == approximately((fp_t{4})));
            REQUIRE(jet[10] == approximately((fp_t{16})));
            REQUIRE(jet[11] == approximately((fp_t{9})));

            REQUIRE(jet[12] == approximately(jet[3] * jet[9]));
            REQUIRE(jet[13] == approximately(jet[4] * jet[10]));
            REQUIRE(jet[14] == approximately(jet[5] * jet[11]));

            REQUIRE(jet[15] == approximately(jet[0] * jet[6]));
            REQUIRE(jet[16] == approximately(jet[1] * jet[7]));
            REQUIRE(jet[17] == approximately(jet[2] * jet[8]));

            REQUIRE(jet[18] == approximately(fp_t{1} / 3 * (jet[9] * jet[9] + jet[3] * 2 * jet[15])));
            REQUIRE(jet[19] == approximately(fp_t{1} / 3 * (jet[10] * jet[10] + jet[4] * 2 * jet[16])));
            REQUIRE(jet[20] == approximately(fp_t{1} / 3 * (jet[11] * jet[11] + jet[5] * 2 * jet[17])));

            REQUIRE(jet[21] == approximately(fp_t{1} / 3 * (jet[6] * jet[6] + jet[0] * 2 * jet[12])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 3 * (jet[7] * jet[7] + jet[1] * 2 * jet[13])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 3 * (jet[8] * jet[8] + jet[2] * 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = square_wrapper(y), prime(y) = square_wrapper(x)}, opt_level,
                                   high_accuracy, compact_mode, rng, .1f, 20.f);
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
