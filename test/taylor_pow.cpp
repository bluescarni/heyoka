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
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

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

// Various tests to ensure that the approximated version of pow() is
// being used under certain conditions.
TEST_CASE("taylor pow approx")
{
    auto x = make_vars("x");

    {
        auto ta = taylor_adaptive{{prime(x) = pow(x, -1.5) + pow(x, 1 / 3.)}, {2.}, kw::tol = .1, kw::opt_level = 0};

        REQUIRE(ir_contains(ta, "@llvm.pow"));
        REQUIRE(ir_contains(ta, "@llvm.sqrt"));
    }

    {
        auto ta = taylor_adaptive{std::vector{std::pair{x, pow(par[0], -1.5)}}, {2.}, kw::tol = .1, kw::opt_level = 0};

        REQUIRE(ir_contains(ta, "@llvm.sqrt"));
    }

    {
        auto ta
            = taylor_adaptive{std::vector{std::pair{x, pow(-1.5_dbl, par[0])}}, {2.}, kw::tol = .1, kw::opt_level = 0};

        REQUIRE(!ir_contains(ta, "@llvm.sqrt"));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, -1.5) + pow(x, 1 / 3.)}}, {2.}, kw::tol = .1, kw::opt_level = 0};

        REQUIRE(ir_contains(ta, "@llvm.pow"));
        REQUIRE(ir_contains(ta, "@llvm.sqrt"));
    }

    {
        auto ta = taylor_adaptive{std::vector{std::pair{x, pow(par[0], -1.5)}}, {2.}, kw::tol = .1, kw::opt_level = 0};

        REQUIRE(ir_contains(ta, "@llvm.sqrt"));
    }

    {
        auto ta = taylor_adaptive{std::vector{std::pair{x, pow(-1.5_dbl, par[0])}},
                                  {2.},
                                  kw::tol = .1,
                                  kw::opt_level = 0,
                                  kw::compact_mode = true};

        REQUIRE(!ir_contains(ta, "taylor_c_diff.pow."));
        REQUIRE(ir_contains(ta, "taylor_c_diff.exp."));
        REQUIRE(ir_contains(ta, "taylor_c_diff.log."));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, 2_dbl)}}, {2.}, kw::tol = .1, kw::opt_level = 0, kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_square."));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, .5_dbl)}}, {2.}, kw::tol = .1, kw::opt_level = 0, kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_sqrt."));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, 1.5_dbl)}}, {2.}, kw::tol = .1, kw::opt_level = 0, kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_pos_small_half_3."));
    }

    {
        auto ta = taylor_adaptive{std::vector{std::pair{x, pow(x, -1.5_dbl)}},
                                  {2.},
                                  kw::tol = .1,
                                  kw::opt_level = 0,
                                  kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_neg_small_half_3."));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, 4_dbl)}}, {2.}, kw::tol = .1, kw::opt_level = 0, kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_pos_small_int_4."));
    }

    {
        auto ta = taylor_adaptive{
            std::vector{std::pair{x, pow(x, -4_dbl)}}, {2.}, kw::tol = .1, kw::opt_level = 0, kw::compact_mode = true};

        REQUIRE(ir_contains(ta, "taylor_c_diff.pow_neg_small_int_4."));
    }
}

TEST_CASE("taylor pow")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::pow;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = pow(expression{number{fp_t{3}}}, par[0]), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t{1} / fp_t{3}}};

            // REQUIRE(!ir_contains(ta, "@heyoka.taylor_c_diff_single_iter.pow.num_par"));

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = pow(par[0], expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
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
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[5] == approximately(pow(fp_t{3}, fp_t{1} / 3)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = pow(expression{number{fp_t{3}}}, par[1]), prime(y) = x + y},
                                              {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
                                              2,
                                              kw::tol = .5,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{0}, fp_t{0}, fp_t{1} / 3, fp_t{1} / 3}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[5] == approximately(pow(fp_t{3}, fp_t{1} / 3)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
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
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[5] == approximately(pow(fp_t{3}, fp_t{1} / 3)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == fp_t{1} / 2 * (jet[4] + jet[6]));
            REQUIRE(jet[11] == fp_t{1} / 2 * (jet[5] + jet[7]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -4);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[7] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[8] == approximately(pow(fp_t{3}, fp_t{1} / 3)));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == 4);
            REQUIRE(jet[11] == 2);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == fp_t{1} / 2 * (jet[6] + jet[9]));
            REQUIRE(jet[16] == fp_t{1} / 2 * (jet[7] + jet[10]));
            REQUIRE(jet[17] == fp_t{1} / 2 * (jet[8] + jet[11]));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[12] + 2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[13] + 2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[14] + 2 * jet[17])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = pow(par[0], par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars
                                                  = {fp_t{3}, fp_t{3}, fp_t{3}, fp_t{1} / 3, fp_t{1} / 3, fp_t{1} / 3}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -4);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[7] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[8] == approximately(pow(fp_t{3}, fp_t{1} / 3)));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == 4);
            REQUIRE(jet[11] == 2);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == fp_t{1} / 2 * (jet[6] + jet[9]));
            REQUIRE(jet[16] == fp_t{1} / 2 * (jet[7] + jet[10]));
            REQUIRE(jet[17] == fp_t{1} / 2 * (jet[8] + jet[11]));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[12] + 2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[13] + 2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[14] + 2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>(
            {prime(x) = pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), prime(y) = x + y},
            opt_level, high_accuracy, compact_mode, rng, .1f, 20.f);

        // Variable-number tests.
        {
            auto ta
                = taylor_adaptive<fp_t>{{prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                         prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                        {fp_t(2), fp_t(3)},
                                        kw::tol = 1,
                                        kw::high_accuracy = high_accuracy,
                                        kw::compact_mode = compact_mode,
                                        kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff_single_iter.pow.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
        }

        {
            auto ta
                = taylor_adaptive<fp_t>{{prime(x) = pow(y, par[0]),
                                         prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                        {fp_t(2), fp_t(3)},
                                        kw::tol = 1,
                                        kw::high_accuracy = high_accuracy,
                                        kw::compact_mode = compact_mode,
                                        kw::opt_level = opt_level,
                                        kw::pars = {fp_t{3} / 2}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                 prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                {fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 5);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(pow(jet[2], fp_t{3} / 2)));
            REQUIRE(jet[5] == approximately(pow(jet[3], fp_t{3} / 2)));

            REQUIRE(jet[6] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[7] == approximately(pow(jet[1], fp_t{-1} / 3)));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                 prime(y) = pow(x, par[1])},
                {fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{0}, fp_t{0}, fp_t{-1} / 3, fp_t{-1} / 3}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 5);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(pow(jet[2], fp_t{3} / 2)));
            REQUIRE(jet[5] == approximately(pow(jet[3], fp_t{3} / 2)));

            REQUIRE(jet[6] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[7] == approximately(pow(jet[1], fp_t{-1} / 3)));
        }

        {
            auto ta
                = taylor_adaptive<fp_t>{{prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                         prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                        {fp_t{2}, fp_t{3}},
                                        kw::tol = 1,
                                        kw::high_accuracy = high_accuracy,
                                        kw::compact_mode = compact_mode,
                                        kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[1], fp_t{1} / 2) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[2]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                 prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                {fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}},
                2,
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 5);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(pow(jet[2], fp_t{3} / 2)));
            REQUIRE(jet[5] == approximately(pow(jet[3], fp_t{3} / 2)));

            REQUIRE(jet[6] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[7] == approximately(pow(jet[1], fp_t{-1} / 3)));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[2], fp_t{1} / 2) * jet[6]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[3], fp_t{1} / 2) * jet[7]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[4]));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[1], fp_t{-4} / 3) * jet[5]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                 prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                {fp_t{2}, fp_t{5}, fp_t{1}, fp_t{3}, fp_t{4}, fp_t{6}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 5);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(pow(jet[3], fp_t{3} / 2)));
            REQUIRE(jet[7] == approximately(pow(jet[4], fp_t{3} / 2)));
            REQUIRE(jet[8] == approximately(pow(jet[5], fp_t{3} / 2)));

            REQUIRE(jet[9] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[10] == approximately(pow(jet[1], fp_t{-1} / 3)));
            REQUIRE(jet[11] == approximately(pow(jet[2], fp_t{-1} / 3)));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[3], fp_t{1} / 2) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[4], fp_t{1} / 2) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[5], fp_t{1} / 2) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[1], fp_t{-4} / 3) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[2], fp_t{-4} / 3) * jet[8]));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[3], fp_t{-1} / 2) * jet[9] * jet[9]
                                        + pow(jet[3], fp_t{1} / 2) * 2 * jet[15])));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[4], fp_t{-1} / 2) * jet[10] * jet[10]
                                        + pow(jet[4], fp_t{1} / 2) * 2 * jet[16])));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[5], fp_t{-1} / 2) * jet[11] * jet[11]
                                        + pow(jet[5], fp_t{1} / 2) * 2 * jet[17])));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[0], fp_t{-7} / 3) * jet[6] * jet[6]
                                        + pow(jet[0], fp_t{-4} / 3) * 2 * jet[12])));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[1], fp_t{-7} / 3) * jet[7] * jet[7]
                                        + pow(jet[1], fp_t{-4} / 3) * 2 * jet[13])));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[2], fp_t{-7} / 3) * jet[8] * jet[8]
                                        + pow(jet[2], fp_t{-4} / 3) * 2 * jet[14])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = pow(y, par[0]), prime(y) = pow(x, par[1])},
                {fp_t{2}, fp_t{5}, fp_t{1}, fp_t{3}, fp_t{4}, fp_t{6}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::pars = {fp_t{3} / 2, fp_t{3} / 2, fp_t{3} / 2, fp_t{-1} / 3, fp_t{-1} / 3, fp_t{-1} / 3}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 5);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(pow(jet[3], fp_t{3} / 2)));
            REQUIRE(jet[7] == approximately(pow(jet[4], fp_t{3} / 2)));
            REQUIRE(jet[8] == approximately(pow(jet[5], fp_t{3} / 2)));

            REQUIRE(jet[9] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[10] == approximately(pow(jet[1], fp_t{-1} / 3)));
            REQUIRE(jet[11] == approximately(pow(jet[2], fp_t{-1} / 3)));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[3], fp_t{1} / 2) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[4], fp_t{1} / 2) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[5], fp_t{1} / 2) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[1], fp_t{-4} / 3) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[2], fp_t{-4} / 3) * jet[8]));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[3], fp_t{-1} / 2) * jet[9] * jet[9]
                                        + pow(jet[3], fp_t{1} / 2) * 2 * jet[15])));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[4], fp_t{-1} / 2) * jet[10] * jet[10]
                                        + pow(jet[4], fp_t{1} / 2) * 2 * jet[16])));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6 * fp_t{3} / 2
                                     * (fp_t{1} / 2 * pow(jet[5], fp_t{-1} / 2) * jet[11] * jet[11]
                                        + pow(jet[5], fp_t{1} / 2) * 2 * jet[17])));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[0], fp_t{-7} / 3) * jet[6] * jet[6]
                                        + pow(jet[0], fp_t{-4} / 3) * 2 * jet[12])));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[1], fp_t{-7} / 3) * jet[7] * jet[7]
                                        + pow(jet[1], fp_t{-4} / 3) * 2 * jet[13])));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6 * fp_t{-1} / 3
                                     * (fp_t{-4} / 3 * pow(jet[2], fp_t{-7} / 3) * jet[8] * jet[8]
                                        + pow(jet[2], fp_t{-4} / 3) * 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                    prime(y) = pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                   opt_level, high_accuracy, compact_mode, rng, .1f, 20.f);
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

// Tests for the preprocessing that turns pow into exp+log.
TEST_CASE("pow_to_explog")
{
    auto [x, y] = make_vars("x", "y");

    for (auto cm : {false, true}) {
        auto tmp1 = x + pow(y, par[0]);
        auto tmp2 = pow(x, tmp1);
        auto tmp3 = pow(tmp1, par[1]);

        auto ta = taylor_adaptive<double>{
            {prime(x) = (tmp1 * tmp2) / tmp3, prime(y) = tmp1}, {.1, .2}, kw::pars = {1.2, 3.4}, kw::compact_mode = cm};

        REQUIRE(ta.get_decomposition().size() == 16u);

        // Count the number of exps and logs.
        auto n_exp = 0, n_log = 0;
        for (const auto &[ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&ex.value())) {
                n_exp += (fptr->extract<detail::exp_impl>() != nullptr);
                n_log += (fptr->extract<detail::log_impl>() != nullptr);
            }
        }

        REQUIRE(n_exp == 3);
        REQUIRE(n_log == 3);

        // Create an analogous of ta in which the pars have been hard-coded to numbers.
        tmp1 = x + pow(y, 1.2);
        tmp2 = pow(x, tmp1);
        tmp3 = pow(tmp1, 3.4);

        auto ta_hc = taylor_adaptive<double>{
            {prime(x) = (tmp1 * tmp2) / tmp3, prime(y) = tmp1}, {.1, .2}, kw::compact_mode = cm};

        // Compute the Taylor coefficients.
        ta.step(0., true);
        ta_hc.step(0., true);

        // Compare them.
        auto n_tc = ta.get_tc().size();
        for (decltype(n_tc) i = 0; i < n_tc; ++i) {
            REQUIRE(ta.get_tc()[i] == approximately(ta_hc.get_tc()[i], 1000.));
        }
    }
}
