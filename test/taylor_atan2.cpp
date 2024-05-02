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
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/atan2.hpp>
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

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("taylor atan2 decompose bug 00")
{

    auto x = "x"_var;

    auto ta = taylor_adaptive<double>({prime(x) = atan2(0_dbl, x) + atan2(x, 0_dbl) + atan2(0_dbl, 0_dbl) - x}, {0.},
                                      kw::tol = 1.);
}

// Test CSE involving hidden dependencies.
TEST_CASE("taylor atan2 test simplifications")
{
    using std::atan2;

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{
        {prime(x) = atan2(x, y), prime(y) = pow(x, 2_dbl) + pow(y, 2_dbl)}, {.2, -.3}, kw::opt_level = 0, kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 6u);

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == .2);
    REQUIRE(jet[1] == -.3);
    REQUIRE(jet[2] == approximately(atan2(jet[0], jet[1])));
    REQUIRE(jet[3] == jet[0] * jet[0] + jet[1] * jet[1]);
}

TEST_CASE("taylor atan2")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::atan2;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        const auto a = fp_t{1} / 3;
        const auto b = 1 + fp_t{3} / fp_t{7};
        const auto c = fp_t{2} / 7;

        // Number-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(expression{number{a}}, par[0]), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {b}};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "@heyoka.taylor_c_diff.atan2.num_par"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(par[0], expression{number{b}}), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {a}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(expression{number{a}}, expression{number{b}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(expression{number{a}}, par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{0}, fp_t{0}, b, b}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = atan2(expression{number{a}}, expression{number{b}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(expression{number{a}}, expression{number{b}}), prime(y) = x + y},
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

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == fp_t{1} / 2 * (jet[4] + jet[6]));
            REQUIRE(jet[11] == fp_t{1} / 2 * (jet[5] + jet[7]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(par[0], par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {a, a, a, b, b, b}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -4);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(atan2(a, b)));
            REQUIRE(jet[7] == approximately(atan2(a, b)));
            REQUIRE(jet[8] == approximately(atan2(a, b)));

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
        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            compare_batch_scalar<fp_t>(
                {prime(x) = atan2(expression{number{a}}, expression{number{b}}), prime(y) = x + y}, opt_level,
                high_accuracy, compact_mode, rng, -10.f, 10.f, fp_t(10000));
        }

        // Variable-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = atan2(y, expression{number{a}}), prime(y) = atan2(x, expression{number{b}})},
                {fp_t(.2), fp_t(.3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "@heyoka.taylor_c_diff.atan2.var_num"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(jet[1], a)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(y, par[0]), prime(y) = atan2(x, expression{number{b}})},
                                            {fp_t(.2), fp_t(.3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {a}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(jet[1], a)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, expression{number{b}})},
                {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, par[1])},
                                              {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                                              2,
                                              kw::tol = 1,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{0}, fp_t{0}, b, b}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, expression{number{b}})},
                {fp_t(.2), fp_t(.3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(jet[1], b)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * b * jet[3] / (jet[1] * jet[1] + b * b)));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * b * jet[2] / (jet[0] * jet[0] + b * b)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, expression{number{b}})},
                {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                2,
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * b * jet[6] / (jet[2] * jet[2] + b * b)));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * b * jet[7] / (jet[3] * jet[3] + b * b)));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * b * jet[4] / (jet[0] * jet[0] + b * b)));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * b * jet[5] / (jet[1] * jet[1] + b * b)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, expression{number{b}})},
                {fp_t(.2), fp_t{.5}, fp_t(.1), fp_t(.3), fp_t(.4), fp_t(.6)},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));
            REQUIRE(jet[2] == fp_t(.1));

            REQUIRE(jet[3] == fp_t(.3));
            REQUIRE(jet[4] == fp_t(.4));
            REQUIRE(jet[5] == fp_t(.6));

            REQUIRE(jet[6] == approximately(atan2(jet[3], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[4], b)));
            REQUIRE(jet[8] == approximately(atan2(jet[5], b)));

            REQUIRE(jet[9] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[10] == approximately(atan2(jet[1], b)));
            REQUIRE(jet[11] == approximately(atan2(jet[2], b)));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * b * jet[9] / (jet[3] * jet[3] + b * b)));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * b * jet[10] / (jet[4] * jet[4] + b * b)));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * b * jet[11] / (jet[5] * jet[5] + b * b)));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * b * jet[6] / (jet[0] * jet[0] + b * b)));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * b * jet[7] / (jet[1] * jet[1] + b * b)));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * b * jet[8] / (jet[2] * jet[2] + b * b)));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[15] * (jet[3] * jet[3] + b * b) - b * jet[9] * 2 * jet[3] * jet[9])
                                     / ((jet[3] * jet[3] + b * b) * (jet[3] * jet[3] + b * b))));
            REQUIRE(jet[19]
                    == approximately(
                        fp_t{1} / 6 * (b * 2 * jet[16] * (jet[4] * jet[4] + b * b) - b * jet[10] * 2 * jet[4] * jet[10])
                        / ((jet[4] * jet[4] + b * b) * (jet[4] * jet[4] + b * b))));
            REQUIRE(jet[20]
                    == approximately(
                        fp_t{1} / 6 * (b * 2 * jet[17] * (jet[5] * jet[5] + b * b) - b * jet[11] * 2 * jet[5] * jet[11])
                        / ((jet[5] * jet[5] + b * b) * (jet[5] * jet[5] + b * b))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[12] * (jet[0] * jet[0] + b * b) - b * jet[6] * 2 * jet[0] * jet[6])
                                     / ((jet[0] * jet[0] + b * b) * (jet[0] * jet[0] + b * b))));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[13] * (jet[1] * jet[1] + b * b) - b * jet[7] * 2 * jet[1] * jet[7])
                                     / ((jet[1] * jet[1] + b * b) * (jet[1] * jet[1] + b * b))));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[14] * (jet[2] * jet[2] + b * b) - b * jet[8] * 2 * jet[2] * jet[8])
                                     / ((jet[2] * jet[2] + b * b) * (jet[2] * jet[2] + b * b))));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>(
                {prime(x) = atan2(y, expression{number{b}}), prime(y) = atan2(x, expression{number{b}})}, opt_level,
                high_accuracy, compact_mode, rng, -10.f, 10.f, fp_t(10000));
        }

        // Number-variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)},
                {fp_t(.2), fp_t(.3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "@heyoka.taylor_c_diff.atan2.num_var"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(par[0], y), prime(y) = atan2(expression{number{c}}, x)},
                                            {fp_t(.2), fp_t(.3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {a}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)},
                {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(par[1], x)},
                                              {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                                              2,
                                              kw::tol = 1,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{0}, fp_t{0}, c, c}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)},
                {fp_t(.2), fp_t(.3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[4] == approximately(-fp_t{1} / 2 * a * jet[3] / (jet[1] * jet[1] + a * a)));
            REQUIRE(jet[5] == approximately(-fp_t{1} / 2 * c * jet[2] / (jet[0] * jet[0] + c * c)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)},
                {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                2,
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));

            REQUIRE(jet[8] == approximately(-fp_t{1} / 2 * a * jet[6] / (jet[2] * jet[2] + a * a)));
            REQUIRE(jet[9] == approximately(-fp_t{1} / 2 * a * jet[7] / (jet[3] * jet[3] + a * a)));

            REQUIRE(jet[10] == approximately(-fp_t{1} / 2 * c * jet[4] / (jet[0] * jet[0] + c * c)));
            REQUIRE(jet[11] == approximately(-fp_t{1} / 2 * c * jet[5] / (jet[1] * jet[1] + c * c)));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)},
                {fp_t(.2), fp_t{.5}, fp_t(.1), fp_t(.3), fp_t(.4), fp_t(.6)},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));
            REQUIRE(jet[2] == fp_t(.1));

            REQUIRE(jet[3] == fp_t(.3));
            REQUIRE(jet[4] == fp_t(.4));
            REQUIRE(jet[5] == fp_t(.6));

            REQUIRE(jet[6] == approximately(atan2(a, jet[3])));
            REQUIRE(jet[7] == approximately(atan2(a, jet[4])));
            REQUIRE(jet[8] == approximately(atan2(a, jet[5])));

            REQUIRE(jet[9] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[10] == approximately(atan2(c, jet[1])));
            REQUIRE(jet[11] == approximately(atan2(c, jet[2])));

            REQUIRE(jet[12] == approximately(-fp_t{1} / 2 * a * jet[9] / (jet[3] * jet[3] + a * a)));
            REQUIRE(jet[13] == approximately(-fp_t{1} / 2 * a * jet[10] / (jet[4] * jet[4] + a * a)));
            REQUIRE(jet[14] == approximately(-fp_t{1} / 2 * a * jet[11] / (jet[5] * jet[5] + a * a)));

            REQUIRE(jet[15] == approximately(-fp_t{1} / 2 * c * jet[6] / (jet[0] * jet[0] + c * c)));
            REQUIRE(jet[16] == approximately(-fp_t{1} / 2 * c * jet[7] / (jet[1] * jet[1] + c * c)));
            REQUIRE(jet[17] == approximately(-fp_t{1} / 2 * c * jet[8] / (jet[2] * jet[2] + c * c)));

            REQUIRE(jet[18]
                    == approximately(-fp_t{1} / 6
                                     * (a * 2 * jet[15] * (jet[3] * jet[3] + a * a) - a * jet[9] * 2 * jet[3] * jet[9])
                                     / ((jet[3] * jet[3] + a * a) * (jet[3] * jet[3] + a * a))));
            REQUIRE(
                jet[19]
                == approximately(-fp_t{1} / 6
                                 * (a * 2 * jet[16] * (jet[4] * jet[4] + a * a) - a * jet[10] * 2 * jet[4] * jet[10])
                                 / ((jet[4] * jet[4] + a * a) * (jet[4] * jet[4] + a * a))));
            REQUIRE(
                jet[20]
                == approximately(-fp_t{1} / 6
                                 * (a * 2 * jet[17] * (jet[5] * jet[5] + a * a) - a * jet[11] * 2 * jet[5] * jet[11])
                                 / ((jet[5] * jet[5] + a * a) * (jet[5] * jet[5] + a * a))));

            REQUIRE(jet[21]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[12] * (jet[0] * jet[0] + c * c) - c * jet[6] * 2 * jet[0] * jet[6])
                                     / ((jet[0] * jet[0] + c * c) * (jet[0] * jet[0] + c * c))));
            REQUIRE(jet[22]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[13] * (jet[1] * jet[1] + c * c) - c * jet[7] * 2 * jet[1] * jet[7])
                                     / ((jet[1] * jet[1] + c * c) * (jet[1] * jet[1] + c * c))));
            REQUIRE(jet[23]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[14] * (jet[2] * jet[2] + c * c) - c * jet[8] * 2 * jet[2] * jet[8])
                                     / ((jet[2] * jet[2] + c * c) * (jet[2] * jet[2] + c * c))));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>(
                {prime(x) = atan2(expression{number{a}}, y), prime(y) = atan2(expression{number{c}}, x)}, opt_level,
                high_accuracy, compact_mode, rng, -10.f, 10.f, fp_t(10000));
        }

        // Variable-variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                            {fp_t(.2), fp_t(.3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "@heyoka.taylor_c_diff.atan2.var_var"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(atan2(jet[1], jet[0])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                                  {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(atan2(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(atan2(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(atan2(jet[3], jet[1])));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                            {fp_t(.2), fp_t(.3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.3));
            REQUIRE(jet[2] == approximately(atan2(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(atan2(jet[1], jet[0])));
            REQUIRE(jet[4]
                    == approximately(fp_t{1} / 2 * (jet[1] * jet[2] - jet[0] * jet[3])
                                     / (jet[0] * jet[0] + jet[1] * jet[1])));
            REQUIRE(jet[5]
                    == approximately(-fp_t{1} / 2 * (jet[1] * jet[2] - jet[0] * jet[3])
                                     / (jet[0] * jet[0] + jet[1] * jet[1])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                                  {fp_t(.2), fp_t{.5}, fp_t(.3), fp_t(.4)},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));

            REQUIRE(jet[2] == fp_t(.3));
            REQUIRE(jet[3] == fp_t(.4));

            REQUIRE(jet[4] == approximately(atan2(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(atan2(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(atan2(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(atan2(jet[3], jet[1])));

            REQUIRE(jet[8]
                    == approximately(fp_t{1} / 2 * (jet[2] * jet[4] - jet[0] * jet[6])
                                     / (jet[0] * jet[0] + jet[2] * jet[2])));
            REQUIRE(jet[9]
                    == approximately(fp_t{1} / 2 * (jet[3] * jet[5] - jet[1] * jet[7])
                                     / (jet[1] * jet[1] + jet[3] * jet[3])));

            REQUIRE(jet[10]
                    == approximately(-fp_t{1} / 2 * (jet[2] * jet[4] - jet[0] * jet[6])
                                     / (jet[0] * jet[0] + jet[2] * jet[2])));
            REQUIRE(jet[11]
                    == approximately(-fp_t{1} / 2 * (jet[3] * jet[5] - jet[1] * jet[7])
                                     / (jet[1] * jet[1] + jet[3] * jet[3])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                                  {fp_t(.2), fp_t{.5}, fp_t(.1), fp_t(.3), fp_t(.4), fp_t(.6)},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == fp_t(.2));
            REQUIRE(jet[1] == fp_t(.5));
            REQUIRE(jet[2] == fp_t(.1));

            REQUIRE(jet[3] == fp_t(.3));
            REQUIRE(jet[4] == fp_t(.4));
            REQUIRE(jet[5] == fp_t(.6));

            REQUIRE(jet[6] == approximately(atan2(jet[0], jet[3])));
            REQUIRE(jet[7] == approximately(atan2(jet[1], jet[4])));
            REQUIRE(jet[8] == approximately(atan2(jet[2], jet[5])));

            REQUIRE(jet[9] == approximately(atan2(jet[3], jet[0])));
            REQUIRE(jet[10] == approximately(atan2(jet[4], jet[1])));
            REQUIRE(jet[11] == approximately(atan2(jet[5], jet[2])));

            REQUIRE(jet[12]
                    == approximately(fp_t{1} / 2 * (jet[3] * jet[6] - jet[0] * jet[9])
                                     / (jet[0] * jet[0] + jet[3] * jet[3])));
            REQUIRE(jet[13]
                    == approximately(fp_t{1} / 2 * (jet[4] * jet[7] - jet[1] * jet[10])
                                     / (jet[1] * jet[1] + jet[4] * jet[4])));
            REQUIRE(jet[14]
                    == approximately(fp_t{1} / 2 * (jet[5] * jet[8] - jet[2] * jet[11])
                                     / (jet[2] * jet[2] + jet[5] * jet[5])));

            REQUIRE(jet[15]
                    == approximately(-fp_t{1} / 2 * (jet[3] * jet[6] - jet[0] * jet[9])
                                     / (jet[0] * jet[0] + jet[3] * jet[3])));
            REQUIRE(jet[16]
                    == approximately(-fp_t{1} / 2 * (jet[4] * jet[7] - jet[1] * jet[10])
                                     / (jet[1] * jet[1] + jet[4] * jet[4])));
            REQUIRE(jet[17]
                    == approximately(-fp_t{1} / 2 * (jet[5] * jet[8] - jet[2] * jet[11])
                                     / (jet[2] * jet[2] + jet[5] * jet[5])));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * (2 * (jet[12] * jet[3] - jet[0] * jet[15]) * (jet[0] * jet[0] + jet[3] * jet[3])
                                        - 2 * (jet[6] * jet[3] - jet[0] * jet[9]) * (jet[0] * jet[6] + jet[3] * jet[9]))
                                     / ((jet[0] * jet[0] + jet[3] * jet[3]) * (jet[0] * jet[0] + jet[3] * jet[3]))));
            REQUIRE(
                jet[19]
                == approximately(fp_t{1} / 6
                                 * (2 * (jet[13] * jet[4] - jet[1] * jet[16]) * (jet[1] * jet[1] + jet[4] * jet[4])
                                    - 2 * (jet[7] * jet[4] - jet[1] * jet[10]) * (jet[1] * jet[7] + jet[4] * jet[10]))
                                 / ((jet[1] * jet[1] + jet[4] * jet[4]) * (jet[1] * jet[1] + jet[4] * jet[4]))));
            REQUIRE(
                jet[20]
                == approximately(fp_t{1} / 6
                                 * (2 * (jet[14] * jet[5] - jet[2] * jet[17]) * (jet[2] * jet[2] + jet[5] * jet[5])
                                    - 2 * (jet[8] * jet[5] - jet[2] * jet[11]) * (jet[2] * jet[8] + jet[5] * jet[11]))
                                 / ((jet[2] * jet[2] + jet[5] * jet[5]) * (jet[2] * jet[2] + jet[5] * jet[5]))));

            REQUIRE(jet[21]
                    == approximately(-fp_t{1} / 6
                                     * (2 * (jet[12] * jet[3] - jet[0] * jet[15]) * (jet[0] * jet[0] + jet[3] * jet[3])
                                        - 2 * (jet[6] * jet[3] - jet[0] * jet[9]) * (jet[0] * jet[6] + jet[3] * jet[9]))
                                     / ((jet[0] * jet[0] + jet[3] * jet[3]) * (jet[0] * jet[0] + jet[3] * jet[3]))));
            REQUIRE(
                jet[22]
                == approximately(-fp_t{1} / 6
                                 * (2 * (jet[13] * jet[4] - jet[1] * jet[16]) * (jet[1] * jet[1] + jet[4] * jet[4])
                                    - 2 * (jet[7] * jet[4] - jet[1] * jet[10]) * (jet[1] * jet[7] + jet[4] * jet[10]))
                                 / ((jet[1] * jet[1] + jet[4] * jet[4]) * (jet[1] * jet[1] + jet[4] * jet[4]))));
            REQUIRE(
                jet[23]
                == approximately(-fp_t{1} / 6
                                 * (2 * (jet[14] * jet[5] - jet[2] * jet[17]) * (jet[2] * jet[2] + jet[5] * jet[5])
                                    - 2 * (jet[8] * jet[5] - jet[2] * jet[11]) * (jet[2] * jet[8] + jet[5] * jet[11]))
                                 / ((jet[2] * jet[2] + jet[5] * jet[5]) * (jet[2] * jet[2] + jet[5] * jet[5]))));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>({prime(x) = atan2(x, y), prime(y) = atan2(y, x)}, opt_level, high_accuracy,
                                       compact_mode, rng, -10.f, 10.f, fp_t(10000));
        }
    };

    for (auto cm : {true, false}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}
