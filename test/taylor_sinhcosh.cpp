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

#include <llvm/Config/llvm-config.h>

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

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

// Potential issue in the decomposition when x = 0 (not
// currently the case).
TEST_CASE("taylor sinhcosh decompose bug 00")
{
    auto x = "x"_var;

    auto ta = taylor_adaptive<double>({prime(x) = sinh(0_dbl) + cosh(0_dbl) - x}, {0.}, kw::tol = 1.);
}

TEST_CASE("ode test")
{
    using std::cosh;
    using std::sinh;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, s, c] = make_vars("x", "s", "c");

                taylor_adaptive<double> ta0({prime(x) = sinh(x) + cosh(x)}, {-.005}, kw::high_accuracy = ha,
                                            kw::compact_mode = cm, kw::opt_level = opt_level);
                taylor_adaptive<double> ta1({prime(x) = s + c, prime(s) = c * (s + c), prime(c) = s * (s + c)},
                                            {-.005, sinh(-.005), cosh(-.005)}, kw::high_accuracy = ha,
                                            kw::compact_mode = cm, kw::opt_level = opt_level);

                ta0.propagate_until(-5.);
                ta1.propagate_until(-5.);

                REQUIRE(ta0.get_state()[0] == approximately(ta1.get_state()[0]));

                const auto v0 = sinh(ta0.get_state()[0]) + cosh(ta0.get_state()[0]);
                const auto v1 = ta1.get_state()[1] + ta1.get_state()[2];

                REQUIRE(v0 == approximately(v1));
            }
        }
    }
}

TEST_CASE("taylor sinhcosh")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::sinh;
        using std::cosh;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sinh(par[0]) + cosh(par[1]), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::pars = {fp_t{2}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[5] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sinh(par[0]) + cosh(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[5] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] + jet[3])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                2,
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[5] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[5] + jet[7])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[7] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[8] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));

            REQUIRE(jet[9] == approximately(jet[0] + jet[3]));
            REQUIRE(jet[10] == approximately(jet[1] + jet[4]));
            REQUIRE(jet[11] == approximately(jet[2] + jet[5]));

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15] + 2 * jet[18])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16] + 2 * jet[19])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17] + 2 * jet[20])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sinh(par[0]) + cosh(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[7] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));
            REQUIRE(jet[8] == approximately(sinh(fp_t{2}) + cosh(fp_t{3})));

            REQUIRE(jet[9] == approximately(jet[0] + jet[3]));
            REQUIRE(jet[10] == approximately(jet[1] + jet[4]));
            REQUIRE(jet[11] == approximately(jet[2] + jet[5]));

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15] + 2 * jet[18])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16] + 2 * jet[19])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17] + 2 * jet[20])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>(
                {prime(x) = sinh(expression{number{fp_t{2}}}) + cosh(expression{number{fp_t{3}}}), prime(y) = x + y},
                opt_level, high_accuracy, compact_mode, rng, -10.f, 10.f, fp_t(10000));
        }

        // Variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sinh(y), prime(y) = cosh(x)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sinh(jet[1])));
            REQUIRE(jet[3] == approximately(cosh(jet[0])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sinh(y), prime(y) = cosh(x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}},
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
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sinh(jet[2])));
            REQUIRE(jet[5] == approximately(sinh(jet[3])));

            REQUIRE(jet[6] == approximately(cosh(jet[0])));
            REQUIRE(jet[7] == approximately(cosh(jet[1])));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sinh(y), prime(y) = cosh(x)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sinh(jet[1])));
            REQUIRE(jet[3] == approximately(cosh(jet[0])));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * jet[3] * cosh(jet[1])));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * jet[2] * sinh(jet[0])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sinh(y), prime(y) = cosh(x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}},
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
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sinh(jet[2])));
            REQUIRE(jet[5] == approximately(sinh(jet[3])));

            REQUIRE(jet[6] == approximately(cosh(jet[0])));
            REQUIRE(jet[7] == approximately(cosh(jet[1])));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[6] * cosh(jet[2])));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[7] * cosh(jet[3])));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * jet[4] * sinh(jet[0])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * jet[5] * sinh(jet[1])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sinh(y), prime(y) = cosh(x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{-5}, fp_t{3}, fp_t{-4}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -5);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(sinh(jet[3])));
            REQUIRE(jet[7] == approximately(sinh(jet[4])));
            REQUIRE(jet[8] == approximately(sinh(jet[5])));

            REQUIRE(jet[9] == approximately(cosh(jet[0])));
            REQUIRE(jet[10] == approximately(cosh(jet[1])));
            REQUIRE(jet[11] == approximately(cosh(jet[2])));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[9] * cosh(jet[3])));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[10] * cosh(jet[4])));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[11] * cosh(jet[5])));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * jet[6] * sinh(jet[0])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * jet[7] * sinh(jet[1])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * jet[8] * sinh(jet[2])));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6 * (2 * jet[15] * cosh(jet[3]) + jet[9] * jet[9] * sinh(jet[3]))));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6 * (2 * jet[16] * cosh(jet[4]) + jet[10] * jet[10] * sinh(jet[4]))));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6 * (2 * jet[17] * cosh(jet[5]) + jet[11] * jet[11] * sinh(jet[5]))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6 * (2 * jet[12] * sinh(jet[0]) + jet[6] * jet[6] * cosh(jet[0]))));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6 * (2 * jet[13] * sinh(jet[1]) + jet[7] * jet[7] * cosh(jet[1]))));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6 * (2 * jet[14] * sinh(jet[2]) + jet[8] * jet[8] * cosh(jet[2]))));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>({prime(x) = sinh(y), prime(y) = cosh(x)}, opt_level, high_accuracy, compact_mode,
                                       rng, -10.f, 10.f, fp_t(10000));
        }
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

// Test expression simplification with sinh/cosh.
TEST_CASE("taylor sinhcosh cse")
{
    using std::cosh;
    using std::sinh;

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{
        {prime(x) = (sinh(x) + cosh(y)) + (sinh(y) + cosh(x)), prime(y) = (sinh(x) + cosh(y)) + (sinh(y) + cosh(x))},
        {2., 3.},
        kw::opt_level = 0,
        kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 11u);

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(sinh(jet[1]) + cosh(jet[0]) + sinh(jet[0]) + cosh(jet[1])));
    REQUIRE(jet[3] == approximately(sinh(jet[1]) + cosh(jet[0]) + sinh(jet[0]) + cosh(jet[1])));
    REQUIRE(jet[4]
            == approximately(
                (cosh(jet[1]) * jet[3] + sinh(jet[0]) * jet[2] + cosh(jet[0]) * jet[2] + sinh(jet[1]) * jet[3]) / 2));
    REQUIRE(jet[5]
            == approximately(
                (cosh(jet[1]) * jet[3] + sinh(jet[0]) * jet[2] + cosh(jet[0]) * jet[2] + sinh(jet[1]) * jet[3]) / 2));
}
