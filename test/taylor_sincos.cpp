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

// Potential issue in the decomposition when x = 0 (not
// currently the case).
TEST_CASE("taylor sincos decompose bug 00")
{
    auto x = "x"_var;

    auto ta = taylor_adaptive<double>({prime(x) = sin(0_dbl) + cos(0_dbl) - x}, {0.}, kw::tol = 1.);
}

TEST_CASE("taylor sincos")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode, bool fast_math) {
        using std::sin;
        using std::cos;

        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(par[0]) + cos(par[1]), prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::fast_math = fast_math,
                                            kw::pars = {fp_t{2}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                2,
                kw::tol = 1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[5] == approximately(sin(fp_t{2}) + cos(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sin(par[0]) + cos(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::fast_math = fast_math,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[5] == approximately(sin(fp_t{2}) + cos(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        {
            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t(2), fp_t(3)},
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] + jet[3])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}},
                2,
                kw::tol = .5,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[5] == approximately(sin(fp_t{2}) + cos(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[5] + jet[7])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{
                {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
                {fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}},
                3,
                kw::tol = .1,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::opt_level = opt_level,
                kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[7] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[8] == approximately(sin(fp_t{2}) + cos(fp_t{3})));

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

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sin(par[0]) + cos(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::fast_math = fast_math,
                                                  kw::pars = {fp_t{2}, fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}, fp_t{3}}};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[7] == approximately(sin(fp_t{2}) + cos(fp_t{3})));
            REQUIRE(jet[8] == approximately(sin(fp_t{2}) + cos(fp_t{3})));

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

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>(
            {prime(x) = sin(expression{number{fp_t{2}}}) + cos(expression{number{fp_t{3}}}), prime(y) = x + y},
            opt_level, high_accuracy, compact_mode, rng, -10.f, 10.f);

        // Variable tests.
        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(y + 1_dbl), prime(y) = cos(x + 1_dbl)},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sin(jet[1] + 1)));
            REQUIRE(jet[3] == approximately(cos(jet[0] + 1)));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sin(y + 1_dbl), prime(y) = cos(x + 1_dbl)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sin(jet[2] + 1)));
            REQUIRE(jet[5] == approximately(sin(jet[3] + 1)));

            REQUIRE(jet[6] == approximately(cos(jet[0] + 1)));
            REQUIRE(jet[7] == approximately(cos(jet[1] + 1)));
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(y + 1_dbl), prime(y) = cos(x + 1_dbl)},
                                            {fp_t{2}, fp_t{3}},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level,
                                            kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sin(jet[1] + 1)));
            REQUIRE(jet[3] == approximately(cos(jet[0] + 1)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * jet[3] * cos(jet[1] + 1)));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * -jet[2] * sin(jet[0] + 1)));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sin(y), prime(y) = cos(x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sin(jet[2])));
            REQUIRE(jet[5] == approximately(sin(jet[3])));

            REQUIRE(jet[6] == approximately(cos(jet[0])));
            REQUIRE(jet[7] == approximately(cos(jet[1])));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[6] * cos(jet[2])));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[7] * cos(jet[3])));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * -jet[4] * sin(jet[0])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * -jet[5] * sin(jet[1])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = sin(y), prime(y) = cos(x)},
                                                  {fp_t{2}, fp_t{-1}, fp_t{-5}, fp_t{3}, fp_t{-4}, fp_t{6}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::fast_math = fast_math};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -5);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(sin(jet[3])));
            REQUIRE(jet[7] == approximately(sin(jet[4])));
            REQUIRE(jet[8] == approximately(sin(jet[5])));

            REQUIRE(jet[9] == approximately(cos(jet[0])));
            REQUIRE(jet[10] == approximately(cos(jet[1])));
            REQUIRE(jet[11] == approximately(cos(jet[2])));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[9] * cos(jet[3])));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[10] * cos(jet[4])));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[11] * cos(jet[5])));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * -jet[6] * sin(jet[0])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * -jet[7] * sin(jet[1])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * -jet[8] * sin(jet[2])));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6 * (2 * jet[15] * cos(jet[3]) - jet[9] * jet[9] * sin(jet[3]))));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6 * (2 * jet[16] * cos(jet[4]) - jet[10] * jet[10] * sin(jet[4]))));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6 * (2 * jet[17] * cos(jet[5]) - jet[11] * jet[11] * sin(jet[5]))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6 * (-2 * jet[12] * sin(jet[0]) - jet[6] * jet[6] * cos(jet[0]))));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6 * (-2 * jet[13] * sin(jet[1]) - jet[7] * jet[7] * cos(jet[1]))));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6 * (-2 * jet[14] * sin(jet[2]) - jet[8] * jet[8] * cos(jet[2]))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = sin(y), prime(y) = cos(x)}, opt_level, high_accuracy, compact_mode, rng,
                                   -10.f, 10.f);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            // NOTE: when enabling the fast math flag, we are observing buggy behaviour
            // on x86 with 80-bit long double/real128. Specifically, it seems like C++ code
            // such as sin(expression{number{fp_t{2}}}) (which ultimately just invokes
            // std::sin() due to constant folding) returns sometimes a value of NaN.
            // This happens when both 80-bit long double and real128 are available,
            // and it is triggered when both the no-signed-zeros and no-nans fast math
            // flags are activated. Disabling either the long double or real128 testing
            // makes the problem go away.
            //
            // My impression is that there is some code generation bug in LLVM which leaks
            // out into the surrounding C++ code - perhaps some register not being properly
            // restored or something like that. For now, let us implement this workaround.
            constexpr bool enable_fm_test =
#if defined(HEYOKA_ARCH_PPC)
                true
#else
                std::numeric_limits<long double>::digits != 64
#endif
                ;

            for (auto fm : {false, enable_fm_test}) {
                tuple_for_each(fp_types, [&tester, f, cm, fm](auto x) { tester(x, 0, f, cm, fm); });
                tuple_for_each(fp_types, [&tester, f, cm, fm](auto x) { tester(x, 1, f, cm, fm); });
                tuple_for_each(fp_types, [&tester, f, cm, fm](auto x) { tester(x, 2, f, cm, fm); });
                tuple_for_each(fp_types, [&tester, f, cm, fm](auto x) { tester(x, 3, f, cm, fm); });
            }
        }
    }
}

// Test expression simplification with sine/cosine.
TEST_CASE("taylor sincos cse")
{
    using std::cos;
    using std::sin;

    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{
        {prime(x) = sin(y) + cos(x) + sin(x) + cos(y), prime(y) = sin(y) + cos(x) + sin(x) + cos(y)},
        {2., 3.},
        kw::opt_level = 0,
        kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 9u);

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1])));
    REQUIRE(jet[3] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1])));
    REQUIRE(jet[4]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
    REQUIRE(jet[5]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
}
