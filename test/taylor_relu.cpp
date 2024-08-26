// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/relu.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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

template <typename T>
T cpp_relu(T x, T slope = 0)
{
    return x > 0 ? x : slope * x;
}

template <typename T>
T cpp_relup(T x, T slope = 0)
{
    return x > 0 ? T(1) : slope;
}

TEST_CASE("taylor relu relup")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = relu(par[0]) + relup(par[1]), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level,
                                                  kw::pars = {fp_t{-1}, fp_t{2}, fp_t{4}, fp_t{-3}}};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relu.par"));
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relup.par"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            const std::vector pars = {fp_t{-1}, fp_t{2}, fp_t{4}, fp_t{-3}};

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(pars[0]) + cpp_relup(pars[2])));
            REQUIRE(jet[5] == approximately(cpp_relu(pars[1]) + cpp_relup(pars[3])));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == fp_t(0));
            REQUIRE(jet[9] == fp_t(0));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == fp_t(0));
            REQUIRE(jet[13] == fp_t(0));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
        }

        // Variable tests.
        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = relu(x) + relup(y), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relu.var"));
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relup.var"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(jet[0]) + cpp_relup(jet[2])));
            REQUIRE(jet[5] == approximately(cpp_relu(jet[1]) + cpp_relup(jet[3])));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == approximately((cpp_relup(jet[0]) * jet[4]) / 2));
            REQUIRE(jet[9] == approximately((cpp_relup(jet[1]) * jet[5]) / 2));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == approximately((cpp_relup(jet[0]) * 2 * jet[8]) / 6));
            REQUIRE(jet[13] == approximately((cpp_relup(jet[1]) * 2 * jet[9]) / 6));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
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

TEST_CASE("taylor relu relup leaky")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            auto ta
                = taylor_adaptive_batch<fp_t>{{prime(x) = relu(par[0], 0.01) + relup(par[1], 0.02), prime(y) = x + y},
                                              {fp_t{2}, fp_t{-1}, fp_t{-3}, fp_t{5}},
                                              2,
                                              kw::tol = .1,
                                              kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode,
                                              kw::opt_level = opt_level,
                                              kw::pars = {fp_t{-1}, fp_t{2}, fp_t{4}, fp_t{-3}}};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relu_0x"));
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relup_0x"));
                REQUIRE(ir_contains(ta, ".par"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            const std::vector pars = {fp_t{-1}, fp_t{2}, fp_t{4}, fp_t{-3}};

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(pars[0], fp_t(0.01)) + cpp_relup(pars[2], fp_t(0.02))));
            REQUIRE(jet[5] == approximately(cpp_relu(pars[1], fp_t(0.01)) + cpp_relup(pars[3], fp_t(0.02))));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == fp_t(0));
            REQUIRE(jet[9] == fp_t(0));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == fp_t(0));
            REQUIRE(jet[13] == fp_t(0));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
        }

        // Variable tests.
        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = relu(x, 0.01) + relup(y, 0.02), prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-1}, fp_t{-3}, fp_t{5}},
                                                  2,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            if (opt_level == 0u && compact_mode) {
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relu_0x"));
                REQUIRE(ir_contains(ta, "@heyoka.taylor_c_diff.relup_0x"));
                REQUIRE(ir_contains(ta, ".var"));
            }

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(jet[0], fp_t(0.01)) + cpp_relup(jet[2], fp_t(0.02))));
            REQUIRE(jet[5] == approximately(cpp_relu(jet[1], fp_t(0.01)) + cpp_relup(jet[3], fp_t(0.02))));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == approximately((cpp_relup(jet[0], fp_t(0.01)) * jet[4]) / 2));
            REQUIRE(jet[9] == approximately((cpp_relup(jet[1], fp_t(0.01)) * jet[5]) / 2));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == approximately((cpp_relup(jet[0], fp_t(0.01)) * 2 * jet[8]) / 6));
            REQUIRE(jet[13] == approximately((cpp_relup(jet[1], fp_t(0.01)) * 2 * jet[9]) / 6));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
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
