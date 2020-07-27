// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

template <typename T, typename U>
void compare_batch_scalar(std::initializer_list<U> sys, unsigned opt_level)
{
    const auto batch_size = 23u;

    llvm_state s{"", opt_level};

    s.add_taylor_jet_batch<T>("jet_batch", sys, 3, batch_size);
    s.add_taylor_jet_batch<T>("jet_scalar", sys, 3, 1);

    s.compile();

    auto jptr_batch = s.fetch_taylor_jet_batch<T>("jet_batch");
    auto jptr_scalar = s.fetch_taylor_jet_batch<T>("jet_scalar");

    std::vector<T> jet_batch;
    jet_batch.resize(8 * batch_size);
    std::uniform_real_distribution<float> dist(.1f, 20.f);
    std::generate(jet_batch.begin(), jet_batch.end(), [&dist]() { return T{dist(rng)}; });

    std::vector<T> jet_scalar;
    jet_scalar.resize(8);

    jptr_batch(jet_batch.data());

    for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
        // Assign the initial values of x and y.
        for (auto i = 0u; i < 2u; ++i) {
            jet_scalar[i] = jet_batch[i * batch_size + batch_idx];
        }

        jptr_scalar(jet_scalar.data());

        for (auto i = 2u; i < 8u; ++i) {
            REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx]));
        }
    }
}

TEST_CASE("taylor pow")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using std::pow;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>(
                "jet", {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 1, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>(
                "jet", {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 1, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            jptr(jet.data());

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
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>(
                "jet", {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 2, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>(
                "jet", {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 2, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(12);

            jptr(jet.data());

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
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>(
                "jet", {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 3, 3);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data());

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

            REQUIRE(jet[21] == fp_t{1} / 6 * (2 * jet[12] + 2 * jet[15]));
            REQUIRE(jet[22] == fp_t{1} / 6 * (2 * jet[13] + 2 * jet[16]));
            REQUIRE(jet[23] == fp_t{1} / 6 * (2 * jet[14] + 2 * jet[17]));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y},
                                   opt_level);

        // Variable-number tests.
        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                          pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                         1, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                          pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                         1, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            jptr(jet.data());

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
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                          pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                         2, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[1], fp_t{1} / 2) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[2]));
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                          pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                         2, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data());

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
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                          pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                         3, 3);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{1}, fp_t{3}, fp_t{4}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data());

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
        compare_batch_scalar<fp_t>({pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                    pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                   opt_level);
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 0); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 1); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 2); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 3); });
}

#if 0

#if defined(HEYOKA_HAVE_REAL128)

TEST_CASE("f128")
{
    using Catch::Matchers::Message;
    using namespace mppp::literals;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(y, 3_f128 / 2_f128), pow(x, -3_f128 / 2_f128)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3_rq, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2_rq, -3 / 2._rq)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(y, 3_f128 / 2_f128), pow(x, -3_f128 / 2_f128)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));
        REQUIRE(jet[4] == approximately(0.5_rq * 3 / 2._rq * sqrt(3._rq) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5_rq * -3 / 2._rq * pow(2, -5 / 2._rq) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));
        REQUIRE(jet[4] == approximately(0.5_rq * 3 / 2._rq * sqrt(3._rq) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5_rq * -3 / 2._rq * pow(2, -5 / 2._rq) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6._rq * 3 / 2._rq
                                 * (1 / 2._rq * 1 / sqrt(3._rq) * jet[3] * jet[3] + sqrt(3._rq) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(1 / 6._rq
                                 * (15 / 4._rq * pow(2, -7 / 2._rq) * jet[2] * jet[2]
                                    - 3 / 2._rq * pow(2, -5 / 2._rq) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(3_f128, 3_f128 / 2_f128), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5_rq * (pow(3, 3 / 2._rq) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5_rq * (pow(3, 3 / 2._rq) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6._rq * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_f128("jet", {pow(1_f128, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_f128("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}

#endif

TEST_CASE("dbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(y, 3_dbl / 2_dbl), pow(x, -3_dbl / 2_dbl)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(y, 3_dbl / 2_dbl), pow(x, -3_dbl / 2_dbl)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6. * 3 / 2.
                                 * (1 / 2. * 1 / std::sqrt(3) * jet[3] * jet[3] + std::sqrt(3) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(
                    1 / 6.
                    * (15 / 4. * std::pow(2, -7 / 2.) * jet[2] * jet[2] - 3 / 2. * std::pow(2, -5 / 2.) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(3_dbl, 3_dbl / 2_dbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6. * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_dbl("jet", {pow(1_dbl, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_dbl("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}

TEST_CASE("ldbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(y, 3_ldbl / 2_ldbl), pow(x, -3_ldbl / 2_ldbl)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(y, 3_ldbl / 2_ldbl), pow(x, -3_ldbl / 2_ldbl)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3.l) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.l) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3.l) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.l) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6.l * 3 / 2.
                                 * (1 / 2. * 1 / std::sqrt(3.l) * jet[3] * jet[3] + std::sqrt(3.l) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(1 / 6.l
                                 * (15 / 4. * std::pow(2, -7 / 2.l) * jet[2] * jet[2]
                                    - 3 / 2. * std::pow(2, -5 / 2.l) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(3_ldbl, 3_ldbl / 2_ldbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.l) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.l) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6.l * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_ldbl("jet", {pow(1_ldbl, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_ldbl("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}

#endif
