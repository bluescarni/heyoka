// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
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

TEST_CASE("taylor const sys")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet", {prime(x) = y, prime(y) = x}, 1, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-3}};
            jet.resize(4);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -3);
            REQUIRE(jet[2] == jet[1]);
            REQUIRE(jet[3] == jet[0]);
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet", {prime(x) = y, prime(y) = x}, 1, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{-3}, fp_t{5}};
            jet.resize(8);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == jet[2]);
            REQUIRE(jet[5] == jet[3]);

            REQUIRE(jet[6] == jet[0]);
            REQUIRE(jet[7] == jet[1]);
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet", {prime(x) = y, prime(y) = x}, 2, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-3}};
            jet.resize(6);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -3);
            REQUIRE(jet[2] == jet[1]);
            REQUIRE(jet[3] == jet[0]);
            REQUIRE(jet[4] == jet[3] / 2);
            REQUIRE(jet[5] == jet[2] / 2);
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet", {prime(x) = y, prime(y) = x}, 2, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{-3}, fp_t{5}};
            jet.resize(12);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == -3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == jet[2]);
            REQUIRE(jet[5] == jet[3]);

            REQUIRE(jet[6] == jet[0]);
            REQUIRE(jet[7] == jet[1]);

            REQUIRE(jet[8] == jet[6] / 2);
            REQUIRE(jet[9] == jet[7] / 2);

            REQUIRE(jet[10] == jet[4] / 2);
            REQUIRE(jet[11] == jet[5] / 2);
        }

        {
            llvm_state s{"", opt_level};

            s.add_taylor_jet_batch<fp_t>("jet", {prime(x) = y, prime(y) = x}, 3, 3);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{0}, fp_t{-3}, fp_t{5}, fp_t{4}};
            jet.resize(24);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 4);

            REQUIRE(jet[6] == jet[3]);
            REQUIRE(jet[7] == jet[4]);
            REQUIRE(jet[8] == jet[5]);

            REQUIRE(jet[9] == jet[0]);
            REQUIRE(jet[10] == jet[1]);
            REQUIRE(jet[11] == jet[2]);

            REQUIRE(jet[12] == jet[9] / 2);
            REQUIRE(jet[13] == jet[10] / 2);
            REQUIRE(jet[14] == jet[11] / 2);

            REQUIRE(jet[15] == jet[6] / 2);
            REQUIRE(jet[16] == jet[7] / 2);
            REQUIRE(jet[17] == jet[8] / 2);

            REQUIRE(jet[18] == fp_t{1} / 6 * jet[15] * 2);
            REQUIRE(jet[19] == fp_t{1} / 6 * jet[16] * 2);
            REQUIRE(jet[20] == fp_t{1} / 6 * jet[17] * 2);

            REQUIRE(jet[21] == fp_t{1} / 6 * jet[12] * 2);
            REQUIRE(jet[22] == fp_t{1} / 6 * jet[13] * 2);
            REQUIRE(jet[23] == fp_t{1} / 6 * jet[14] * 2);
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = y, prime(y) = x}, opt_level);
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 0); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 1); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 2); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 3); });
}
