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

    llvm_state s{kw::opt_level = opt_level};

    s.add_taylor_jet_batch<T>("jet_batch", sys, 3, batch_size);
    s.add_taylor_jet_batch<T>("jet_scalar", sys, 3, 1);

    s.compile();

    auto jptr_batch = s.fetch_taylor_jet_batch<T>("jet_batch");
    auto jptr_scalar = s.fetch_taylor_jet_batch<T>("jet_scalar");

    std::vector<T> jet_batch;
    jet_batch.resize(12 * batch_size);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    std::generate(jet_batch.begin(), jet_batch.end(), [&dist]() { return T{dist(rng)}; });

    std::vector<T> jet_scalar;
    jet_scalar.resize(12);

    jptr_batch(jet_batch.data());

    for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
        // Assign the initial values of x and y.
        for (auto i = 0u; i < 3u; ++i) {
            jet_scalar[i] = jet_batch[i * batch_size + batch_idx];
        }

        jptr_scalar(jet_scalar.data());

        for (auto i = 3u; i < 12u; ++i) {
            REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx]));
        }
    }
}

TEST_CASE("taylor const sys")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        auto [x, y, z] = make_vars("x", "y", "z");

        {
            llvm_state s{kw::opt_level = opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {prime(x) = expression{number{fp_t{1}}},
                                          prime(y) = expression{number{fp_t{-2}}},
                                          prime(z) = expression{number{fp_t{0}}}},
                                         1, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}, fp_t{4}};
            jet.resize(6);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {prime(x) = expression{number{fp_t{1}}},
                                          prime(y) = expression{number{fp_t{-2}}},
                                          prime(z) = expression{number{fp_t{0}}}},
                                         1, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}};
            jet.resize(12);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -4);

            REQUIRE(jet[6] == 1);
            REQUIRE(jet[7] == 1);

            REQUIRE(jet[8] == -2);
            REQUIRE(jet[9] == -2);

            REQUIRE(jet[10] == 0);
            REQUIRE(jet[11] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {prime(x) = expression{number{fp_t{1}}},
                                          prime(y) = expression{number{fp_t{-2}}},
                                          prime(z) = expression{number{fp_t{0}}}},
                                         2, 1);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}, fp_t{4}};
            jet.resize(9);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
            REQUIRE(jet[6] == 0);
            REQUIRE(jet[7] == 0);
            REQUIRE(jet[8] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {prime(x) = expression{number{fp_t{1}}},
                                          prime(y) = expression{number{fp_t{-2}}},
                                          prime(z) = expression{number{fp_t{0}}}},
                                         2, 2);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}};
            jet.resize(18);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -4);

            REQUIRE(jet[6] == 1);
            REQUIRE(jet[7] == 1);

            REQUIRE(jet[8] == -2);
            REQUIRE(jet[9] == -2);

            REQUIRE(jet[10] == 0);
            REQUIRE(jet[11] == 0);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);

            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == 0);

            REQUIRE(jet[16] == 0);
            REQUIRE(jet[17] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            s.add_taylor_jet_batch<fp_t>("jet",
                                         {prime(x) = expression{number{fp_t{1}}},
                                          prime(y) = expression{number{fp_t{-2}}},
                                          prime(z) = expression{number{fp_t{0}}}},
                                         3, 3);

            s.compile();

            auto jptr = s.fetch_taylor_jet_batch<fp_t>("jet");

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{0}, fp_t{3}, fp_t{-3}, fp_t{0}, fp_t{4}, fp_t{-4}, fp_t{0}};
            jet.resize(36);

            jptr(jet.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == 4);
            REQUIRE(jet[7] == -4);
            REQUIRE(jet[8] == 0);

            REQUIRE(jet[9] == 1);
            REQUIRE(jet[10] == 1);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == -2);
            REQUIRE(jet[13] == -2);
            REQUIRE(jet[14] == -2);

            REQUIRE(jet[15] == 0);
            REQUIRE(jet[16] == 0);
            REQUIRE(jet[17] == 0);

            for (auto i = 18u; i < 36u; ++i) {
                REQUIRE(jet[i] == 0);
            }
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                    prime(z) = expression{number{fp_t{0}}}},
                                   opt_level);
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 0); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 1); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 2); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 3); });
}
