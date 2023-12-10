// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <random>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/math/prod.hpp>
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

// Helper to ease the removal of mul() in the test code.
auto mul(const expression &a, const expression &b)
{
    return expression{func{detail::prod_impl({a, b})}};
}

template <typename T, typename U>
void compare_batch_scalar(std::initializer_list<U> sys, unsigned opt_level, bool high_accuracy, bool compact_mode)
{
    const auto batch_size = 5u;

    llvm_state s{kw::opt_level = opt_level};

    taylor_add_jet<T>(s, "jet_batch", sys, 3, batch_size, high_accuracy, compact_mode);
    taylor_add_jet<T>(s, "jet_scalar", sys, 3, 1, high_accuracy, compact_mode);

    s.compile();

    auto jptr_batch = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_batch"));
    auto jptr_scalar = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_scalar"));

    std::vector<T> jet_batch;
    jet_batch.resize(12 * batch_size);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    std::generate(jet_batch.begin(), jet_batch.end(), [&dist]() { return T{dist(rng)}; });

    std::vector<T> jet_scalar;
    jet_scalar.resize(12);

    jptr_batch(jet_batch.data(), nullptr, nullptr);

    for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
        // Assign the initial values of x and y.
        for (auto i = 0u; i < 3u; ++i) {
            jet_scalar[i] = jet_batch[i * batch_size + batch_idx];
        }

        jptr_scalar(jet_scalar.data(), nullptr, nullptr);

        for (auto i = 3u; i < 12u; ++i) {
            REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx]));
        }
    }
}

TEST_CASE("taylor const sys")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y, z] = make_vars("x", "y", "z");

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                  prime(z) = expression{number{fp_t{0}}}},
                                 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}, fp_t{4}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {prime(x) = par[0], prime(y) = expression{number{fp_t{-2}}}, prime(z) = expression{number{fp_t{0}}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}, fp_t{4}};
            jet.resize(6);

            std::vector<fp_t> pars{fp_t{1}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 4);
            REQUIRE(jet[3] == 1);
            REQUIRE(jet[4] == -2);
            REQUIRE(jet[5] == 0);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                  prime(z) = expression{number{fp_t{0}}}},
                                 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

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

            taylor_add_jet<fp_t>(
                s, "jet",
                {prime(x) = expression{number{fp_t{1}}}, prime(y) = par[1], prime(z) = expression{number{fp_t{0}}}}, 1,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}};
            jet.resize(12);

            std::vector<fp_t> pars{fp_t{1}, fp_t{1}, fp_t{-2}, fp_t{-2}};

            jptr(jet.data(), pars.data(), nullptr);

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

            taylor_add_jet<fp_t>(s, "jet",
                                 {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                  prime(z) = expression{number{fp_t{0}}}},
                                 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}, fp_t{4}};
            jet.resize(9);

            jptr(jet.data(), nullptr, nullptr);

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

            taylor_add_jet<fp_t>(s, "jet",
                                 {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                  prime(z) = expression{number{fp_t{0}}}},
                                 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}, fp_t{4}, fp_t{-4}};
            jet.resize(18);

            jptr(jet.data(), nullptr, nullptr);

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

            taylor_add_jet<fp_t>(s, "jet",
                                 {prime(x) = expression{number{fp_t{1}}}, prime(y) = expression{number{fp_t{-2}}},
                                  prime(z) = expression{number{fp_t{0}}}},
                                 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{0}, fp_t{3}, fp_t{-3}, fp_t{0}, fp_t{4}, fp_t{-4}, fp_t{0}};
            jet.resize(36);

            jptr(jet.data(), nullptr, nullptr);

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

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prime(x) = par[0], prime(y) = par[1], prime(z) = par[2]}, 3, 3,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{0}, fp_t{3}, fp_t{-3}, fp_t{0}, fp_t{4}, fp_t{-4}, fp_t{0}};
            jet.resize(36);

            std::vector<fp_t> pars{fp_t{1}, fp_t{1}, fp_t{1}, fp_t{-2}, fp_t{-2}, fp_t{-2}, fp_t{0}, fp_t{0}, fp_t{0}};

            jptr(jet.data(), pars.data(), nullptr);

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
                                   opt_level, high_accuracy, compact_mode);
    };

    for (auto cm : {true, false}) {
        for (auto ha : {true, false}) {
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 0, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 1, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 2, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 3, ha, cm); });
        }
    }
}

// A test in which equations have an expression without variables
// at the end.
TEST_CASE("taylor end novars")
{
    using std::cos;
    using std::sin;

    auto x = "x"_var, y = "y"_var;

    llvm_state s;

    auto no_vars = expression{mul(2_dbl, 3_dbl)};
    auto dc = taylor_add_jet<double>(
        s, "jet", {sin(y) + cos(x) + sin(x) + cos(y) + no_vars, sin(y) + cos(x) + sin(x) + cos(y) + no_vars}, 2, 1,
        false, false);

    s.compile();

    auto jptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

    std::vector<double> jet{2., 3.};
    jet.resize(6);

    jptr(jet.data(), nullptr, nullptr);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1]) + 6));
    REQUIRE(jet[3] == approximately(sin(jet[1]) + cos(jet[0]) + sin(jet[0]) + cos(jet[1]) + 6));
    REQUIRE(jet[4]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
    REQUIRE(jet[5]
            == approximately((cos(jet[1]) * jet[3] - sin(jet[0]) * jet[2] + cos(jet[0]) * jet[2] - sin(jet[1]) * jet[3])
                             / 2));
}
