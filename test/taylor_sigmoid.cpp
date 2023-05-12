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
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sigmoid.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
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
#if LLVM_VERSION_MAJOR >= 13 && LLVM_VERSION_MAJOR <= 16
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

template <typename T, typename U>
void compare_batch_scalar(std::initializer_list<U> sys, unsigned opt_level, bool high_accuracy, bool compact_mode)
{
    for (auto batch_size : {2u, 4u, 8u, 5u}) {
        llvm_state s{kw::opt_level = opt_level};

        taylor_add_jet<T>(s, "jet_batch", sys, 3, batch_size, high_accuracy, compact_mode);
        taylor_add_jet<T>(s, "jet_scalar", sys, 3, 1, high_accuracy, compact_mode);

        s.compile();

        auto jptr_batch = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_batch"));
        auto jptr_scalar = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_scalar"));

        std::vector<T> jet_batch;
        jet_batch.resize(8 * batch_size);
        std::uniform_real_distribution<float> dist(-10.f, 10.f);
        std::generate(jet_batch.begin(), jet_batch.end(), [&dist]() { return T{dist(rng)}; });

        std::vector<T> jet_scalar;
        jet_scalar.resize(8);

        jptr_batch(jet_batch.data(), nullptr, nullptr);

        for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
            // Assign the initial values of x and y.
            for (auto i = 0u; i < 2u; ++i) {
                jet_scalar[i] = jet_batch[i * batch_size + batch_idx];
            }

            jptr_scalar(jet_scalar.data(), nullptr, nullptr);

            for (auto i = 2u; i < 8u; ++i) {
                REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx], T(10000)));
            }
        }
    }
}

double sigmoid(double x)
{
    return 1. / (1. + std::exp(-x));
}

long double sigmoid(long double x)
{
    return 1. / (1. + std::exp(-x));
}

#if defined(HEYOKA_HAVE_REAL128)

mppp::real128 sigmoid(mppp::real128 x)
{
    return 1. / (1. + mppp::exp(-x));
}

#endif

TEST_CASE("ode test")
{
    using std::abs;

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto [x, s] = make_vars("x", "s");

                taylor_adaptive<double> ta0({prime(x) = sigmoid(1e-2 * x) + x}, {.5}, kw::high_accuracy = ha,
                                            kw::compact_mode = cm, kw::opt_level = opt_level);
                taylor_adaptive<double> ta1({prime(x) = s + x, prime(s) = s * (1_dbl - s) * 1e-2 * (s + x)},
                                            {.5, sigmoid(1e-2 * .5)}, kw::high_accuracy = ha, kw::compact_mode = cm,
                                            kw::opt_level = opt_level);

                ta0.propagate_until(5.);
                ta1.propagate_until(5.);

                REQUIRE(abs((ta0.get_state()[0] - ta1.get_state()[0]) / ta0.get_state()[0]) < 1e-14);

                const auto v0 = sigmoid(ta0.get_state()[0] * 1e-2);
                const auto v1 = ta1.get_state()[1];

                REQUIRE(abs((v0 - v1) / v0) < 1e-14);
            }
        }
    }
}

// Test CSE involving hidden dependencies.
TEST_CASE("taylor sigmoid test simplifications")
{
    auto x = "x"_var, y = "y"_var;

    llvm_state s{kw::opt_level = 0u};

    taylor_add_jet<double>(s, "jet", {sigmoid(x + y) * sigmoid(x + y) + sigmoid(x + y), x}, 2, 1, false, false);

    s.compile();

    auto jptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

    std::vector<double> jet{double{2}, double{3}};
    jet.resize(6);

    jptr(jet.data(), nullptr, nullptr);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(sigmoid(jet[0] + jet[1]) * sigmoid(jet[0] + jet[1]) + sigmoid(jet[0] + jet[1])));
    REQUIRE(jet[3] == jet[0]);
    REQUIRE(jet[4]
            == approximately(.5
                             * (2. * sigmoid(jet[0] + jet[1]) * (1. - sigmoid(jet[0] + jet[1]))
                                    * sigmoid(jet[0] + jet[1]) * (jet[2] + jet[3])
                                + (1. - sigmoid(jet[0] + jet[1])) * sigmoid(jet[0] + jet[1]) * (jet[2] + jet[3]))));
    REQUIRE(jet[5] == approximately(.5 * jet[2]));
}

TEST_CASE("taylor sigmoid")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(expression{number{fp_t{2}}}), x + y}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(par[0]), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(expression{number{fp_t{2}}}), x + y}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[5] == approximately(sigmoid(fp_t{2})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(par[1]), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sigmoid(fp_t{3})));
            REQUIRE(jet[5] == approximately(sigmoid(fp_t{3})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(expression{number{fp_t{2}}}), x + y}, 2, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[3] == approximately(jet[0] + jet[1]));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] + jet[3])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(expression{number{fp_t{2}}}), x + y}, 2, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-4}, fp_t{3}, fp_t{5}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[5] == approximately(sigmoid(fp_t{2})));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[5] + jet[7])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(expression{number{fp_t{2}}}), x + y}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[7] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[8] == approximately(sigmoid(fp_t{2})));

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(par[0]), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-4}, fp_t{-1}, fp_t{3}, fp_t{5}, fp_t{-2}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{2}, fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}, fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -4);
            REQUIRE(jet[2] == -1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == -2);

            REQUIRE(jet[6] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[7] == approximately(sigmoid(fp_t{2})));
            REQUIRE(jet[8] == approximately(sigmoid(fp_t{2})));

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
            compare_batch_scalar<fp_t>({sigmoid(expression{number{fp_t{2}}}), x + y}, opt_level, high_accuracy,
                                       compact_mode);
        }

        // Variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(y), sigmoid(x)}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sigmoid(jet[1])));
            REQUIRE(jet[3] == approximately(sigmoid(jet[0])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(y), sigmoid(x)}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sigmoid(jet[2])));
            REQUIRE(jet[5] == approximately(sigmoid(jet[3])));

            REQUIRE(jet[6] == approximately(sigmoid(jet[0])));
            REQUIRE(jet[7] == approximately(sigmoid(jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(y), sigmoid(x)}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(sigmoid(jet[1])));
            REQUIRE(jet[3] == approximately(sigmoid(jet[0])));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * ((1 - jet[2]) * jet[2]) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * ((1 - jet[3]) * jet[3]) * jet[2]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(y), sigmoid(x)}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(sigmoid(jet[2])));
            REQUIRE(jet[5] == approximately(sigmoid(jet[3])));

            REQUIRE(jet[6] == approximately(sigmoid(jet[0])));
            REQUIRE(jet[7] == approximately(sigmoid(jet[1])));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[4] * (1 - jet[4]) * jet[6]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[5] * (1 - jet[5]) * jet[7]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * jet[6] * (1 - jet[6]) * jet[4]));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * jet[7] * (1 - jet[7]) * jet[5]));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sigmoid(y), sigmoid(x)}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{-5}, fp_t{3}, fp_t{-4}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -5);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(sigmoid(jet[3])));
            REQUIRE(jet[7] == approximately(sigmoid(jet[4])));
            REQUIRE(jet[8] == approximately(sigmoid(jet[5])));

            REQUIRE(jet[9] == approximately(sigmoid(jet[0])));
            REQUIRE(jet[10] == approximately(sigmoid(jet[1])));
            REQUIRE(jet[11] == approximately(sigmoid(jet[2])));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[6] * (1 - jet[6]) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[7] * (1 - jet[7]) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[8] * (1 - jet[8]) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * jet[9] * (1 - jet[9]) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * jet[10] * (1 - jet[10]) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * jet[11] * (1 - jet[11]) * jet[8]));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[6]) * jet[6] * (1 - jet[6]) * jet[9] * jet[9]
                                        + (jet[6] * (1 - jet[6])) * (1 - jet[9]) * jet[9] * jet[6])));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[7]) * jet[7] * (1 - jet[7]) * jet[10] * jet[10]
                                        + (jet[7] * (1 - jet[7])) * (1 - jet[10]) * jet[10] * jet[7])));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[8]) * jet[8] * (1 - jet[8]) * jet[11] * jet[11]
                                        + (jet[8] * (1 - jet[8])) * (1 - jet[11]) * jet[11] * jet[8])));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[9]) * jet[9] * (1 - jet[9]) * jet[6] * jet[6]
                                        + (jet[9] * (1 - jet[9])) * (1 - jet[6]) * jet[6] * jet[9])));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[10]) * jet[10] * (1 - jet[10]) * jet[7] * jet[7]
                                        + (jet[10] * (1 - jet[10])) * (1 - jet[7]) * jet[7] * jet[10])));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6
                                     * ((1 - 2 * jet[11]) * jet[11] * (1 - jet[11]) * jet[8] * jet[8]
                                        + (jet[11] * (1 - jet[11])) * (1 - jet[8]) * jet[8] * jet[11])));
        }

        if constexpr (!std::is_same_v<long double, fp_t> || !skip_batch_ld) {
            // Do the batch/scalar comparison.
            compare_batch_scalar<fp_t>({sigmoid(y), sigmoid(x)}, opt_level, high_accuracy, compact_mode);
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
