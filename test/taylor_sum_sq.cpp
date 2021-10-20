// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum_sq.hpp>
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

template <typename T, typename U>
void compare_batch_scalar(std::initializer_list<U> sys, unsigned opt_level, bool high_accuracy, bool compact_mode)
{
    for (auto batch_size : {2u, 4u, 8u, 23u}) {
        llvm_state s{kw::opt_level = opt_level};

        taylor_add_jet<T>(s, "jet_batch", sys, 3, batch_size, high_accuracy, compact_mode);
        taylor_add_jet<T>(s, "jet_scalar", sys, 3, 1, high_accuracy, compact_mode);

        s.compile();

        auto jptr_batch = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_batch"));
        auto jptr_scalar = reinterpret_cast<void (*)(T *, const T *, const T *)>(s.jit_lookup("jet_scalar"));

        std::vector<T> jet_batch;
        jet_batch.resize(8 * batch_size);
        std::uniform_real_distribution<float> dist(.1f, 20.f);
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
                REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx]));
            }
        }
    }
}

TEST_CASE("taylor sum_sq")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 14);
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({1_dbl, par[0], 2_dbl}), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 14);
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 14);
            REQUIRE(jet[5] == 14);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({1_dbl, par[1], 2_dbl}), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{2}, fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 9);
            REQUIRE(jet[5] == 9);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 14);
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + 14)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == 14);
            REQUIRE(jet[5] == 14);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + 14)));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + 14)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == 14);
            REQUIRE(jet[7] == 14);
            REQUIRE(jet[8] == 14);

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + 14)));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + 14)));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + 14)));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({1_dbl, par[0], 2_dbl}), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{3}, fp_t{3}, fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == 14);
            REQUIRE(jet[7] == 14);
            REQUIRE(jet[8] == 9);

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + 14)));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + 14)));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + 9)));

            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({sum_sq({2_dbl, 3_dbl, 1_dbl}), x + y}, opt_level, high_accuracy, compact_mode);

        // Variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 14);
            REQUIRE(jet[3] == 17);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == 2 * 2 + 3 * 3 + 1);
            REQUIRE(jet[5] == 4 * 4 + 5 * 5 + 1);

            REQUIRE(jet[6] == 2 * 2 + 3 * 3 + 4);
            REQUIRE(jet[7] == 4 * 4 + 5 * 5 + 4);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, 2, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == 14);
            REQUIRE(jet[3] == 17);
            REQUIRE(jet[4] == jet[1] * jet[3] + jet[0] * jet[2]);
            REQUIRE(jet[5] == jet[1] * jet[3] + jet[0] * jet[2]);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, 2, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{4}, fp_t{3}, fp_t{5}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == 2 * 2 + 3 * 3 + 1);
            REQUIRE(jet[5] == 4 * 4 + 5 * 5 + 1);

            REQUIRE(jet[6] == 2 * 2 + 3 * 3 + 4);
            REQUIRE(jet[7] == 4 * 4 + 5 * 5 + 4);

            REQUIRE(jet[8] == jet[2] * jet[6] + jet[0] * jet[4]);
            REQUIRE(jet[9] == jet[3] * jet[7] + jet[1] * jet[5]);

            REQUIRE(jet[10] == jet[2] * jet[6] + jet[0] * jet[4]);
            REQUIRE(jet[11] == jet[3] * jet[7] + jet[1] * jet[5]);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{4}, fp_t{3}, fp_t{3}, fp_t{5}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 4);
            REQUIRE(jet[2] == 3);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == jet[3] * jet[3] + jet[0] * jet[0] + 1);
            REQUIRE(jet[7] == jet[4] * jet[4] + jet[1] * jet[1] + 1);
            REQUIRE(jet[8] == jet[5] * jet[5] + jet[2] * jet[2] + 1);

            REQUIRE(jet[9] == jet[3] * jet[3] + jet[0] * jet[0] + 4);
            REQUIRE(jet[10] == jet[4] * jet[4] + jet[1] * jet[1] + 4);
            REQUIRE(jet[11] == jet[5] * jet[5] + jet[2] * jet[2] + 4);

            REQUIRE(jet[12] == jet[3] * jet[9] + jet[0] * jet[6]);
            REQUIRE(jet[13] == jet[4] * jet[10] + jet[1] * jet[7]);
            REQUIRE(jet[14] == jet[5] * jet[11] + jet[2] * jet[8]);

            REQUIRE(jet[15] == jet[3] * jet[9] + jet[0] * jet[6]);
            REQUIRE(jet[16] == jet[4] * jet[10] + jet[1] * jet[7]);
            REQUIRE(jet[17] == jet[5] * jet[11] + jet[2] * jet[8]);

            REQUIRE(
                jet[18]
                == approximately(fp_t{1} / 3
                                 * (jet[9] * jet[9] + jet[3] * 2 * jet[15] + jet[6] * jet[6] + jet[0] * 2 * jet[12])));
            REQUIRE(
                jet[19]
                == approximately(
                    fp_t{1} / 3 * (jet[10] * jet[10] + jet[4] * 2 * jet[16] + jet[7] * jet[7] + jet[1] * 2 * jet[13])));
            REQUIRE(
                jet[20]
                == approximately(
                    fp_t{1} / 3 * (jet[11] * jet[11] + jet[5] * 2 * jet[17] + jet[8] * jet[8] + jet[2] * 2 * jet[14])));

            REQUIRE(
                jet[21]
                == approximately(fp_t{1} / 3
                                 * (jet[9] * jet[9] + jet[3] * 2 * jet[15] + jet[6] * jet[6] + jet[0] * 2 * jet[12])));
            REQUIRE(
                jet[22]
                == approximately(
                    fp_t{1} / 3 * (jet[10] * jet[10] + jet[4] * 2 * jet[16] + jet[7] * jet[7] + jet[1] * 2 * jet[13])));
            REQUIRE(
                jet[23]
                == approximately(
                    fp_t{1} / 3 * (jet[11] * jet[11] + jet[5] * 2 * jet[17] + jet[8] * jet[8] + jet[2] * 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({sum_sq({y, x, 1_dbl}), sum_sq({x, y, 2_dbl})}, opt_level, high_accuracy,
                                   compact_mode);
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
