// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <initializer_list>
#include <random>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/sub.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

auto sub_wrapper(expression a, expression b)
{
    return detail::sub(std::move(a), std::move(b));
}

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
            REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx], T(1e4)));
        }
    }
}

// Test issue when two different c_diff functions would end up mangled with the same name.
TEST_CASE("multimangle")
{
    auto x = "x"_var, y = "y"_var;

    llvm_state s{kw::opt_level = 0u};

    taylor_add_jet<double>(s, "jet1", {sub_wrapper(2_dbl, 3_dbl), x + y}, 1, 1, false, true);

    s.compile();

    auto jptr1 = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet1"));

    std::vector<double> jet{double{2}, double{3}};
    jet.resize(4);

    jptr1(jet.data(), nullptr, nullptr);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == 3);
    REQUIRE(jet[2] == approximately(double{-1}));
    REQUIRE(jet[3] == approximately(double{5}));
}

TEST_CASE("taylor sub")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, 3_dbl), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{-1}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(par[0], 3_dbl), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{-1}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, 3_dbl), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, par[1]), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{3}, fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, 3_dbl), x + y}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == approximately(fp_t{5}));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] + jet[3])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, 3_dbl), x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == -1);
            REQUIRE(jet[5] == -1);
            REQUIRE(jet[6] == approximately(fp_t{5}));
            REQUIRE(jet[7] == approximately(-fp_t{5}));
            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);
            REQUIRE(jet[10] == approximately(fp_t(.5) * (jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t(.5) * (jet[5] + jet[7])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, 3_dbl), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{-1}, fp_t{3}, fp_t{2}, fp_t{4}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 2);
            REQUIRE(jet[5] == 4);
            REQUIRE(jet[6] == -1);
            REQUIRE(jet[7] == -1);
            REQUIRE(jet[8] == -1);
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(par[0], par[1]), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{-1}, fp_t{3}, fp_t{2}, fp_t{4}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{2}, fp_t{2}, fp_t{2}, fp_t{3}, fp_t{3}, fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == -1);
            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 2);
            REQUIRE(jet[5] == 4);
            REQUIRE(jet[6] == -1);
            REQUIRE(jet[7] == -1);
            REQUIRE(jet[8] == -1);
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({sub_wrapper(2_dbl, 3_dbl), x + y}, opt_level, high_accuracy, compact_mode);

        // Variable-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, par[0]), sub_wrapper(x, -4_dbl)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));

            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, par[1])}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{-4}, fp_t{-4}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));

            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, 2, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(jet[1] - 2));
            REQUIRE(jet[3] == approximately(jet[0] + 4));
            REQUIRE(jet[4] == approximately(fp_t(.5) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t(.5) * jet[2]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, 2, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(jet[2] - 2));
            REQUIRE(jet[5] == approximately(jet[3] - 2));
            REQUIRE(jet[6] == approximately(jet[0] + 4));
            REQUIRE(jet[7] == approximately(jet[1] + 4));
            REQUIRE(jet[8] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[9] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[10] == approximately(fp_t(.5) * jet[4]));
            REQUIRE(jet[11] == approximately(fp_t(.5) * jet[5]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(jet[3] - 2));
            REQUIRE(jet[7] == approximately(jet[4] - 2));
            REQUIRE(jet[8] == approximately(jet[5] - 2));

            REQUIRE(jet[9] == approximately(jet[0] + 4));
            REQUIRE(jet[10] == approximately(jet[1] + 4));
            REQUIRE(jet[11] == approximately(jet[2] + 4));

            REQUIRE(jet[12] == approximately(fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * jet[14]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(y, par[0]), sub_wrapper(x, par[1])}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{2}, fp_t{2}, fp_t{2}, fp_t{-4}, fp_t{-4}, fp_t{-4}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(jet[3] - 2));
            REQUIRE(jet[7] == approximately(jet[4] - 2));
            REQUIRE(jet[8] == approximately(jet[5] - 2));

            REQUIRE(jet[9] == approximately(jet[0] + 4));
            REQUIRE(jet[10] == approximately(jet[1] + 4));
            REQUIRE(jet[11] == approximately(jet[2] + 4));

            REQUIRE(jet[12] == approximately(fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * jet[14]));
        }

        compare_batch_scalar<fp_t>({sub_wrapper(y, 2_dbl), sub_wrapper(x, -4_dbl)}, opt_level, high_accuracy,
                                   compact_mode);

        // Number/variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(2 - jet[1]));
            REQUIRE(jet[3] == approximately(-4 - jet[0]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(par[0], y), sub_wrapper(-4_dbl, x)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(2 - jet[1]));
            REQUIRE(jet[3] == approximately(-4 - jet[0]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));

            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(par[1], x)}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{-4}, fp_t{-4}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));

            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, 2, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(-jet[1] + 2));
            REQUIRE(jet[3] == approximately(-jet[0] - 4));
            REQUIRE(jet[4] == approximately(-fp_t(.5) * jet[3]));
            REQUIRE(jet[5] == approximately(-fp_t(.5) * jet[2]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, 2, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(-jet[2] + 2));
            REQUIRE(jet[5] == approximately(-jet[3] + 2));
            REQUIRE(jet[6] == approximately(-jet[0] - 4));
            REQUIRE(jet[7] == approximately(-jet[1] - 4));
            REQUIRE(jet[8] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[9] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[10] == approximately(-fp_t(.5) * jet[4]));
            REQUIRE(jet[11] == approximately(-fp_t(.5) * jet[5]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(-jet[3] + 2));
            REQUIRE(jet[7] == approximately(-jet[4] + 2));
            REQUIRE(jet[8] == approximately(-jet[5] + 2));

            REQUIRE(jet[9] == approximately(-jet[0] - 4));
            REQUIRE(jet[10] == approximately(-jet[1] - 4));
            REQUIRE(jet[11] == approximately(-jet[2] - 4));

            REQUIRE(jet[12] == approximately(-fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(-fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(-fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(-fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(-1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(-1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(-1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{3} * jet[14]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(par[0], y), sub_wrapper(par[1], x)}, 3, 3, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{0}, fp_t{3}, fp_t{4}, fp_t{-5}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{2}, fp_t{2}, fp_t{2}, fp_t{-4}, fp_t{-4}, fp_t{-4}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 0);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 4);
            REQUIRE(jet[5] == -5);

            REQUIRE(jet[6] == approximately(-jet[3] + 2));
            REQUIRE(jet[7] == approximately(-jet[4] + 2));
            REQUIRE(jet[8] == approximately(-jet[5] + 2));

            REQUIRE(jet[9] == approximately(-jet[0] - 4));
            REQUIRE(jet[10] == approximately(-jet[1] - 4));
            REQUIRE(jet[11] == approximately(-jet[2] - 4));

            REQUIRE(jet[12] == approximately(-fp_t(.5) * jet[9]));
            REQUIRE(jet[13] == approximately(-fp_t(.5) * jet[10]));
            REQUIRE(jet[14] == approximately(-fp_t(.5) * jet[11]));

            REQUIRE(jet[15] == approximately(-fp_t(.5) * jet[6]));
            REQUIRE(jet[16] == approximately(-fp_t(.5) * jet[7]));
            REQUIRE(jet[17] == approximately(-fp_t(.5) * jet[8]));

            REQUIRE(jet[18] == approximately(-1 / fp_t{3} * jet[15]));
            REQUIRE(jet[19] == approximately(-1 / fp_t{3} * jet[16]));
            REQUIRE(jet[20] == approximately(-1 / fp_t{3} * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{3} * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{3} * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{3} * jet[14]));
        }

        compare_batch_scalar<fp_t>({sub_wrapper(2_dbl, y), sub_wrapper(-4_dbl, x)}, opt_level, high_accuracy,
                                   compact_mode);

        // Variable/variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(x, y), sub_wrapper(y, x)}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == jet[0] - jet[1]);
            REQUIRE(jet[3] == -jet[0] + jet[1]);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(x, y), sub_wrapper(y, x)}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == jet[0] - jet[2]);
            REQUIRE(jet[5] == jet[1] - jet[3]);

            REQUIRE(jet[6] == -jet[0] + jet[2]);
            REQUIRE(jet[7] == -jet[1] + jet[3]);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(x, y), sub_wrapper(y, x)}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == jet[0] - jet[1]);
            REQUIRE(jet[3] == -jet[0] + jet[1]);
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * (jet[2] - jet[3])));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (-jet[2] + jet[3])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(x, y), sub_wrapper(y, x)}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == jet[0] - jet[2]);
            REQUIRE(jet[5] == jet[1] - jet[3]);

            REQUIRE(jet[6] == -jet[0] + jet[2]);
            REQUIRE(jet[7] == -jet[1] + jet[3]);

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * (jet[4] - jet[6])));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * (jet[5] - jet[7])));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (-jet[4] + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (-jet[5] + jet[7])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {sub_wrapper(x, y), sub_wrapper(y, x)}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{3}, fp_t{-4}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);
            REQUIRE(jet[2] == 3);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -4);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == jet[0] - jet[3]);
            REQUIRE(jet[7] == jet[1] - jet[4]);
            REQUIRE(jet[8] == jet[2] - jet[5]);

            REQUIRE(jet[9] == -jet[0] + jet[3]);
            REQUIRE(jet[10] == -jet[1] + jet[4]);
            REQUIRE(jet[11] == -jet[2] + jet[5]);

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * (jet[6] - jet[9])));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * (jet[7] - jet[10])));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * (jet[8] - jet[11])));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (-jet[6] + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (-jet[7] + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (-jet[8] + jet[11])));

            REQUIRE(jet[18] == approximately(1 / fp_t{3} * (jet[12] - jet[15])));
            REQUIRE(jet[19] == approximately(1 / fp_t{3} * (jet[13] - jet[16])));
            REQUIRE(jet[20] == approximately(1 / fp_t{3} * (jet[14] - jet[17])));

            REQUIRE(jet[21] == approximately(1 / fp_t{3} * (-jet[12] + jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{3} * (-jet[13] + jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{3} * (-jet[14] + jet[17])));
        }

        compare_batch_scalar<fp_t>({sub_wrapper(x, y), sub_wrapper(y, x)}, opt_level, high_accuracy, compact_mode);
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
