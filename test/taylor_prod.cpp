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
#include <stdexcept>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/prod.hpp>
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

auto prod_mul_wrapper(const expression &x, const expression &y)
{
    return expression{func{detail::prod_impl({x, y})}};
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

TEST_CASE("taylor mul")
{
    using Catch::Matchers::Message;

    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, 3_dbl), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(par[0], 3_dbl), x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, 3_dbl), x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == 6);
            REQUIRE(jet[5] == 6);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, par[1]), x + y}, 1, 2, high_accuracy, compact_mode);

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
            REQUIRE(jet[4] == 6);
            REQUIRE(jet[5] == 6);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(-1_dbl, par[1]), x + y}, 1, 2, high_accuracy,
                                 compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod_neg."));
            }

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
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == -3);
            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, 3_dbl), x + y}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[3])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(-1_dbl, par[0]), x + y}, 2, 1, high_accuracy,
                                 compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod_neg."));
            }

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            std::vector<fp_t> pars{fp_t{-2}};
            jet.resize(6);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{2}));
            REQUIRE(jet[3] == approximately(fp_t{5}));
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (fp_t{2} + jet[3])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, 3_dbl), x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);
            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(fp_t{6}));
            REQUIRE(jet[6] == approximately(fp_t{5}));
            REQUIRE(jet[7] == approximately(-fp_t{5}));
            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);
            REQUIRE(jet[10] == approximately(fp_t(.5) * (fp_t{6} + jet[6])));
            REQUIRE(jet[11] == approximately(fp_t(.5) * (fp_t{6} + jet[7])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(2_dbl, 3_dbl), x + y}, 3, 3, high_accuracy, compact_mode);

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
            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{6}));
            REQUIRE(jet[8] == approximately(fp_t{6}));
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {prod_mul_wrapper(par[0], par[1]), x + y}, 3, 3, high_accuracy,
                                 compact_mode);

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
            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{6}));
            REQUIRE(jet[8] == approximately(fp_t{6}));
            REQUIRE(jet[9] == approximately(fp_t{5}));
            REQUIRE(jet[10] == approximately(fp_t{0}));
            REQUIRE(jet[11] == approximately(fp_t{3}));
            REQUIRE(jet[12] == 0);
            REQUIRE(jet[13] == 0);
            REQUIRE(jet[14] == 0);
            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[9])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[10])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (fp_t{6} + jet[11])));
            REQUIRE(jet[18] == 0);
            REQUIRE(jet[19] == 0);
            REQUIRE(jet[20] == 0);
            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[15])));
            REQUIRE(jet[22] == approximately(1 / fp_t{6} * (2 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prod_mul_wrapper(2_dbl, 3_dbl), x + y}, opt_level, high_accuracy, compact_mode);

        // Variable-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, -4_dbl})}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, par[0]})}}, expression{func{detail::prod_impl({x, -4_dbl})}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(-fp_t{8}));

            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(-fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, par[1]})}}}, 1,
                2, high_accuracy, compact_mode);

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

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(-fp_t{8}));

            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(-fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, -4_dbl})}}}, 2,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
            REQUIRE(jet[4] == approximately(jet[3]));
            REQUIRE(jet[5] == approximately(-2 * jet[2]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, -4_dbl})}}}, 2,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(fp_t{8}));
            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(fp_t{4}));
            REQUIRE(jet[8] == approximately(jet[6]));
            REQUIRE(jet[9] == approximately(jet[7]));
            REQUIRE(jet[10] == approximately(-2 * jet[4]));
            REQUIRE(jet[11] == approximately(-2 * jet[5]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, -4_dbl})}}}, 3,
                3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{8}));
            REQUIRE(jet[8] == approximately(fp_t{-10}));

            REQUIRE(jet[9] == approximately(-fp_t{8}));
            REQUIRE(jet[10] == approximately(fp_t{4}));
            REQUIRE(jet[11] == approximately(fp_t{0}));

            REQUIRE(jet[12] == approximately(jet[9]));
            REQUIRE(jet[13] == approximately(jet[10]));
            REQUIRE(jet[14] == approximately(jet[11]));

            REQUIRE(jet[15] == approximately(-2 * jet[6]));
            REQUIRE(jet[16] == approximately(-2 * jet[7]));
            REQUIRE(jet[17] == approximately(-2 * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{6} * 4 * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{6} * 4 * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{6} * 4 * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{6} * 8 * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{6} * 8 * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{6} * 8 * jet[14]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({y, par[0]})}}, expression{func{detail::prod_impl({x, par[1]})}}}, 3,
                3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{8}));
            REQUIRE(jet[8] == approximately(fp_t{-10}));

            REQUIRE(jet[9] == approximately(-fp_t{8}));
            REQUIRE(jet[10] == approximately(fp_t{4}));
            REQUIRE(jet[11] == approximately(fp_t{0}));

            REQUIRE(jet[12] == approximately(jet[9]));
            REQUIRE(jet[13] == approximately(jet[10]));
            REQUIRE(jet[14] == approximately(jet[11]));

            REQUIRE(jet[15] == approximately(-2 * jet[6]));
            REQUIRE(jet[16] == approximately(-2 * jet[7]));
            REQUIRE(jet[17] == approximately(-2 * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{6} * 4 * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{6} * 4 * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{6} * 4 * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{6} * 8 * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{6} * 8 * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{6} * 8 * jet[14]));
        }

        compare_batch_scalar<fp_t>(
            {expression{func{detail::prod_impl({y, 2_dbl})}}, expression{func{detail::prod_impl({x, -4_dbl})}}},
            opt_level, high_accuracy, compact_mode);

        // Number/variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({-1_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                1, high_accuracy, compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod_neg."));
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod."));
            }

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{-3}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({par[0], y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{2}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(fp_t{8}));

            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({-1_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 1,
                2, high_accuracy, compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod_neg."));
                REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.prod."));
            }

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);

            REQUIRE(jet[4] == approximately(fp_t{-3}));
            REQUIRE(jet[5] == approximately(fp_t{-4}));

            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({par[1], x})}}}, 1,
                2, high_accuracy, compact_mode);

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

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(fp_t{8}));

            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 2,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(-fp_t{8}));
            REQUIRE(jet[4] == approximately(jet[3]));
            REQUIRE(jet[5] == approximately(-2 * jet[2]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 2,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 4);
            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(fp_t{8}));
            REQUIRE(jet[6] == approximately(-fp_t{8}));
            REQUIRE(jet[7] == approximately(fp_t{4}));
            REQUIRE(jet[8] == approximately(jet[6]));
            REQUIRE(jet[9] == approximately(jet[7]));
            REQUIRE(jet[10] == approximately(-2 * jet[4]));
            REQUIRE(jet[11] == approximately(-2 * jet[5]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}}, 3,
                3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{8}));
            REQUIRE(jet[8] == approximately(fp_t{-10}));

            REQUIRE(jet[9] == approximately(-fp_t{8}));
            REQUIRE(jet[10] == approximately(fp_t{4}));
            REQUIRE(jet[11] == approximately(fp_t{0}));

            REQUIRE(jet[12] == approximately(jet[9]));
            REQUIRE(jet[13] == approximately(jet[10]));
            REQUIRE(jet[14] == approximately(jet[11]));

            REQUIRE(jet[15] == approximately(-2 * jet[6]));
            REQUIRE(jet[16] == approximately(-2 * jet[7]));
            REQUIRE(jet[17] == approximately(-2 * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{6} * 4 * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{6} * 4 * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{6} * 4 * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{6} * 8 * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{6} * 8 * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{6} * 8 * jet[14]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet",
                {expression{func{detail::prod_impl({par[0], y})}}, expression{func{detail::prod_impl({par[1], x})}}}, 3,
                3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(fp_t{8}));
            REQUIRE(jet[8] == approximately(fp_t{-10}));

            REQUIRE(jet[9] == approximately(-fp_t{8}));
            REQUIRE(jet[10] == approximately(fp_t{4}));
            REQUIRE(jet[11] == approximately(fp_t{0}));

            REQUIRE(jet[12] == approximately(jet[9]));
            REQUIRE(jet[13] == approximately(jet[10]));
            REQUIRE(jet[14] == approximately(jet[11]));

            REQUIRE(jet[15] == approximately(-2 * jet[6]));
            REQUIRE(jet[16] == approximately(-2 * jet[7]));
            REQUIRE(jet[17] == approximately(-2 * jet[8]));

            REQUIRE(jet[18] == approximately(1 / fp_t{6} * 4 * jet[15]));
            REQUIRE(jet[19] == approximately(1 / fp_t{6} * 4 * jet[16]));
            REQUIRE(jet[20] == approximately(1 / fp_t{6} * 4 * jet[17]));

            REQUIRE(jet[21] == approximately(-1 / fp_t{6} * 8 * jet[12]));
            REQUIRE(jet[22] == approximately(-1 / fp_t{6} * 8 * jet[13]));
            REQUIRE(jet[23] == approximately(-1 / fp_t{6} * 8 * jet[14]));
        }

        compare_batch_scalar<fp_t>(
            {expression{func{detail::prod_impl({2_dbl, y})}}, expression{func{detail::prod_impl({-4_dbl, x})}}},
            opt_level, high_accuracy, compact_mode);

        // Variable/variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet", {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, 1,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(fp_t{6}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet", {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, 1,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(-fp_t{4}));

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(-fp_t{4}));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet", {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, 2,
                1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(fp_t{6}));
            REQUIRE(jet[3] == approximately(fp_t{6}));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * (jet[2] * 3 + jet[3] * 2)));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[2] * 3 + jet[3] * 2)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet", {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, 2,
                2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{1}, fp_t{3}, fp_t{-4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -4);

            REQUIRE(jet[4] == approximately(fp_t{6}));
            REQUIRE(jet[5] == approximately(-fp_t{4}));

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(-fp_t{4}));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * (jet[4] * 3 + jet[6] * 2)));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * (jet[5] * -4 + jet[7] * 1)));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[4] * 3 + jet[6] * 2)));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[5] * -4 + jet[7] * 1)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(
                s, "jet", {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, 3,
                3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(fp_t{6}));
            REQUIRE(jet[7] == approximately(-fp_t{4}));
            REQUIRE(jet[8] == approximately(fp_t{18}));

            REQUIRE(jet[9] == approximately(fp_t{6}));
            REQUIRE(jet[10] == approximately(-fp_t{4}));
            REQUIRE(jet[11] == approximately(fp_t{18}));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * (jet[6] * 3 + jet[9] * 2)));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * (jet[7] * -4 + jet[10] * 1)));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * (jet[8] * 6 + jet[11] * 3)));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[6] * 3 + jet[9] * 2)));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[7] * -4 + jet[10] * 1)));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[8] * 6 + jet[11] * 3)));

            REQUIRE(jet[18] == approximately(1 / fp_t{6} * (2 * jet[12] * 3 + 2 * jet[6] * jet[9] + 2 * 2 * jet[15])));
            REQUIRE(jet[19]
                    == approximately(1 / fp_t{6} * (2 * jet[13] * -4 + 2 * jet[7] * jet[10] + 2 * 1 * jet[16])));
            REQUIRE(jet[20] == approximately(1 / fp_t{6} * (2 * jet[14] * 6 + 2 * jet[8] * jet[11] + 2 * 3 * jet[17])));

            REQUIRE(jet[21] == approximately(1 / fp_t{6} * (2 * jet[12] * 3 + 2 * jet[6] * jet[9] + 2 * 2 * jet[15])));
            REQUIRE(jet[22]
                    == approximately(1 / fp_t{6} * (2 * jet[13] * -4 + 2 * jet[7] * jet[10] + 2 * 1 * jet[16])));
            REQUIRE(jet[23] == approximately(1 / fp_t{6} * (2 * jet[14] * 6 + 2 * jet[8] * jet[11] + 2 * 3 * jet[17])));
        }

        compare_batch_scalar<fp_t>(
            {expression{func{detail::prod_impl({x, y})}}, expression{func{detail::prod_impl({y, x})}}}, opt_level,
            high_accuracy, compact_mode);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }

    // Error throwing if prod() is not a binary function.
    {
        llvm_state s;
        REQUIRE_THROWS_MATCHES(taylor_add_jet<double>(s, "jet",
                                                      {prime("x"_var) = expression{func{detail::prod_impl()}}}, 3, 3,
                                                      false, false),
                               std::invalid_argument,
                               Message("The Taylor derivative of a product can be computed only for products "
                                       "of 2 terms, but the current product has 0 term(s) instead"));
    }
    {
        llvm_state s;
        REQUIRE_THROWS_MATCHES(
            taylor_add_jet<double>(s, "jet", {prime("x"_var) = expression{func{detail::prod_impl()}}}, 3, 3, false,
                                   true),
            std::invalid_argument,
            Message("The Taylor derivative of a product in compact mode can be computed only for products "
                    "of 2 terms, but the current product has 0 term(s) instead"));
    }
    {
        llvm_state s;
        REQUIRE_THROWS_MATCHES(taylor_add_jet<double>(s, "jet",
                                                      {prime("x"_var) = expression{func{detail::prod_impl({par[0]})}}},
                                                      3, 3, false, false),
                               std::invalid_argument,
                               Message("The Taylor derivative of a product can be computed only for products "
                                       "of 2 terms, but the current product has 1 term(s) instead"));
    }
    {
        llvm_state s;
        REQUIRE_THROWS_MATCHES(
            taylor_add_jet<double>(s, "jet", {prime("x"_var) = expression{func{detail::prod_impl({par[0]})}}}, 3, 3,
                                   false, true),
            std::invalid_argument,
            Message("The Taylor derivative of a product in compact mode can be computed only for products "
                    "of 2 terms, but the current product has 1 term(s) instead"));
    }
}
