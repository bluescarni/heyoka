// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
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
    for (auto batch_size : {2u, 4u, 8u, 5u}) {
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
                REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx], T(1000)));
            }
        }
    }
}

// Various tests to ensure that the approximated version of pow() is
// being used under certain conditions.
TEST_CASE("taylor pow approx")
{
    auto [x] = make_vars("x");

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", {pow(x, -1.5) + pow(x, 1 / 3.)}, 3, 1, false, false);

        REQUIRE(boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
        REQUIRE(boost::contains(s.get_ir(), "call contract double @llvm.pow"));
    }

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", std::vector{std::pair{x, pow(par[0], -1.5)}}, 3, 1, false, false);

        REQUIRE(boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
    }

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", std::vector{std::pair{x, pow(-1.5_dbl, par[0])}}, 3, 1, false, false);

        REQUIRE(!boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
    }

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", {pow(x, -1.5) + pow(x, 1 / 3.)}, 3, 1, false, true);

        REQUIRE(boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
        REQUIRE(boost::contains(s.get_ir(), "call contract double @llvm.pow"));

        REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.pow_approx."));
        REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.pow."));
    }

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", std::vector{std::pair{x, pow(par[0], -1.5)}}, 3, 1, false, true);

        REQUIRE(boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
        REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.pow_approx."));
    }

    {
        llvm_state s{kw::opt_level = 0};

        taylor_add_jet<double>(s, "jet", std::vector{std::pair{x, pow(-1.5_dbl, par[0])}}, 3, 1, false, true);

        REQUIRE(!boost::contains(s.get_ir(), "call contract afn double @llvm.pow"));
        REQUIRE(boost::contains(s.get_ir(), "taylor_c_diff.pow."));
    }
}

TEST_CASE("taylor pow")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::pow;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        // Number-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {pow(par[0], expression{number{fp_t{1} / fp_t{3}}}), x + y}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{3}};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 1, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {pow(expression{number{fp_t{3}}}, par[1]), x + y}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{1} / 3, fp_t{1} / 3};

            jptr(jet.data(), pars.data(), nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(fp_t{3}, fp_t{1} / 3)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 2, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y}, 3, 3,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

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

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[12] + 2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[13] + 2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[14] + 2 * jet[17])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {pow(par[0], par[1]), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{3}, fp_t{3}, fp_t{3}, fp_t{1} / 3, fp_t{1} / 3, fp_t{1} / 3};

            jptr(jet.data(), pars.data(), nullptr);

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

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[12] + 2 * jet[15])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[13] + 2 * jet[16])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[14] + 2 * jet[17])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({pow(expression{number{fp_t{3}}}, expression{number{fp_t{1} / fp_t{3}}}), x + y},
                                   opt_level, high_accuracy, compact_mode);

        // Variable-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                  pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, par[0]), pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{fp_t{3} / 2};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                  pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}), pow(x, par[1])}, 1,
                                 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, fp_t{-1} / 3, fp_t{-1} / 3};

            jptr(jet.data(), pars.data(), nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                  pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(pow(jet[1], fp_t{3} / 2)));
            REQUIRE(jet[3] == approximately(pow(jet[0], fp_t{-1} / 3)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * fp_t{3} / 2 * pow(jet[1], fp_t{1} / 2) * jet[3]));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * fp_t{-1} / 3 * pow(jet[0], fp_t{-4} / 3) * jet[2]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                  pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{3}, fp_t{4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

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
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet",
                                 {pow(y, expression{number{fp_t{3}}} / expression{number{fp_t{2}}}),
                                  pow(x, expression{number{fp_t{-1}}} / expression{number{fp_t{3}}})},
                                 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{1}, fp_t{3}, fp_t{4}, fp_t{6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

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

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {pow(y, par[0]), pow(x, par[1])}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{5}, fp_t{1}, fp_t{3}, fp_t{4}, fp_t{6}};
            jet.resize(24);

            std::vector<fp_t> pars{fp_t{3} / 2, fp_t{3} / 2, fp_t{3} / 2, fp_t{-1} / 3, fp_t{-1} / 3, fp_t{-1} / 3};

            jptr(jet.data(), pars.data(), nullptr);

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
                                   opt_level, high_accuracy, compact_mode);

        // Failure modes for non-implemented cases.
        {
            llvm_state s{kw::opt_level = opt_level};

            REQUIRE_THROWS_MATCHES(taylor_add_jet<fp_t>(s, "jet", {pow(1_dbl, x)}, 3, 3, high_accuracy, compact_mode),
                                   std::invalid_argument,
                                   Message(compact_mode
                                               ? "An invalid argument type was encountered while trying to build the "
                                                 "Taylor derivative of a pow() in compact mode"
                                               : "An invalid argument type was encountered while trying to build the "
                                                 "Taylor derivative of a pow()"));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            REQUIRE_THROWS_MATCHES(taylor_add_jet<fp_t>(s, "jet", {y, pow(y, x)}, 3, 3, high_accuracy, compact_mode),
                                   std::invalid_argument,
                                   Message(compact_mode
                                               ? "An invalid argument type was encountered while trying to build the "
                                                 "Taylor derivative of a pow() in compact mode"
                                               : "An invalid argument type was encountered while trying to build the "
                                                 "Taylor derivative of a pow()"));
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
