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
#include <heyoka/math/atan2.hpp>
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
        std::uniform_real_distribution<float> dist(.1f, 1.f);
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

TEST_CASE("taylor atan2")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::atan2;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        const auto a = fp_t{1} / 3;
        const auto b = 1 + fp_t{3} / fp_t{7};
        const auto c = fp_t{2} / 7;

        // Number-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, expression{number{b}}), x + y}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(par[0], expression{number{b}}), x + y}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, expression{number{b}}), x + y}, 1, 2,
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

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, par[1]), x + y}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, b, b};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, expression{number{b}}), x + y}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(atan2(a, b)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, expression{number{b}}), x + y}, 2, 2,
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

            REQUIRE(jet[4] == approximately(atan2(a, b)));
            REQUIRE(jet[5] == approximately(atan2(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == fp_t{1} / 2 * (jet[4] + jet[6]));
            REQUIRE(jet[11] == fp_t{1} / 2 * (jet[5] + jet[7]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(par[0], par[1]), x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{-4}, fp_t{3}, fp_t{5}, fp_t{6}};
            jet.resize(24);

            std::vector<fp_t> pars{a, a, a, b, b, b};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);
            REQUIRE(jet[2] == -4);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == 5);
            REQUIRE(jet[5] == 6);

            REQUIRE(jet[6] == approximately(atan2(a, b)));
            REQUIRE(jet[7] == approximately(atan2(a, b)));
            REQUIRE(jet[8] == approximately(atan2(a, b)));

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
        compare_batch_scalar<fp_t>({atan2(expression{number{a}}, expression{number{b}}), x + y}, opt_level,
                                   high_accuracy, compact_mode);

        // Variable-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{a}}), atan2(x, expression{number{b}})}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(atan2(jet[1], a)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, par[0]), atan2(x, expression{number{b}})}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(atan2(jet[1], a)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{b}}), atan2(x, expression{number{b}})}, 1, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{b}}), atan2(x, par[1])}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, b, b};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{b}}), atan2(x, expression{number{b}})}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(atan2(jet[1], b)));
            REQUIRE(jet[3] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * b * jet[3] / (jet[1] * jet[1] + b * b)));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * b * jet[2] / (jet[0] * jet[0] + b * b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{b}}), atan2(x, expression{number{b}})}, 2, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(jet[2], b)));
            REQUIRE(jet[5] == approximately(atan2(jet[3], b)));

            REQUIRE(jet[6] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[1], b)));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * b * jet[6] / (jet[2] * jet[2] + b * b)));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * b * jet[7] / (jet[3] * jet[3] + b * b)));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * b * jet[4] / (jet[0] * jet[0] + b * b)));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * b * jet[5] / (jet[1] * jet[1] + b * b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(y, expression{number{b}}), atan2(x, expression{number{b}})}, 3, 3,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.1}, fp_t{.3}, fp_t{.4}, fp_t{.6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);
            REQUIRE(jet[2] == .1);

            REQUIRE(jet[3] == .3);
            REQUIRE(jet[4] == .4);
            REQUIRE(jet[5] == .6);

            REQUIRE(jet[6] == approximately(atan2(jet[3], b)));
            REQUIRE(jet[7] == approximately(atan2(jet[4], b)));
            REQUIRE(jet[8] == approximately(atan2(jet[5], b)));

            REQUIRE(jet[9] == approximately(atan2(jet[0], b)));
            REQUIRE(jet[10] == approximately(atan2(jet[1], b)));
            REQUIRE(jet[11] == approximately(atan2(jet[2], b)));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * b * jet[9] / (jet[3] * jet[3] + b * b)));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * b * jet[10] / (jet[4] * jet[4] + b * b)));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * b * jet[11] / (jet[5] * jet[5] + b * b)));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * b * jet[6] / (jet[0] * jet[0] + b * b)));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * b * jet[7] / (jet[1] * jet[1] + b * b)));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * b * jet[8] / (jet[2] * jet[2] + b * b)));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[15] * (jet[3] * jet[3] + b * b) - b * jet[9] * 2 * jet[3] * jet[9])
                                     / ((jet[3] * jet[3] + b * b) * (jet[3] * jet[3] + b * b))));
            REQUIRE(jet[19]
                    == approximately(
                        fp_t{1} / 6 * (b * 2 * jet[16] * (jet[4] * jet[4] + b * b) - b * jet[10] * 2 * jet[4] * jet[10])
                        / ((jet[4] * jet[4] + b * b) * (jet[4] * jet[4] + b * b))));
            REQUIRE(jet[20]
                    == approximately(
                        fp_t{1} / 6 * (b * 2 * jet[17] * (jet[5] * jet[5] + b * b) - b * jet[11] * 2 * jet[5] * jet[11])
                        / ((jet[5] * jet[5] + b * b) * (jet[5] * jet[5] + b * b))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[12] * (jet[0] * jet[0] + b * b) - b * jet[6] * 2 * jet[0] * jet[6])
                                     / ((jet[0] * jet[0] + b * b) * (jet[0] * jet[0] + b * b))));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[13] * (jet[1] * jet[1] + b * b) - b * jet[7] * 2 * jet[1] * jet[7])
                                     / ((jet[1] * jet[1] + b * b) * (jet[1] * jet[1] + b * b))));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6
                                     * (b * 2 * jet[14] * (jet[2] * jet[2] + b * b) - b * jet[8] * 2 * jet[2] * jet[8])
                                     / ((jet[2] * jet[2] + b * b) * (jet[2] * jet[2] + b * b))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({atan2(y, expression{number{b}}), atan2(x, expression{number{b}})}, opt_level,
                                   high_accuracy, compact_mode);

        // Number-variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(par[0], y), atan2(expression{number{c}}, x)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, 1, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(par[1], x)}, 1, 2, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            std::vector<fp_t> pars{fp_t{0}, fp_t{0}, c, c};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(atan2(a, jet[1])));
            REQUIRE(jet[3] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[4] == approximately(-fp_t{1} / 2 * a * jet[3] / (jet[1] * jet[1] + a * a)));
            REQUIRE(jet[5] == approximately(-fp_t{1} / 2 * c * jet[2] / (jet[0] * jet[0] + c * c)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, 2, 2,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(a, jet[2])));
            REQUIRE(jet[5] == approximately(atan2(a, jet[3])));

            REQUIRE(jet[6] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[7] == approximately(atan2(c, jet[1])));

            REQUIRE(jet[8] == approximately(-fp_t{1} / 2 * a * jet[6] / (jet[2] * jet[2] + a * a)));
            REQUIRE(jet[9] == approximately(-fp_t{1} / 2 * a * jet[7] / (jet[3] * jet[3] + a * a)));

            REQUIRE(jet[10] == approximately(-fp_t{1} / 2 * c * jet[4] / (jet[0] * jet[0] + c * c)));
            REQUIRE(jet[11] == approximately(-fp_t{1} / 2 * c * jet[5] / (jet[1] * jet[1] + c * c)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, 3, 3,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.1}, fp_t{.3}, fp_t{.4}, fp_t{.6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);
            REQUIRE(jet[2] == .1);

            REQUIRE(jet[3] == .3);
            REQUIRE(jet[4] == .4);
            REQUIRE(jet[5] == .6);

            REQUIRE(jet[6] == approximately(atan2(a, jet[3])));
            REQUIRE(jet[7] == approximately(atan2(a, jet[4])));
            REQUIRE(jet[8] == approximately(atan2(a, jet[5])));

            REQUIRE(jet[9] == approximately(atan2(c, jet[0])));
            REQUIRE(jet[10] == approximately(atan2(c, jet[1])));
            REQUIRE(jet[11] == approximately(atan2(c, jet[2])));

            REQUIRE(jet[12] == approximately(-fp_t{1} / 2 * a * jet[9] / (jet[3] * jet[3] + a * a)));
            REQUIRE(jet[13] == approximately(-fp_t{1} / 2 * a * jet[10] / (jet[4] * jet[4] + a * a)));
            REQUIRE(jet[14] == approximately(-fp_t{1} / 2 * a * jet[11] / (jet[5] * jet[5] + a * a)));

            REQUIRE(jet[15] == approximately(-fp_t{1} / 2 * c * jet[6] / (jet[0] * jet[0] + c * c)));
            REQUIRE(jet[16] == approximately(-fp_t{1} / 2 * c * jet[7] / (jet[1] * jet[1] + c * c)));
            REQUIRE(jet[17] == approximately(-fp_t{1} / 2 * c * jet[8] / (jet[2] * jet[2] + c * c)));

            REQUIRE(jet[18]
                    == approximately(-fp_t{1} / 6
                                     * (a * 2 * jet[15] * (jet[3] * jet[3] + a * a) - a * jet[9] * 2 * jet[3] * jet[9])
                                     / ((jet[3] * jet[3] + a * a) * (jet[3] * jet[3] + a * a))));
            REQUIRE(
                jet[19]
                == approximately(-fp_t{1} / 6
                                 * (a * 2 * jet[16] * (jet[4] * jet[4] + a * a) - a * jet[10] * 2 * jet[4] * jet[10])
                                 / ((jet[4] * jet[4] + a * a) * (jet[4] * jet[4] + a * a))));
            REQUIRE(
                jet[20]
                == approximately(-fp_t{1} / 6
                                 * (a * 2 * jet[17] * (jet[5] * jet[5] + a * a) - a * jet[11] * 2 * jet[5] * jet[11])
                                 / ((jet[5] * jet[5] + a * a) * (jet[5] * jet[5] + a * a))));

            REQUIRE(jet[21]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[12] * (jet[0] * jet[0] + c * c) - c * jet[6] * 2 * jet[0] * jet[6])
                                     / ((jet[0] * jet[0] + c * c) * (jet[0] * jet[0] + c * c))));
            REQUIRE(jet[22]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[13] * (jet[1] * jet[1] + c * c) - c * jet[7] * 2 * jet[1] * jet[7])
                                     / ((jet[1] * jet[1] + c * c) * (jet[1] * jet[1] + c * c))));
            REQUIRE(jet[23]
                    == approximately(-fp_t{1} / 6
                                     * (c * 2 * jet[14] * (jet[2] * jet[2] + c * c) - c * jet[8] * 2 * jet[2] * jet[8])
                                     / ((jet[2] * jet[2] + c * c) * (jet[2] * jet[2] + c * c))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({atan2(expression{number{a}}, y), atan2(expression{number{c}}, x)}, opt_level,
                                   high_accuracy, compact_mode);

        // Variable-variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(x, y), atan2(y, x)}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(atan2(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(atan2(jet[1], jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(x, y), atan2(y, x)}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(atan2(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(atan2(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(atan2(jet[3], jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(x, y), atan2(y, x)}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(atan2(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(atan2(jet[1], jet[0])));
            REQUIRE(jet[4]
                    == approximately(fp_t{1} / 2 * (jet[1] * jet[2] - jet[0] * jet[3])
                                     / (jet[0] * jet[0] + jet[1] * jet[1])));
            REQUIRE(jet[5]
                    == approximately(-fp_t{1} / 2 * (jet[1] * jet[2] - jet[0] * jet[3])
                                     / (jet[0] * jet[0] + jet[1] * jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(x, y), atan2(y, x)}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(atan2(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(atan2(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(atan2(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(atan2(jet[3], jet[1])));

            REQUIRE(jet[8]
                    == approximately(fp_t{1} / 2 * (jet[2] * jet[4] - jet[0] * jet[6])
                                     / (jet[0] * jet[0] + jet[2] * jet[2])));
            REQUIRE(jet[9]
                    == approximately(fp_t{1} / 2 * (jet[3] * jet[5] - jet[1] * jet[7])
                                     / (jet[1] * jet[1] + jet[3] * jet[3])));

            REQUIRE(jet[10]
                    == approximately(-fp_t{1} / 2 * (jet[2] * jet[4] - jet[0] * jet[6])
                                     / (jet[0] * jet[0] + jet[2] * jet[2])));
            REQUIRE(jet[11]
                    == approximately(-fp_t{1} / 2 * (jet[3] * jet[5] - jet[1] * jet[7])
                                     / (jet[1] * jet[1] + jet[3] * jet[3])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {atan2(x, y), atan2(y, x)}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.1}, fp_t{.3}, fp_t{.4}, fp_t{.6}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);
            REQUIRE(jet[2] == .1);

            REQUIRE(jet[3] == .3);
            REQUIRE(jet[4] == .4);
            REQUIRE(jet[5] == .6);

            REQUIRE(jet[6] == approximately(atan2(jet[0], jet[3])));
            REQUIRE(jet[7] == approximately(atan2(jet[1], jet[4])));
            REQUIRE(jet[8] == approximately(atan2(jet[2], jet[5])));

            REQUIRE(jet[9] == approximately(atan2(jet[3], jet[0])));
            REQUIRE(jet[10] == approximately(atan2(jet[4], jet[1])));
            REQUIRE(jet[11] == approximately(atan2(jet[5], jet[2])));

            REQUIRE(jet[12]
                    == approximately(fp_t{1} / 2 * (jet[3] * jet[6] - jet[0] * jet[9])
                                     / (jet[0] * jet[0] + jet[3] * jet[3])));
            REQUIRE(jet[13]
                    == approximately(fp_t{1} / 2 * (jet[4] * jet[7] - jet[1] * jet[10])
                                     / (jet[1] * jet[1] + jet[4] * jet[4])));
            REQUIRE(jet[14]
                    == approximately(fp_t{1} / 2 * (jet[5] * jet[8] - jet[2] * jet[11])
                                     / (jet[2] * jet[2] + jet[5] * jet[5])));

            REQUIRE(jet[15]
                    == approximately(-fp_t{1} / 2 * (jet[3] * jet[6] - jet[0] * jet[9])
                                     / (jet[0] * jet[0] + jet[3] * jet[3])));
            REQUIRE(jet[16]
                    == approximately(-fp_t{1} / 2 * (jet[4] * jet[7] - jet[1] * jet[10])
                                     / (jet[1] * jet[1] + jet[4] * jet[4])));
            REQUIRE(jet[17]
                    == approximately(-fp_t{1} / 2 * (jet[5] * jet[8] - jet[2] * jet[11])
                                     / (jet[2] * jet[2] + jet[5] * jet[5])));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * (2 * (jet[12] * jet[3] - jet[0] * jet[15]) * (jet[0] * jet[0] + jet[3] * jet[3])
                                        - 2 * (jet[6] * jet[3] - jet[0] * jet[9]) * (jet[0] * jet[6] + jet[3] * jet[9]))
                                     / ((jet[0] * jet[0] + jet[3] * jet[3]) * (jet[0] * jet[0] + jet[3] * jet[3]))));
            REQUIRE(
                jet[19]
                == approximately(fp_t{1} / 6
                                 * (2 * (jet[13] * jet[4] - jet[1] * jet[16]) * (jet[1] * jet[1] + jet[4] * jet[4])
                                    - 2 * (jet[7] * jet[4] - jet[1] * jet[10]) * (jet[1] * jet[7] + jet[4] * jet[10]))
                                 / ((jet[1] * jet[1] + jet[4] * jet[4]) * (jet[1] * jet[1] + jet[4] * jet[4]))));
            REQUIRE(
                jet[20]
                == approximately(fp_t{1} / 6
                                 * (2 * (jet[14] * jet[5] - jet[2] * jet[17]) * (jet[2] * jet[2] + jet[5] * jet[5])
                                    - 2 * (jet[8] * jet[5] - jet[2] * jet[11]) * (jet[2] * jet[8] + jet[5] * jet[11]))
                                 / ((jet[2] * jet[2] + jet[5] * jet[5]) * (jet[2] * jet[2] + jet[5] * jet[5]))));

            REQUIRE(jet[21]
                    == approximately(-fp_t{1} / 6
                                     * (2 * (jet[12] * jet[3] - jet[0] * jet[15]) * (jet[0] * jet[0] + jet[3] * jet[3])
                                        - 2 * (jet[6] * jet[3] - jet[0] * jet[9]) * (jet[0] * jet[6] + jet[3] * jet[9]))
                                     / ((jet[0] * jet[0] + jet[3] * jet[3]) * (jet[0] * jet[0] + jet[3] * jet[3]))));
            REQUIRE(
                jet[22]
                == approximately(-fp_t{1} / 6
                                 * (2 * (jet[13] * jet[4] - jet[1] * jet[16]) * (jet[1] * jet[1] + jet[4] * jet[4])
                                    - 2 * (jet[7] * jet[4] - jet[1] * jet[10]) * (jet[1] * jet[7] + jet[4] * jet[10]))
                                 / ((jet[1] * jet[1] + jet[4] * jet[4]) * (jet[1] * jet[1] + jet[4] * jet[4]))));
            REQUIRE(
                jet[23]
                == approximately(-fp_t{1} / 6
                                 * (2 * (jet[14] * jet[5] - jet[2] * jet[17]) * (jet[2] * jet[2] + jet[5] * jet[5])
                                    - 2 * (jet[8] * jet[5] - jet[2] * jet[11]) * (jet[2] * jet[8] + jet[5] * jet[11]))
                                 / ((jet[2] * jet[2] + jet[5] * jet[5]) * (jet[2] * jet[2] + jet[5] * jet[5]))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({atan2(x, y), atan2(y, x)}, opt_level, high_accuracy, compact_mode);
    };

    for (auto cm : {true, false}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}
