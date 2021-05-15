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
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

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

// Boost-based Kepler solver.
auto bmt_inv_kep_E = [](auto ecc, auto M) {
    using std::sin;
    using std::cos;

    using fp_t = decltype(ecc);

    // Initial guess.
    auto ig = ecc < 0.8 ? M : static_cast<fp_t>(boost::math::constants::pi<double>());

    auto func = [ecc, M](auto E) { return std::make_pair(E - ecc * sin(E) - M, 1 - ecc * cos(E)); };

    boost::uintmax_t max_iter = 50;

    return boost::math::tools::newton_raphson_iterate(func, ig, fp_t(0), fp_t(2 * boost::math::constants::pi<double>()),
                                                      std::numeric_limits<fp_t>::digits - 2, max_iter);
};

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

TEST_CASE("taylor kepE")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::sin;
        using std::cos;

        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        const auto a = fp_t{1} / 3;
        const auto b = 1 + fp_t{3} / fp_t{7};
        const auto c = fp_t{2} / 7;

        // Number-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, expression{number{b}}), x + y}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(par[0], expression{number{b}}), x + y}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, expression{number{b}}), x + y}, 1, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, par[1]), x + y}, 1, 2, high_accuracy,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, expression{number{b}}), x + y}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == fp_t{1} / 2 * (jet[2] + jet[3]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, expression{number{b}}), x + y}, 2, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, b)));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == fp_t{1} / 2 * (jet[4] + jet[6]));
            REQUIRE(jet[11] == fp_t{1} / 2 * (jet[5] + jet[7]));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(par[0], par[1]), x + y}, 3, 3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(a, b)));
            REQUIRE(jet[8] == approximately(bmt_inv_kep_E(a, b)));

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
        compare_batch_scalar<fp_t>({kepE(expression{number{a}}, expression{number{b}}), x + y}, opt_level,
                                   high_accuracy, compact_mode);

        // Variable-number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{a}}), kepE(x, expression{number{b}})}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(jet[1], a)));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(jet[0], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, par[0]), kepE(x, expression{number{b}})}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(jet[1], a)));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(jet[0], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{b}}), kepE(x, expression{number{b}})}, 1, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(jet[2], b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(jet[3], b)));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[0], b)));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[1], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{b}}), kepE(x, par[1])}, 1, 2, high_accuracy,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(jet[2], b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(jet[3], b)));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[0], b)));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[1], b)));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{b}}), kepE(x, expression{number{b}})}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(jet[1], b)));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(jet[0], b)));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * jet[3] * sin(jet[2]) / (1 - jet[1] * cos(jet[2]))));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * jet[2] * sin(jet[3]) / (1 - jet[0] * cos(jet[3]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{b}}), kepE(x, expression{number{b}})}, 2, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(jet[2], b)));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(jet[3], b)));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[0], b)));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[1], b)));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[6] * sin(jet[4]) / (1 - jet[2] * cos(jet[4]))));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[7] * sin(jet[5]) / (1 - jet[3] * cos(jet[5]))));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * jet[4] * sin(jet[6]) / (1 - jet[0] * cos(jet[6]))));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * jet[5] * sin(jet[7]) / (1 - jet[1] * cos(jet[7]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(y, expression{number{b}}), kepE(x, expression{number{b}})}, 3, 3,
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

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[3], b)));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[4], b)));
            REQUIRE(jet[8] == approximately(bmt_inv_kep_E(jet[5], b)));

            REQUIRE(jet[9] == approximately(bmt_inv_kep_E(jet[0], b)));
            REQUIRE(jet[10] == approximately(bmt_inv_kep_E(jet[1], b)));
            REQUIRE(jet[11] == approximately(bmt_inv_kep_E(jet[2], b)));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[9] * sin(jet[6]) / (1 - jet[3] * cos(jet[6]))));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[10] * sin(jet[7]) / (1 - jet[4] * cos(jet[7]))));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[11] * sin(jet[8]) / (1 - jet[5] * cos(jet[8]))));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * jet[6] * sin(jet[9]) / (1 - jet[0] * cos(jet[9]))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * jet[7] * sin(jet[10]) / (1 - jet[1] * cos(jet[10]))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * jet[8] * sin(jet[11]) / (1 - jet[2] * cos(jet[11]))));

            REQUIRE(jet[18]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[15] * sin(jet[6]) / (1 - jet[3] * cos(jet[6]))
                           + jet[9]
                                 * (2 * jet[12] * cos(jet[6]) * (1 - jet[3] * cos(jet[6]))
                                    + sin(jet[6]) * (jet[9] * cos(jet[6]) - jet[3] * 2 * jet[12] * sin(jet[6])))
                                 / ((1 - jet[3] * cos(jet[6])) * (1 - jet[3] * cos(jet[6]))))));
            REQUIRE(jet[19]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[16] * sin(jet[7]) / (1 - jet[4] * cos(jet[7]))
                           + jet[10]
                                 * (2 * jet[13] * cos(jet[7]) * (1 - jet[4] * cos(jet[7]))
                                    + sin(jet[7]) * (jet[10] * cos(jet[7]) - jet[4] * 2 * jet[13] * sin(jet[7])))
                                 / ((1 - jet[4] * cos(jet[7])) * (1 - jet[4] * cos(jet[7]))))));
            REQUIRE(jet[20]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[17] * sin(jet[8]) / (1 - jet[5] * cos(jet[8]))
                           + jet[11]
                                 * (2 * jet[14] * cos(jet[8]) * (1 - jet[5] * cos(jet[8]))
                                    + sin(jet[8]) * (jet[11] * cos(jet[8]) - jet[5] * 2 * jet[14] * sin(jet[8])))
                                 / ((1 - jet[5] * cos(jet[8])) * (1 - jet[5] * cos(jet[8]))))));

            REQUIRE(jet[21]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[12] * sin(jet[9]) / (1 - jet[0] * cos(jet[9]))
                           + jet[6]
                                 * (2 * jet[15] * cos(jet[9]) * (1 - jet[0] * cos(jet[9]))
                                    + sin(jet[9]) * (jet[6] * cos(jet[9]) - jet[0] * 2 * jet[15] * sin(jet[9])))
                                 / ((1 - jet[0] * cos(jet[9])) * (1 - jet[0] * cos(jet[9]))))));
            REQUIRE(jet[22]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[13] * sin(jet[10]) / (1 - jet[1] * cos(jet[10]))
                           + jet[7]
                                 * (2 * jet[16] * cos(jet[10]) * (1 - jet[1] * cos(jet[10]))
                                    + sin(jet[10]) * (jet[7] * cos(jet[10]) - jet[1] * 2 * jet[16] * sin(jet[10])))
                                 / ((1 - jet[1] * cos(jet[10])) * (1 - jet[1] * cos(jet[10]))))));
            REQUIRE(jet[23]
                    == approximately(
                        fp_t{1} / 6
                        * (2 * jet[14] * sin(jet[11]) / (1 - jet[2] * cos(jet[11]))
                           + jet[8]
                                 * (2 * jet[17] * cos(jet[11]) * (1 - jet[2] * cos(jet[11]))
                                    + sin(jet[11]) * (jet[8] * cos(jet[11]) - jet[2] * 2 * jet[17] * sin(jet[11])))
                                 / ((1 - jet[2] * cos(jet[11])) * (1 - jet[2] * cos(jet[11]))))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({kepE(y, expression{number{b}}), kepE(x, expression{number{b}})}, opt_level,
                                   high_accuracy, compact_mode);

        // Number-variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, 1, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, jet[1])));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(c, jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(par[0], y), kepE(expression{number{c}}, x)}, 1, 1, high_accuracy,
                                 compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            std::vector<fp_t> pars{a};

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, jet[1])));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(c, jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, 1, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, jet[2])));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, jet[3])));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(c, jet[0])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(c, jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(par[1], x)}, 1, 2, high_accuracy,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, jet[2])));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, jet[3])));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(c, jet[0])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(c, jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, 2, 1,
                                 high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(a, jet[1])));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(c, jet[0])));
            REQUIRE(jet[4] == approximately(fp_t{1} / 2 * jet[3] / (1 - a * cos(jet[2]))));
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * jet[2] / (1 - c * cos(jet[3]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, 2, 2,
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

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(a, jet[2])));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(a, jet[3])));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(c, jet[0])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(c, jet[1])));

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[6] / (1 - a * cos(jet[4]))));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[7] / (1 - a * cos(jet[5]))));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * jet[4] / (1 - c * cos(jet[6]))));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * jet[5] / (1 - c * cos(jet[7]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, 3, 3,
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

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(a, jet[3])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(a, jet[4])));
            REQUIRE(jet[8] == approximately(bmt_inv_kep_E(a, jet[5])));

            REQUIRE(jet[9] == approximately(bmt_inv_kep_E(c, jet[0])));
            REQUIRE(jet[10] == approximately(bmt_inv_kep_E(c, jet[1])));
            REQUIRE(jet[11] == approximately(bmt_inv_kep_E(c, jet[2])));

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[9] / (1 - a * cos(jet[6]))));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[10] / (1 - a * cos(jet[7]))));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[11] / (1 - a * cos(jet[8]))));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * jet[6] / (1 - c * cos(jet[9]))));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * jet[7] / (1 - c * cos(jet[10]))));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * jet[8] / (1 - c * cos(jet[11]))));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[15] * (1 - a * cos(jet[6])) - a * jet[9] * 2 * jet[12] * sin(jet[6]))
                                     / ((1 - a * cos(jet[6])) * (1 - a * cos(jet[6])))));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[16] * (1 - a * cos(jet[7])) - a * jet[10] * 2 * jet[13] * sin(jet[7]))
                                     / ((1 - a * cos(jet[7])) * (1 - a * cos(jet[7])))));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[17] * (1 - a * cos(jet[8])) - a * jet[11] * 2 * jet[14] * sin(jet[8]))
                                     / ((1 - a * cos(jet[8])) * (1 - a * cos(jet[8])))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[12] * (1 - c * cos(jet[9])) - c * jet[6] * 2 * jet[15] * sin(jet[9]))
                                     / ((1 - c * cos(jet[9])) * (1 - c * cos(jet[9])))));
            REQUIRE(jet[22]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[13] * (1 - c * cos(jet[10])) - c * jet[7] * 2 * jet[16] * sin(jet[10]))
                                     / ((1 - c * cos(jet[10])) * (1 - c * cos(jet[10])))));
            REQUIRE(jet[23]
                    == approximately(fp_t{1} / 6
                                     * (2 * jet[14] * (1 - c * cos(jet[11])) - c * jet[8] * 2 * jet[17] * sin(jet[11]))
                                     / ((1 - c * cos(jet[11])) * (1 - c * cos(jet[11])))));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({kepE(expression{number{a}}, y), kepE(expression{number{c}}, x)}, opt_level,
                                   high_accuracy, compact_mode);

        // Variable-variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(x, y), kepE(y, x)}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(4);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == fp_t{.2});
            REQUIRE(jet[1] == fp_t{.3});
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(jet[1], jet[0])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(x, y), kepE(y, x)}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(8);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[3], jet[1])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(x, y), kepE(y, x)}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.3}};
            jet.resize(6);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .3);
            REQUIRE(jet[2] == approximately(bmt_inv_kep_E(jet[0], jet[1])));
            REQUIRE(jet[3] == approximately(bmt_inv_kep_E(jet[1], jet[0])));
            REQUIRE(jet[4]
                    == approximately(fp_t{1} / 2 * (jet[2] * sin(jet[2]) + jet[3]) / (1 - jet[0] * cos(jet[2]))));
            REQUIRE(jet[5]
                    == approximately(fp_t{1} / 2 * (jet[3] * sin(jet[3]) + jet[2]) / (1 - jet[1] * cos(jet[3]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(x, y), kepE(y, x)}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.2}, fp_t{.5}, fp_t{.3}, fp_t{.4}};
            jet.resize(12);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == .2);
            REQUIRE(jet[1] == .5);

            REQUIRE(jet[2] == .3);
            REQUIRE(jet[3] == .4);

            REQUIRE(jet[4] == approximately(bmt_inv_kep_E(jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(bmt_inv_kep_E(jet[1], jet[3])));

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[2], jet[0])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[3], jet[1])));

            REQUIRE(jet[8]
                    == approximately(fp_t{1} / 2 * (jet[4] * sin(jet[4]) + jet[6]) / (1 - jet[0] * cos(jet[4]))));
            REQUIRE(jet[9]
                    == approximately(fp_t{1} / 2 * (jet[5] * sin(jet[5]) + jet[7]) / (1 - jet[1] * cos(jet[5]))));

            REQUIRE(jet[10]
                    == approximately(fp_t{1} / 2 * (jet[6] * sin(jet[6]) + jet[4]) / (1 - jet[2] * cos(jet[6]))));
            REQUIRE(jet[11]
                    == approximately(fp_t{1} / 2 * (jet[7] * sin(jet[7]) + jet[5]) / (1 - jet[3] * cos(jet[7]))));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepE(x, y), kepE(y, x)}, 3, 3, high_accuracy, compact_mode);

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

            REQUIRE(jet[6] == approximately(bmt_inv_kep_E(jet[0], jet[3])));
            REQUIRE(jet[7] == approximately(bmt_inv_kep_E(jet[1], jet[4])));
            REQUIRE(jet[8] == approximately(bmt_inv_kep_E(jet[2], jet[5])));

            REQUIRE(jet[9] == approximately(bmt_inv_kep_E(jet[3], jet[0])));
            REQUIRE(jet[10] == approximately(bmt_inv_kep_E(jet[4], jet[1])));
            REQUIRE(jet[11] == approximately(bmt_inv_kep_E(jet[5], jet[2])));

            REQUIRE(jet[12]
                    == approximately(fp_t{1} / 2 * (jet[6] * sin(jet[6]) + jet[9]) / (1 - jet[0] * cos(jet[6]))));
            REQUIRE(jet[13]
                    == approximately(fp_t{1} / 2 * (jet[7] * sin(jet[7]) + jet[10]) / (1 - jet[1] * cos(jet[7]))));
            REQUIRE(jet[14]
                    == approximately(fp_t{1} / 2 * (jet[8] * sin(jet[8]) + jet[11]) / (1 - jet[2] * cos(jet[8]))));

            REQUIRE(jet[15]
                    == approximately(fp_t{1} / 2 * (jet[9] * sin(jet[9]) + jet[6]) / (1 - jet[3] * cos(jet[9]))));
            REQUIRE(jet[16]
                    == approximately(fp_t{1} / 2 * (jet[10] * sin(jet[10]) + jet[7]) / (1 - jet[4] * cos(jet[10]))));
            REQUIRE(jet[17]
                    == approximately(fp_t{1} / 2 * (jet[11] * sin(jet[11]) + jet[8]) / (1 - jet[5] * cos(jet[11]))));

            REQUIRE(jet[18]
                    == approximately(fp_t{1} / 6
                                     * ((2 * jet[12] * sin(jet[6]) + jet[6] * 2 * jet[12] * cos(jet[6]) + 2 * jet[15])
                                            * (1 - jet[0] * cos(jet[6]))
                                        - (jet[6] * sin(jet[6]) + jet[9])
                                              * (jet[0] * 2 * jet[12] * sin(jet[6]) - jet[6] * cos(jet[6])))
                                     / ((1 - jet[0] * cos(jet[6])) * (1 - jet[0] * cos(jet[6])))));
            REQUIRE(jet[19]
                    == approximately(fp_t{1} / 6
                                     * ((2 * jet[13] * sin(jet[7]) + jet[7] * 2 * jet[13] * cos(jet[7]) + 2 * jet[16])
                                            * (1 - jet[1] * cos(jet[7]))
                                        - (jet[7] * sin(jet[7]) + jet[10])
                                              * (jet[1] * 2 * jet[13] * sin(jet[7]) - jet[7] * cos(jet[7])))
                                     / ((1 - jet[1] * cos(jet[7])) * (1 - jet[1] * cos(jet[7])))));
            REQUIRE(jet[20]
                    == approximately(fp_t{1} / 6
                                     * ((2 * jet[14] * sin(jet[8]) + jet[8] * 2 * jet[14] * cos(jet[8]) + 2 * jet[17])
                                            * (1 - jet[2] * cos(jet[8]))
                                        - (jet[8] * sin(jet[8]) + jet[11])
                                              * (jet[2] * 2 * jet[14] * sin(jet[8]) - jet[8] * cos(jet[8])))
                                     / ((1 - jet[2] * cos(jet[8])) * (1 - jet[2] * cos(jet[8])))));

            REQUIRE(jet[21]
                    == approximately(fp_t{1} / 6
                                     * ((2 * jet[15] * sin(jet[9]) + jet[9] * 2 * jet[15] * cos(jet[9]) + 2 * jet[12])
                                            * (1 - jet[3] * cos(jet[9]))
                                        - (jet[9] * sin(jet[9]) + jet[6])
                                              * (jet[3] * 2 * jet[15] * sin(jet[9]) - jet[9] * cos(jet[9])))
                                     / ((1 - jet[3] * cos(jet[9])) * (1 - jet[3] * cos(jet[9])))));
            REQUIRE(
                jet[22]
                == approximately(fp_t{1} / 6
                                 * ((2 * jet[16] * sin(jet[10]) + jet[10] * 2 * jet[16] * cos(jet[10]) + 2 * jet[13])
                                        * (1 - jet[4] * cos(jet[10]))
                                    - (jet[10] * sin(jet[10]) + jet[7])
                                          * (jet[4] * 2 * jet[16] * sin(jet[10]) - jet[10] * cos(jet[10])))
                                 / ((1 - jet[4] * cos(jet[10])) * (1 - jet[4] * cos(jet[10])))));
            REQUIRE(
                jet[23]
                == approximately(fp_t{1} / 6
                                 * ((2 * jet[17] * sin(jet[11]) + jet[11] * 2 * jet[17] * cos(jet[11]) + 2 * jet[14])
                                        * (1 - jet[5] * cos(jet[11]))
                                    - (jet[11] * sin(jet[11]) + jet[8])
                                          * (jet[5] * 2 * jet[17] * sin(jet[11]) - jet[11] * cos(jet[11])))
                                 / ((1 - jet[5] * cos(jet[11])) * (1 - jet[5] * cos(jet[11])))));
        }
    };

    // TODO fix.
    for (auto cm : {false, false}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}
