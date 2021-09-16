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
#include <sstream>
#include <tuple>
#include <vector>

#include <boost/math/constants/constants.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;
namespace hy = heyoka;

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

// Variable template for the pi constant at different levels of precision.
template <typename T>
const auto pi_const = boost::math::constants::pi<T>();

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 pi_const<mppp::real128> = mppp::pi_128;

#endif

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

        std::vector<T> jet_batch, time_batch;
        jet_batch.resize(8 * batch_size);
        time_batch.resize(batch_size);
        std::uniform_real_distribution<float> dist(.1f, 20.f);
        std::generate(jet_batch.begin(), jet_batch.end(), [&dist]() { return T{dist(rng)}; });
        std::generate(time_batch.begin(), time_batch.end(), [&dist]() { return T{dist(rng)}; });

        std::vector<T> jet_scalar;
        T time_scalar(0);
        jet_scalar.resize(8);

        jptr_batch(jet_batch.data(), nullptr, time_batch.data());

        for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
            // Assign the initial values of x and y.
            for (auto i = 0u; i < 2u; ++i) {
                jet_scalar[i] = jet_batch[i * batch_size + batch_idx];
            }

            // Assign the time.
            time_scalar = time_batch[batch_idx];

            jptr_scalar(jet_scalar.data(), nullptr, &time_scalar);

            for (auto i = 2u; i < 8u; ++i) {
                REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx]));
            }
        }
    }
}

TEST_CASE("taylor constant")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {hy::pi, x + y}, 1, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(4);

            fp_t t(4);

            jptr(jet.data(), nullptr, &t);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == pi_const<fp_t>);
            REQUIRE(jet[3] == 5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {hy::pi, x + y}, 1, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(8);

            std::vector<fp_t> t{fp_t{-5}, fp_t{6}};

            jptr(jet.data(), nullptr, t.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == pi_const<fp_t>);
            REQUIRE(jet[5] == pi_const<fp_t>);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {hy::pi, x + y}, 2, 1, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
            jet.resize(6);

            fp_t t(-4);

            jptr(jet.data(), nullptr, &t);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == pi_const<fp_t>);
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + jet[2])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {hy::pi + x, x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}};
            jet.resize(12);

            std::vector<fp_t> t{fp_t{-5}, fp_t{6}};

            jptr(jet.data(), nullptr, t.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == pi_const<fp_t> + jet[0]);
            REQUIRE(jet[5] == pi_const<fp_t> + jet[1]);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[4]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[5]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + jet[4])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + jet[5])));
        }

        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {x + hy::pi, x + y}, 3, 3, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}};
            jet.resize(24);

            std::vector<fp_t> t{fp_t{-5}, fp_t{6}, fp_t{-1}};

            jptr(jet.data(), nullptr, t.data());

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == approximately(pi_const<fp_t> + jet[0]));
            REQUIRE(jet[7] == approximately(pi_const<fp_t> + jet[1]));
            REQUIRE(jet[8] == approximately(pi_const<fp_t> + jet[2]));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[6]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[7]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[8]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + jet[6])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + jet[7])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + jet[8])));

            REQUIRE(jet[18] == approximately(fp_t{1} / 6 * 2 * jet[12]));
            REQUIRE(jet[19] == approximately(fp_t{1} / 6 * 2 * jet[13]));
            REQUIRE(jet[20] == approximately(fp_t{1} / 6 * 2 * jet[14]));

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15] + 2 * jet[12])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16] + 2 * jet[13])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17] + 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({x + hy::pi, x + y}, opt_level, high_accuracy, compact_mode);
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
