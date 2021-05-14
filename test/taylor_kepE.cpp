// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
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

TEST_CASE("taylor kepE")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using Catch::Matchers::Message;

        auto x = "x"_var, y = "y"_var;

        const auto a = fp_t{1} / 3;
        const auto b = 1 + fp_t{3} / fp_t{7};

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
