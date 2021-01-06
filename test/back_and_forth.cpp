// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/nbody.hpp>
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

TEST_CASE("pendulum")
{
    auto tester = [](auto fp_x, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        taylor_adaptive<fp_t> ta{{prime("th"_var) = "v"_var, prime("v"_var) = -9.8_dbl / 1.5_dbl * sin("th"_var)},
                                 {fp_t{0.05}, fp_t{0.025}},
                                 kw::high_accuracy = high_accuracy,
                                 kw::compact_mode = compact_mode};

        ta.propagate_for(fp_t{100});
        ta.propagate_for(fp_t{-100});

        const auto &st = ta.get_state();

        REQUIRE(st[0] == approximately(fp_t{.05}, fp_t{1E4}));
        REQUIRE(st[1] == approximately(fp_t{0.025}, fp_t{1E4}));
    };

    for (auto cm : {true, false}) {
        for (auto ha : {true, false}) {
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, ha, cm); });
        }
    }
}

TEST_CASE("three body")
{
    auto tester = [](auto fp_x, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto masses = std::vector{fp_t{1.989e30}, fp_t{1.898e27}, fp_t{5.683e26}};

        const auto sun_mu = fp_t{1.327e20};
        const auto G = fp_t{6.674e-11};

        auto sys = make_nbody_sys(3, kw::masses = masses, kw::Gconst = G);

        // Initial states in orbital elements for Jupiter and Saturn.
        // NOTE: a, e, i are realistic, the angles are random.
        const auto j_kep = std::array{fp_t{778.57e9}, fp_t{0.0489}, fp_t{0.02274164}, fp_t{.1}, fp_t{.2}, fp_t{.3}};
        const auto [j_x, j_v] = kep_to_cart(j_kep, sun_mu);

        const auto s_kep = std::array{fp_t{1433.53e9}, fp_t{0.0565}, fp_t{0.043371432}, fp_t{.4}, fp_t{.5}, fp_t{.6}};
        const auto [s_x, s_v] = kep_to_cart(s_kep, sun_mu);

        taylor_adaptive<fp_t> ta{std::move(sys),
                                 {// Sun in the origin, zero speed.
                                  fp_t{0}, fp_t{0}, fp_t{0}, fp_t{0}, fp_t{0}, fp_t{0},
                                  // Jupiter.
                                  fp_t{j_x[0]}, fp_t{j_x[1]}, fp_t{j_x[2]}, fp_t{j_v[0]}, fp_t{j_v[1]}, fp_t{j_v[2]},
                                  // Saturn.
                                  fp_t{s_x[0]}, fp_t{s_x[1]}, fp_t{s_x[2]}, fp_t{s_v[0]}, fp_t{s_v[1]}, fp_t{s_v[2]}},
                                 kw::high_accuracy = high_accuracy,
                                 kw::compact_mode = compact_mode};

        ta.propagate_for(fp_t{5} * 365 * 86400);
        ta.propagate_for(-fp_t{5} * 365 * 86400);

        const auto &st = ta.get_state();

        REQUIRE(st[6] == approximately(j_x[0], fp_t{1E4}));
        REQUIRE(st[7] == approximately(j_x[1], fp_t{1E4}));
        REQUIRE(st[8] == approximately(j_x[2], fp_t{1E4}));
        REQUIRE(st[9] == approximately(j_v[0], fp_t{1E4}));
        REQUIRE(st[10] == approximately(j_v[1], fp_t{1E4}));
        REQUIRE(st[11] == approximately(j_v[2], fp_t{1E4}));

        REQUIRE(st[12] == approximately(s_x[0], fp_t{1E4}));
        REQUIRE(st[13] == approximately(s_x[1], fp_t{1E4}));
        REQUIRE(st[14] == approximately(s_x[2], fp_t{1E4}));
        REQUIRE(st[15] == approximately(s_v[0], fp_t{1E4}));
        REQUIRE(st[16] == approximately(s_v[1], fp_t{1E4}));
        REQUIRE(st[17] == approximately(s_v[2], fp_t{1E4}));
    };

    for (auto cm : {true, false}) {
        for (auto ha : {true, false}) {
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, ha, cm); });
        }
    }
}
