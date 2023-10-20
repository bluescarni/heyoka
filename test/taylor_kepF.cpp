// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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

// Issue in the decomposition when h/k = 0.
TEST_CASE("taylor kepF decompose bug 00")
{
    llvm_state s;

    auto [lam] = make_vars("lam");

    taylor_add_jet<double>(s, "jet", {kepF(0_dbl, 0_dbl, lam)}, 1, 1, false, false);
}

TEST_CASE("taylor kepF")
{
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y, z] = make_vars("x", "y", "z");

        // cfunc for testing purposes.
        llvm_state s_cfunc;

        add_cfunc<fp_t>(s_cfunc, "cfunc", {kepF(x, y, z)});

        s_cfunc.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s_cfunc.jit_lookup("cfunc"));

        auto kepF_num = [cf_ptr](fp_t h, fp_t k, fp_t lam) {
            const fp_t cf_in[3] = {h, k, lam};
            fp_t cf_out(0);

            cf_ptr(&cf_out, cf_in, nullptr, nullptr);

            return cf_out;
        };

        // Number-number-number test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1), par[0], .3_dbl), x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}}, pars{fp_t(.1), fp_t(.2)};
            jet.resize(12);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(fp_t(.1), pars[0], fp_t(.3))));
            REQUIRE(jet[5] == approximately(kepF_num(fp_t(.1), pars[1], fp_t(.3))));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);
        }

        // Number-number-var test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1), par[0], x), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}}, pars{fp_t(.1), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(fp_t(.1), pars[0], jet[0])));
            REQUIRE(jet[5] == approximately(kepF_num(fp_t(.1), pars[1], jet[1])));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            auto den0 = 1 - fp_t(.1) * sin(jet[4]) - pars[0] * cos(jet[4]);
            auto den1 = 1 - fp_t(.1) * sin(jet[5]) - pars[1] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[4] / den0) / 2));
            REQUIRE(jet[9] == approximately((jet[5] / den1) / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto tmp0 = -fp_t(.1) * cos(jet[4]) * jet[8] * 2 + pars[0] * sin(jet[4]) * jet[8] * 2;
            auto tmp1 = -fp_t(.1) * cos(jet[5]) * jet[9] * 2 + pars[1] * sin(jet[5]) * jet[9] * 2;

            REQUIRE(jet[12] == approximately(((jet[8] * 2 * den0 - jet[4] * tmp0) / (den0 * den0)) / 6));
            REQUIRE(jet[13] == approximately(((jet[9] * 2 * den1 - jet[5] * tmp1) / (den1 * den1)) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }
    };

    // TODO enable compact mode.
    for (auto cm : {false, false}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}