// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/relu.hpp>
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

template <typename T>
T cpp_relu(T x)
{
    return x > 0 ? x : T(0);
}

template <typename T>
T cpp_relup(T x)
{
    return x > 0 ? T(1) : T(0);
}

TEST_CASE("taylor relu relup")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        // Number tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {relu(par[0]) + relup(par[1]), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.relu.par"));
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.relup.par"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}}, pars{fp_t{-1}, fp_t{2}, fp_t{4}, fp_t{-3}};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(pars[0]) + cpp_relup(pars[2])));
            REQUIRE(jet[5] == approximately(cpp_relu(pars[1]) + cpp_relup(pars[3])));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == fp_t(0));
            REQUIRE(jet[9] == fp_t(0));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == fp_t(0));
            REQUIRE(jet[13] == fp_t(0));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
        }

        // Variable tests.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {relu(x) + relup(y), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.relu.var"));
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.relup.var"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}};
            jet.resize(16);

            jptr(jet.data(), nullptr, nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(cpp_relu(jet[0]) + cpp_relup(jet[2])));
            REQUIRE(jet[5] == approximately(cpp_relu(jet[1]) + cpp_relup(jet[3])));

            REQUIRE(jet[6] == approximately(jet[0] + jet[2]));
            REQUIRE(jet[7] == approximately(jet[1] + jet[3]));

            REQUIRE(jet[8] == approximately((cpp_relup(jet[0]) * jet[4]) / 2));
            REQUIRE(jet[9] == approximately((cpp_relup(jet[1]) * jet[5]) / 2));

            REQUIRE(jet[10] == approximately((jet[4] + jet[6]) / 2));
            REQUIRE(jet[11] == approximately((jet[5] + jet[7]) / 2));

            REQUIRE(jet[12] == approximately((cpp_relup(jet[0]) * 2 * jet[8]) / 6));
            REQUIRE(jet[13] == approximately((cpp_relup(jet[1]) * 2 * jet[9]) / 6));

            REQUIRE(jet[14] == approximately((jet[10] + jet[8]) / 3));
            REQUIRE(jet[15] == approximately((jet[11] + jet[9]) / 3));
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
