// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <heyoka/detail/num_identity.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;
namespace hy = heyoka;

TEST_CASE("taylor num_identity")
{
    using fp_t = double;

    auto x = "x"_var, y = "y"_var;

    for (auto cm : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            {
                llvm_state s{kw::opt_level = opt_level};

                taylor_add_jet<fp_t>(s, "jet", {hy::detail::num_identity(42_dbl), x + y}, 1, 1, false, cm);

                s.compile();

                auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
                jet.resize(4);

                fp_t t(4);

                jptr(jet.data(), nullptr, &t);

                REQUIRE(jet[0] == 2);
                REQUIRE(jet[1] == 3);
                REQUIRE(jet[2] == 42);
                REQUIRE(jet[3] == 5);
            }

            {
                llvm_state s{kw::opt_level = opt_level};

                taylor_add_jet<fp_t>(s, "jet", {hy::detail::num_identity(42_dbl), x + y}, 3, 1, false, cm);

                s.compile();

                auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                std::vector<fp_t> jet{fp_t{2}, fp_t{3}};
                jet.resize(8);

                fp_t t(-4);

                jptr(jet.data(), nullptr, &t);

                REQUIRE(jet[0] == 2);
                REQUIRE(jet[1] == 3);
                REQUIRE(jet[2] == 42);
                REQUIRE(jet[3] == 5);
                REQUIRE(jet[4] == 0);
                REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + jet[2])));
                REQUIRE(jet[6] == 0);
                REQUIRE(jet[7] == approximately(fp_t{1} / 6 * (2 * jet[5] + 2 * jet[4])));
            }
        }
    }

    // Def ctor.
    detail::num_identity_impl nu;
    REQUIRE(nu.args() == std::vector{0_dbl});
}
