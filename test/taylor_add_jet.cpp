// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
// #include <stdexcept>
// #include <string>
// #include <tuple>
// #include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
// #include <heyoka/math/sqrt.hpp>
// #include <heyoka/math/square.hpp>
// #include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("add jet sv_funcs")
{
    auto [x, y] = make_vars("x", "y");

    // TODO: test with simplifications, sv variables and repetitions.
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x + y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 3);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == 6);

                    REQUIRE(jet[3] == 12);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 6);

                    REQUIRE(jet[6] == -3);
                    REQUIRE(jet[7] == 6);
                    REQUIRE(jet[8] == 3);

                    REQUIRE(jet[9] == 2);
                    REQUIRE(jet[10] == -1);
                    REQUIRE(jet[11] == 1);
                }
            }
        }
    }
}
