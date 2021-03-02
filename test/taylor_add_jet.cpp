// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
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

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 4);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -6);
                    REQUIRE(jet[3] == 12);

                    REQUIRE(jet[4] == 12);
                    REQUIRE(jet[5] == -6);
                    REQUIRE(jet[6] == 12);
                    REQUIRE(jet[7] == -6);

                    REQUIRE(jet[8] == -3);
                    REQUIRE(jet[9] == 6);
                    REQUIRE(jet[10] == -3);
                    REQUIRE(jet[11] == 6);

                    REQUIRE(jet[12] == 2);
                    REQUIRE(jet[13] == -1);
                    REQUIRE(jet[14] == 2);
                    REQUIRE(jet[15] == -1);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x, y, x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -6);
                    REQUIRE(jet[3] == 12);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 12);

                    REQUIRE(jet[6] == 12);
                    REQUIRE(jet[7] == -6);
                    REQUIRE(jet[8] == 12);
                    REQUIRE(jet[9] == -6);
                    REQUIRE(jet[10] == 12);
                    REQUIRE(jet[11] == -6);

                    REQUIRE(jet[12] == -3);
                    REQUIRE(jet[13] == 6);
                    REQUIRE(jet[14] == -3);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -3);
                    REQUIRE(jet[17] == 6);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -1);
                    REQUIRE(jet[20] == 2);
                    REQUIRE(jet[21] == -1);
                    REQUIRE(jet[22] == 2);
                    REQUIRE(jet[23] == -1);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 12);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == 6);
                    REQUIRE(jet[11] == -18);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -6);
                    REQUIRE(jet[17] == 12);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == 2);
                    REQUIRE(jet[23] == -6);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x + 2. * y, y + 2. * x});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == 18);
                    REQUIRE(jet[5] == 0);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == -30);
                    REQUIRE(jet[11] == -6);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == 18);
                    REQUIRE(jet[17] == 0);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == -10);
                    REQUIRE(jet[23] == -2);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x - y, x + y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == -18);
                    REQUIRE(jet[5] == 6);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == 24);
                    REQUIRE(jet[11] == -12);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -18);
                    REQUIRE(jet[17] == 6);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == 8);
                    REQUIRE(jet[23] == -4);
                }
            }
        }
    }
}
