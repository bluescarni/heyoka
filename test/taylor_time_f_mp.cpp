// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <mp++/real.hpp>

#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/math/tpoly.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("time")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    const auto uprec = static_cast<unsigned>(prec);

                    {
                        llvm_state s{kw::opt_level = opt_level};

                        taylor_add_jet<fp_t>(s, "jet", {heyoka::time, x + y}, 2, 1, ha, cm, {}, false, uprec);

                        s.compile();

                        auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                        std::vector<fp_t> jet{fp_t{2, prec}, fp_t{3, prec}};
                        auto tm = fp_t{3, prec};
                        jet.resize(6, fp_t{0, prec});

                        jptr(jet.data(), nullptr, &tm);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == tm);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == fp_t(.5, prec));
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                }
            }
        }
    }
}

TEST_CASE("tpoly")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    const auto uprec = static_cast<unsigned>(prec);

                    {
                        llvm_state s{kw::opt_level = opt_level};

                        taylor_add_jet<fp_t>(s, "jet", {tpoly(par[0], par[2]), x + y}, 2, 1, ha, cm, {}, false, uprec);

                        s.compile();

                        auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                        std::vector<fp_t> jet{fp_t{2, prec}, fp_t{3, prec}};
                        auto tm = fp_t{3, prec};
                        std::vector<fp_t> pars{fp_t{.2, prec}, fp_t{.3, prec}, fp_t{.4, prec}};
                        jet.resize(6, fp_t{0, prec});

                        jptr(jet.data(), pars.data(), &tm);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == pars[0] + tm * pars[1]);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == fp_t(.5, prec) * pars[1]);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                }
            }
        }
    }
}
