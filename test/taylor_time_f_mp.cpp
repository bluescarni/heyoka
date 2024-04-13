// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>

#include <mp++/real.hpp>

#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
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
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = heyoka::time, prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::time = fp_t{3, prec}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == 3);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == fp_t(.5, prec));
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                }
            }
        }
    }
}
