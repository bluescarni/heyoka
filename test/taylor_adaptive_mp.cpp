// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <tuple>

#include <mp++/real.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basics")
{
    auto [x] = make_vars("x");

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto prec : {30u, 237u}) {
                auto ta = taylor_adaptive<mppp::real>({x}, {mppp::real{1, prec}}, kw::compact_mode = cm,
                                                      kw::opt_level = opt_level);

                for (auto i = 0; i < 10; ++i) {
                    REQUIRE(std::get<0>(ta.step()) == taylor_outcome::success);
                }
                REQUIRE(ta.get_state()[0] == approximately(exp(ta.get_time())));
            }
        }
    }
}
