// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <limits>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("taylor diff")
{
    auto x = "x"_var, y = "y"_var;

    {
        // Variable-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(y, 3_dbl / 2_dbl), pow(x, -3_dbl / 2_dbl)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == Approx(std::pow(2, -3 / 2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == Approx(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == Approx(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == Approx(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == Approx(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == Approx(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == Approx(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));
        REQUIRE(jet[6]
                == Approx(1 / 6. * 3 / 2. * (1 / 2. * 1 / std::sqrt(3) * jet[3] * jet[3] + std::sqrt(3) * jet[5] * 2)));
        REQUIRE(jet[7]
                == Approx(
                    1 / 6.
                    * (15 / 4. * std::pow(2, -7 / 2.) * jet[2] * jet[2] - 3 / 2. * std::pow(2, -5 / 2.) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(3_dbl, 3_dbl / 2_dbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (std::pow(3, 3 / 2.) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (std::pow(3, 3 / 2.) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6. * jet[5] * 2));
    }
}
