// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("dbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(2_dbl) + cos(3_dbl), x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(2_dbl) + cos(3_dbl), x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == 0.5 * (jet[2] + jet[3]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(2_dbl) + cos(3_dbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (jet[2] + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.) + std::cos(3.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (jet[2] + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6. * (2 * jet[4] + 2 * jet[5])));
    }

    // Variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(y), cos(x)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(y), cos(x)}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));
        REQUIRE(jet[4] == Approx(0.5 * std::cos(3.) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5 * std::sin(2.) * jet[2]));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {sin(y), cos(x)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));
        REQUIRE(jet[4] == Approx(0.5 * std::cos(3.) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5 * std::sin(2.) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.)));
        REQUIRE(jet[3] == Approx(std::cos(2.)));
        REQUIRE(jet[4] == Approx(0.5 * std::cos(3.) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5 * std::sin(2.) * jet[2]));
        REQUIRE(jet[6] == Approx(1 / 6. * (-std::sin(3.) * jet[3] * jet[3] + std::cos(3.) * 2 * jet[5])));
        REQUIRE(jet[7] == Approx(1 / 6. * (-std::cos(2.) * jet[2] * jet[2] - std::sin(2.) * 2 * jet[4])));
    }
}

TEST_CASE("ldbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(2_ldbl) + cos(3_ldbl), x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(2_ldbl) + cos(3_ldbl), x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == 0.5l * (jet[2] + jet[3]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(2_ldbl) + cos(3_ldbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5l * (jet[2] + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(2.l) + std::cos(3.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5l * (jet[2] + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6.l * (2 * jet[4] + 2 * jet[5])));
    }

    // Variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(y), cos(x)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(y), cos(x)}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * std::cos(3.l) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5l * std::sin(2.l) * jet[2]));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {sin(y), cos(x)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * std::cos(3.l) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5l * std::sin(2.l) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::sin(3.l)));
        REQUIRE(jet[3] == Approx(std::cos(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * std::cos(3.l) * jet[3]));
        REQUIRE(jet[5] == Approx(-0.5l * std::sin(2.l) * jet[2]));
        REQUIRE(jet[6] == Approx(1 / 6.l * (-std::sin(3.l) * jet[3] * jet[3] + std::cos(3.l) * 2 * jet[5])));
        REQUIRE(jet[7] == Approx(1 / 6.l * (-std::cos(2.l) * jet[2] * jet[2] - std::sin(2.l) * 2 * jet[4])));
    }
}
