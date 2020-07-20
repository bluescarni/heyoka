// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

#if defined(HEYOKA_HAVE_REAL128)

TEST_CASE("f128")
{
    using namespace mppp::literals;

    auto [x, y, z] = make_vars("x", "y", "z");

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {prime(x) = 1_f128, prime(y) = -2_f128, prime(z) = 0_f128}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq, 4_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {prime(x) = 1_f128, prime(y) = -2_f128, prime(z) = 0_f128}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[9] = {2_rq, 3_rq, 4_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {prime(x) = 1_f128, prime(y) = -2_f128, prime(z) = 0_f128}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[12] = {2_rq, 3_rq, 4_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
        REQUIRE(jet[9] == 0);
        REQUIRE(jet[10] == 0);
        REQUIRE(jet[11] == 0);
    }
}

#endif

TEST_CASE("dbl")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {prime(x) = 1_dbl, prime(y) = -2_dbl, prime(z) = 0_dbl}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {prime(x) = 1_dbl, prime(y) = -2_dbl, prime(z) = 0_dbl}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[9] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {prime(x) = 1_dbl, prime(y) = -2_dbl, prime(z) = 0_dbl}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[12] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
        REQUIRE(jet[9] == 0);
        REQUIRE(jet[10] == 0);
        REQUIRE(jet[11] == 0);
    }
}

TEST_CASE("ldbl")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {prime(x) = 1_ldbl, prime(y) = -2_ldbl, prime(z) = 0_ldbl}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {prime(x) = 1_ldbl, prime(y) = -2_ldbl, prime(z) = 0_ldbl}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[9] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {prime(x) = 1_ldbl, prime(y) = -2_ldbl, prime(z) = 0_ldbl}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[12] = {2, 3, 4};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == 4);
        REQUIRE(jet[3] == 1);
        REQUIRE(jet[4] == -2);
        REQUIRE(jet[5] == 0);
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == 0);
        REQUIRE(jet[8] == 0);
        REQUIRE(jet[9] == 0);
        REQUIRE(jet[10] == 0);
        REQUIRE(jet[11] == 0);
    }
}
