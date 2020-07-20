// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>

#include "catch.hpp"

using namespace heyoka;

#if defined(HEYOKA_HAVE_REAL128)

TEST_CASE("f128")
{
    using Catch::Matchers::Message;
    using namespace mppp::literals;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(2_f128), x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(2_f128), x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == 0.5_rq * (jet[2] + jet[3]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(2_f128), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5_rq * (jet[2] + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5_rq * (jet[2] + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6._rq * (2 * jet[4] + 2 * jet[5])));
    }

    // Variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(y), log(x)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(y), log(x)}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));
        REQUIRE(jet[4] == Approx(0.5_rq * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5_rq * jet[2] / jet[0]));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {log(y), log(x)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));
        REQUIRE(jet[4] == Approx(0.5_rq * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5_rq * jet[2] / jet[0]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(log(3._rq)));
        REQUIRE(jet[3] == Approx(log(2._rq)));
        REQUIRE(jet[4] == Approx(0.5_rq * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5_rq * jet[2] / jet[0]));
        REQUIRE(jet[6] == Approx(1 / 6._rq * (jet[5] * 2 * jet[1] - jet[3] * jet[3]) / (jet[1] * jet[1])));
        REQUIRE(jet[7] == Approx(1 / 6._rq * (jet[4] * 2 * jet[0] - jet[2] * jet[2]) / (jet[0] * jet[0])));
    }
}

#endif

TEST_CASE("dbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(2_dbl), x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(2_dbl), x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == 0.5 * (jet[2] + jet[3]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(2_dbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (jet[2] + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5 * (jet[2] + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6. * (2 * jet[4] + 2 * jet[5])));
    }

    // Variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(y), log(x)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(y), log(x)}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));
        REQUIRE(jet[4] == Approx(0.5 * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5 * jet[2] / jet[0]));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {log(y), log(x)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));
        REQUIRE(jet[4] == Approx(0.5 * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5 * jet[2] / jet[0]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.)));
        REQUIRE(jet[3] == Approx(std::log(2.)));
        REQUIRE(jet[4] == Approx(0.5 * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5 * jet[2] / jet[0]));
        REQUIRE(jet[6] == Approx(1 / 6. * (jet[5] * 2 * jet[1] - jet[3] * jet[3]) / (jet[1] * jet[1])));
        REQUIRE(jet[7] == Approx(1 / 6. * (jet[4] * 2 * jet[0] - jet[2] * jet[2]) / (jet[0] * jet[0])));
    }
}

TEST_CASE("ldbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(2_ldbl), x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(2_ldbl), x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == 0.5l * (jet[2] + jet[3]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(2_ldbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5l * (jet[2] + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == Approx(0.5l * (jet[2] + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == Approx(1 / 6.l * (2 * jet[4] + 2 * jet[5])));
    }

    // Variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(y), log(x)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(y), log(x)}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5l * jet[2] / jet[0]));
    }
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {log(y), log(x)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5l * jet[2] / jet[0]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == Approx(std::log(3.l)));
        REQUIRE(jet[3] == Approx(std::log(2.l)));
        REQUIRE(jet[4] == Approx(0.5l * jet[3] / jet[1]));
        REQUIRE(jet[5] == Approx(0.5l * jet[2] / jet[0]));
        REQUIRE(jet[6] == Approx(1 / 6.l * (jet[5] * 2 * jet[1] - jet[3] * jet[3]) / (jet[1] * jet[1])));
        REQUIRE(jet[7] == Approx(1 / 6.l * (jet[4] * 2 * jet[0] - jet[2] * jet[2]) / (jet[0] * jet[0])));
    }
}
