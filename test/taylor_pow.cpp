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
#include <limits>
#include <stdexcept>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

#if defined(HEYOKA_HAVE_REAL128)

TEST_CASE("f128")
{
    using Catch::Matchers::Message;
    using namespace mppp::literals;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(y, 3_f128 / 2_f128), pow(x, -3_f128 / 2_f128)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3_rq, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2_rq, -3 / 2._rq)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(y, 3_f128 / 2_f128), pow(x, -3_f128 / 2_f128)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));
        REQUIRE(jet[4] == approximately(0.5_rq * 3 / 2._rq * sqrt(3._rq) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5_rq * -3 / 2._rq * pow(2, -5 / 2._rq) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == approximately(pow(2, -3 / 2._rq)));
        REQUIRE(jet[4] == approximately(0.5_rq * 3 / 2._rq * sqrt(3._rq) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5_rq * -3 / 2._rq * pow(2, -5 / 2._rq) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6._rq * 3 / 2._rq
                                 * (1 / 2._rq * 1 / sqrt(3._rq) * jet[3] * jet[3] + sqrt(3._rq) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(1 / 6._rq
                                 * (15 / 4._rq * pow(2, -7 / 2._rq) * jet[2] * jet[2]
                                    - 3 / 2._rq * pow(2, -5 / 2._rq) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {pow(3_f128, 3_f128 / 2_f128), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5_rq * (pow(3, 3 / 2._rq) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(pow(3, 3 / 2._rq)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5_rq * (pow(3, 3 / 2._rq) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6._rq * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_f128("jet", {pow(1_f128, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_f128("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}

#endif

TEST_CASE("dbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(y, 3_dbl / 2_dbl), pow(x, -3_dbl / 2_dbl)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {pow(y, 3_dbl / 2_dbl), pow(x, -3_dbl / 2_dbl)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6. * 3 / 2.
                                 * (1 / 2. * 1 / std::sqrt(3) * jet[3] * jet[3] + std::sqrt(3) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(
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
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6. * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_dbl("jet", {pow(1_dbl, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_dbl("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}

TEST_CASE("ldbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(y, 3_ldbl / 2_ldbl), pow(x, -3_ldbl / 2_ldbl)}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(y, 3_ldbl / 2_ldbl), pow(x, -3_ldbl / 2_ldbl)}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3.l) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.l) * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == approximately(std::pow(2, -3 / 2.l)));
        REQUIRE(jet[4] == approximately(0.5 * 3 / 2. * std::sqrt(3.l) * jet[3]));
        REQUIRE(jet[5] == approximately(0.5 * -3 / 2. * std::pow(2, -5 / 2.l) * jet[2]));
        REQUIRE(jet[6]
                == approximately(1 / 6.l * 3 / 2.
                                 * (1 / 2. * 1 / std::sqrt(3.l) * jet[3] * jet[3] + std::sqrt(3.l) * jet[5] * 2)));
        REQUIRE(jet[7]
                == approximately(1 / 6.l
                                 * (15 / 4. * std::pow(2, -7 / 2.l) * jet[2] * jet[2]
                                    - 3 / 2. * std::pow(2, -5 / 2.l) * jet[4] * 2)));
    }

    {
        // Number-number test.
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {pow(3_ldbl, 3_ldbl / 2_ldbl), x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.l) + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(std::pow(3, 3 / 2.l)));
        REQUIRE(jet[3] == 5);
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(0.5 * (std::pow(3, 3 / 2.l) + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6.l * jet[5] * 2));
    }

    // Failure modes for non-implemented cases.
    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_ldbl("jet", {pow(1_ldbl, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }

    {
        llvm_state s{"", 0};

        REQUIRE_THROWS_MATCHES(
            s.add_taylor_jet_ldbl("jet", {y, pow(y, x)}, 3), std::invalid_argument,
            Message("An invalid argument type was encountered while trying to build the Taylor derivative of a pow()"));
    }
}
