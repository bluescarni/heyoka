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

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

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

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {expression{binary_operator{binary_operator::type::mul, 2_f128, 3_f128}}, x + y},
                              1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {expression{binary_operator{binary_operator::type::mul, 2_f128, 3_f128}}, x + y},
                              2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5_rq * (6_rq + jet[3])));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {expression{binary_operator{binary_operator::type::mul, 2_f128, 3_f128}}, x + y},
                              3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5_rq * (6_rq + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(5_rq));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5_rq * (6_rq + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6_rq * (2 * jet[5])));
    }

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {y * 2_f128, x * -4_f128}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {y * 2_f128, x * -4_f128}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {y * 2_f128, x * -4_f128}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6_rq * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3_rq * jet[4]));
    }

    // Number/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {2_f128 * y, -4_f128 * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {2_f128 * y, -4_f128 * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {2_f128 * y, -4_f128 * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(-8_rq));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6_rq * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3_rq * jet[4]));
    }

    // Variable/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {x * y, y * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[4] = {2_rq, 3_rq};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(6_rq));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {x * y, y * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[6] = {2_rq, 3_rq};

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(6_rq));
        REQUIRE(jet[4] == approximately(.5_rq * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5_rq * (jet[2] * 3 + jet[3] * 2)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_f128("jet", {x * y, y * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_f128("jet");

        mppp::real128 jet[8] = {2_rq, 3_rq};

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6_rq));
        REQUIRE(jet[3] == approximately(6_rq));
        REQUIRE(jet[4] == approximately(.5_rq * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5_rq * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[6] == approximately(1 / 6_rq * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2_rq * 2 * jet[5])));
        REQUIRE(jet[7] == approximately(1 / 6_rq * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2_rq * 2 * jet[5])));
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

        s.add_taylor_jet_dbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_dbl, 3_dbl}}, x + y}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_dbl, 3_dbl}}, x + y}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5 * (6. + jet[3])));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_dbl, 3_dbl}}, x + y}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5 * (6. + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(5.));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5 * (6. + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6. * (2 * jet[5])));
    }

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {y * 2_dbl, x * -4_dbl}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {y * 2_dbl, x * -4_dbl}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {y * 2_dbl, x * -4_dbl}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6. * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3. * jet[4]));
    }

    // Number/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {2_dbl * y, -4_dbl * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {2_dbl * y, -4_dbl * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {2_dbl * y, -4_dbl * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(-8.));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6. * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3. * jet[4]));
    }

    // Variable/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {x * y, y * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(6.));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {x * y, y * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[6] = {2, 3};

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(6.));
        REQUIRE(jet[4] == approximately(.5 * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5 * (jet[2] * 3 + jet[3] * 2)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_dbl("jet", {x * y, y * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_dbl("jet");

        double jet[8] = {2, 3};

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.));
        REQUIRE(jet[3] == approximately(6.));
        REQUIRE(jet[4] == approximately(.5 * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5 * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[6] == approximately(1 / 6. * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2. * 2 * jet[5])));
        REQUIRE(jet[7] == approximately(1 / 6. * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2. * 2 * jet[5])));
    }
}

TEST_CASE("ldbl")
{
    using Catch::Matchers::Message;

    auto x = "x"_var, y = "y"_var;

    // Number-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_ldbl, 3_ldbl}}, x + y},
                              1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_ldbl, 3_ldbl}}, x + y},
                              2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5l * (6.l + jet[3])));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {expression{binary_operator{binary_operator::type::mul, 2_ldbl, 3_ldbl}}, x + y},
                              3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5l * (6.l + jet[3])));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(5.l));
        REQUIRE(jet[4] == 0);
        REQUIRE(jet[5] == approximately(.5l * (6.l + jet[3])));
        REQUIRE(jet[6] == 0);
        REQUIRE(jet[7] == approximately(1 / 6.l * (2 * jet[5])));
    }

    // Variable-number tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {y * 2_ldbl, x * -4_ldbl}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {y * 2_ldbl, x * -4_ldbl}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {y * 2_ldbl, x * -4_ldbl}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6.l * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3.l * jet[4]));
    }

    // Number/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {2_ldbl * y, -4_ldbl * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {2_ldbl * y, -4_ldbl * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {2_ldbl * y, -4_ldbl * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(-8.l));
        REQUIRE(jet[4] == approximately(jet[3]));
        REQUIRE(jet[5] == approximately(-2 * jet[2]));
        REQUIRE(jet[6] == approximately(1 / 6.l * 4 * jet[5]));
        REQUIRE(jet[7] == approximately(-4. / 3.l * jet[4]));
    }

    // Variable/variable tests.
    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {x * y, y * x}, 1);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[4] = {2, 3};

        jptr(jet, 1);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(6.l));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {x * y, y * x}, 2);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[6] = {2, 3};

        jptr(jet, 2);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(6.l));
        REQUIRE(jet[4] == approximately(.5l * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5l * (jet[2] * 3 + jet[3] * 2)));
    }

    {
        llvm_state s{"", 0};

        s.add_taylor_jet_ldbl("jet", {x * y, y * x}, 3);

        s.compile();

        auto jptr = s.fetch_taylor_jet_ldbl("jet");

        long double jet[8] = {2, 3};

        jptr(jet, 3);

        REQUIRE(jet[0] == 2);
        REQUIRE(jet[1] == 3);
        REQUIRE(jet[2] == approximately(6.l));
        REQUIRE(jet[3] == approximately(6.l));
        REQUIRE(jet[4] == approximately(.5l * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[5] == approximately(.5l * (jet[2] * 3 + jet[3] * 2)));
        REQUIRE(jet[6] == approximately(1 / 6.l * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2.l * 2 * jet[5])));
        REQUIRE(jet[7] == approximately(1 / 6.l * (jet[4] * 2 * 3 + 2 * jet[2] * jet[3] + 2.l * 2 * jet[5])));
    }
}
