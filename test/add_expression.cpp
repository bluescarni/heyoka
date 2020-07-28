// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>
#include <vector>

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

using namespace mppp::literals;

#endif

TEST_CASE("vararg expression")
{
    auto [x, y, z] = make_vars("x", "y", "z");

#if defined(HEYOKA_HAVE_REAL128)
    {
        llvm_state s{""};

        s.add_nary_function<mppp::real128>("foo", x + 1.1_f128);

        s.compile();

        auto f = s.fetch_nary_function<mppp::real128, 1>("foo");

        REQUIRE(f(mppp::real128(1)) == 1 + mppp::real128{"1.1"});
    }

    {
        llvm_state s{""};

        s.add_nary_function<mppp::real128>("foo", log(x) + 3_dbl * log(x));

        s.compile();

        auto f = s.fetch_nary_function<mppp::real128, 1>("foo");

        REQUIRE(f(mppp::real128(2)) == approximately(4 * log(2_rq)));
    }
#endif
}

TEST_CASE("vector expression")
{
    auto [x, y, z] = make_vars("x", "y", "z");

#if defined(HEYOKA_HAVE_REAL128)
    {
        llvm_state s{""};

        s.add_function<mppp::real128>("foo", x + 1.1_f128);

        s.compile();

        auto f = s.fetch_function<mppp::real128>("foo");

        mppp::real128 args[] = {mppp::real128{1}};

        REQUIRE(f(args) == 1 + mppp::real128{"1.1"});
        REQUIRE(args[0] == mppp::real128{1});
    }
#endif

    {
        llvm_state s{""};

        s.add_function<double>("foo", x + y + z);

        s.compile();

        auto f = s.fetch_function<double>("foo");

        double args[] = {1, 2, 3};

        REQUIRE(f(args) == 6);
    }
}

TEST_CASE("batch expression")
{
    auto [x, y, z] = make_vars("x", "y", "z");

#if defined(HEYOKA_HAVE_REAL128)
    {
        llvm_state s{""};

        s.add_function_batch<mppp::real128>("foo", x + y + z, 4);

        std::vector<mppp::real128> out(4);
        std::vector<mppp::real128> in = {1_rq, 2_rq, 3_rq, 4_rq, 1_rq, 2_rq, 3_rq, 4_rq, 1_rq, 2_rq, 3_rq, 4_rq};

        s.compile();

        auto f = s.fetch_function_batch<mppp::real128>("foo");

        f(out.data(), in.data());

        REQUIRE(out[0] == 3);
        REQUIRE(out[1] == 6);
        REQUIRE(out[2] == 9);
        REQUIRE(out[3] == 12);
    }
#endif
}

TEST_CASE("vector expressions")
{
    auto [x, y, z] = make_vars("x", "y", "z");

#if defined(HEYOKA_HAVE_REAL128)
    {
        llvm_state s{""};

        s.add_vector_function<mppp::real128>("foo", {y / z, x * x - y * z});

        s.compile();

        auto f = s.fetch_vector_function<mppp::real128>("foo");

        mppp::real128 input[] = {1_rq, 2_rq, 3_rq};
        mppp::real128 output[2];

        f(output, input);

        REQUIRE(output[0] == approximately(2._rq / 3));
        REQUIRE(output[1] == approximately(1._rq - 2._rq * 3));
    }
#endif

    {
        llvm_state s{""};

        s.add_vector_function<double>("foo", {x + y, x * x - y * z});

        s.compile();

        auto f = s.fetch_vector_function<double>("foo");

        double input[] = {1, 2, 3};
        double output[2];

        f(output, input);

        REQUIRE(output[0] == approximately(1. + 2));
        REQUIRE(output[1] == approximately(1. - 2. * 3));
    }

    {
        llvm_state s{""};

        s.add_vector_function<long double>("foo", {y / z, x * x - y * z});

        s.compile();

        auto f = s.fetch_vector_function<long double>("foo");

        long double input[] = {1, 2, 3};
        long double output[2];

        f(output, input);

        REQUIRE(output[0] == approximately(2.l / 3));
        REQUIRE(output[1] == approximately(1.l - 2. * 3));
    }
}

TEST_CASE("tj batch")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state s{""};

    s.add_taylor_jet_batch_dbl("tjb", {prime(x) = y, prime(y) = (1_dbl - x * x) * y - x}, 20, 4);

    // std::cout << s.dump_ir() << '\n';
}

TEST_CASE("log tmp test")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state s{""};

    s.add_taylor_jet_batch_dbl("tjb", {prime(x) = log(y), prime(y) = log(x)}, 2, 4);

    // std::cout << s.dump_ir() << '\n';
}
