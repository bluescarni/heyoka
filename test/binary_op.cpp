// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    using binary_op = detail::binary_op;

    REQUIRE(binary_op{}.op() == binary_op::type::add);
    REQUIRE(binary_op{}.lhs() == 0_dbl);
    REQUIRE(binary_op{}.rhs() == 0_dbl);

    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.op() == binary_op::type::div);

    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.lhs() == 1_dbl);
    REQUIRE(binary_op{binary_op::type::div, 1_dbl, 2_dbl}.rhs() == 2_dbl);

    {
        const binary_op op{binary_op::type::div, "x"_var, 2_dbl};

        REQUIRE(op.lhs() == "x"_var);
        REQUIRE(op.rhs() == 2_dbl);
    }

    {
        binary_op op{binary_op::type::div, "x"_var, 2_dbl};

        op.lhs() = 1_dbl;
        op.rhs() = "y"_var;

        REQUIRE(op.lhs() == 1_dbl);
        REQUIRE(op.rhs() == "y"_var);
    }
}

TEST_CASE("stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;

        oss << x + y;

        REQUIRE(oss.str() == "(x + y)");
    }

    {
        std::ostringstream oss;

        oss << x - y;

        REQUIRE(oss.str() == "(x - y)");
    }

    {
        std::ostringstream oss;

        oss << x * y;

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        oss << x / y;

        REQUIRE(oss.str() == "(x / y)");
    }
}

TEST_CASE("equality")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(add(x, y) == add(x, y));
    REQUIRE(add(x, y) != sub(x, y));
    REQUIRE(add(x, y) != mul(x, y));
    REQUIRE(add(x, y) != div(x, y));
}

TEST_CASE("hashing")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(hash(add(x, y)) == hash(add(x, y)));
    REQUIRE(hash(add(x, y)) != hash(sub(x, y)));
    REQUIRE(hash(add(x, y)) != hash(mul(x, y)));
    REQUIRE(hash(add(x, y)) != hash(div(x, y)));
}

TEST_CASE("diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(x + y, "x") == 1_dbl);
    REQUIRE(diff(x - y, "y") == -1_dbl);
    REQUIRE(diff(x * y, "x") == y);
    REQUIRE(diff(x / y, "x") == y / (y * y));
}

TEST_CASE("asin s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = x + y;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = x - y;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == x + y);
}
