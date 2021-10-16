// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <boost/algorithm/string/predicate.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/math/sin.hpp>
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

TEST_CASE("diff var")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(x + y, "x") == 1_dbl);
    REQUIRE(diff(x - y, "y") == -1_dbl);
    REQUIRE(diff(x * y, "x") == y);
    REQUIRE(diff(x / y, "x") == y / (y * y));
}

TEST_CASE("diff par")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(par[0] + y, par[0]) == 1_dbl);
    REQUIRE(diff(par[0] + y, par[1]) == 0_dbl);

    REQUIRE(diff(x - par[0], par[0]) == -1_dbl);
    REQUIRE(diff(x - par[0], par[1]) == 0_dbl);

    REQUIRE(diff(par[2] * y, par[2]) == y);
    REQUIRE(diff(par[2] * y, par[1]) == 0_dbl);

    REQUIRE(diff(par[3] / y, par[3]) == y / (y * y));
    REQUIRE(diff(par[3] / y, par[4]) == 0_dbl);
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

TEST_CASE("neg stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;
        oss << mul(-1_dbl, x);

        REQUIRE(oss.str() == "-x");
    }

    {
        std::ostringstream oss;
        oss << mul(-1_dbl, x + y);

        REQUIRE(oss.str() == "-(x + y)");
    }

    {
        std::ostringstream oss;
        oss << mul(x + y, -1_dbl);

        REQUIRE(oss.str() == "-(x + y)");
    }

    {
        std::ostringstream oss;
        oss << mul(-1_dbl, par[0]);

        REQUIRE(oss.str() == "-p0");
    }

    {
        std::ostringstream oss;
        oss << mul(par[0], -1_dbl);

        REQUIRE(oss.str() == "-p0");
    }

    {
        std::ostringstream oss;
        oss << mul(-1_dbl, -5_dbl);

        REQUIRE(boost::starts_with(oss.str(), "--5.0000"));
    }

    {
        std::ostringstream oss;
        oss << mul(-5_dbl, -1_dbl);

        REQUIRE(boost::starts_with(oss.str(), "--5.0000"));
    }
}

TEST_CASE("neg diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(-1_dbl * (x + y), "x") == -1_dbl);
    REQUIRE(diff(-(x + y), "x") == -1_dbl);
    REQUIRE(diff(-(x * x + y * x), "x") == -(2. * x + y));

    REQUIRE(diff(-(par[0] + y), par[0]) == -1_dbl);
    REQUIRE(diff(-(par[0] + y), par[0]) == -1_dbl);
    REQUIRE(diff(-(x * x + par[1] * x), par[1]) == -x);
}

TEST_CASE("unary minus simpl")
{
    REQUIRE(-1_dbl == expression{-1.});
    REQUIRE(-1.1_ldbl == expression{-1.1l});

    auto [x] = make_vars("x");

    REQUIRE(-x == -1_dbl * x);
}

TEST_CASE("unary minus minus simpl")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(-(-(x + y)) == x + y);
    REQUIRE(-(-sin(x + y)) == sin(x + y));
    REQUIRE(-sin(x + y) == -1_dbl * (sin(x + y)));
}
