// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_sq_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum_sq");
    }

    {
        detail::sum_sq_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_sq_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == "()");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x**2");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2)");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2 + z**2)");
    }

    {
        std::ostringstream oss;

        oss << sum_sq({x, y, z}, 2u);

        REQUIRE(oss.str() == "((x**2 + y**2) + z**2)");
    }

    {
        std::ostringstream oss;

        oss << sum_sq({x, y, z, x - y}, 2u);

        REQUIRE(oss.str() == "((x**2 + y**2) + (z**2 + (x - y)**2))");
    }

    {
        std::ostringstream oss;

        oss << sum_sq({x, par[42], z, 4_dbl}, 2u);

        REQUIRE(boost::starts_with(oss.str(), "((x**2 + p42**2) + (z**2 + 4"));
    }
}

TEST_CASE("diff test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_sq_impl ss;
        REQUIRE(diff(expression(func(ss)), "x") == 0_dbl);
    }

    {
        detail::sum_sq_impl ss({x, y, z});
        REQUIRE(diff(expression(func(ss)), "x") == 2_dbl * "x"_var);
        REQUIRE(diff(expression(func(ss)), "y") == 2_dbl * "y"_var);
        REQUIRE(diff(expression(func(ss)), "z") == 2_dbl * "z"_var);
        REQUIRE(diff(expression(func(ss)), par[0]) == 0_dbl);
    }

    {
        detail::sum_sq_impl ss({par[0], par[1], par[2]});
        REQUIRE(diff(expression(func(ss)), par[0]) == 2_dbl * par[0]);
        REQUIRE(diff(expression(func(ss)), par[1]) == 2_dbl * par[1]);
        REQUIRE(diff(expression(func(ss)), par[2]) == 2_dbl * par[2]);
        REQUIRE(diff(expression(func(ss)), "x") == 0_dbl);
    }

    {
        REQUIRE(diff(sum_sq({x, y, z}), "x") == 2_dbl * x);
        REQUIRE(diff(sum_sq({x, x * x, z}), "x") == 2_dbl * sum({x, (x * x) * (2_dbl * x)}));
        REQUIRE(diff(sum_sq({x, x * x, -z}), "z") == 2_dbl * z);
    }

    {
        REQUIRE(diff(sum_sq({par[0] - 1_dbl, par[1] + y, par[0] + x}), par[0])
                == 2_dbl * sum({par[0] - 1_dbl, par[0] + x}));
    }
}

TEST_CASE("sum_sq function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum_sq({}) == 0_dbl);
    REQUIRE(sum_sq({x}) == x * x);

    REQUIRE_THROWS_MATCHES(sum_sq({x}, 0), std::invalid_argument,
                           Message("The 'split' value for a sum of squares must be at least 2, but it is 0 instead"));
    REQUIRE_THROWS_MATCHES(sum_sq({x}, 1), std::invalid_argument,
                           Message("The 'split' value for a sum of squares must be at least 2, but it is 1 instead"));

    REQUIRE(sum_sq({x, y, z, t}, 2) == sum({sum_sq({x, y}), sum_sq({z, t})}));
    REQUIRE(sum_sq({x, y, z, t}, 3) == sum({sum_sq({x, y, z}), sum_sq({t})}));
    REQUIRE(sum_sq({x, y, z, t}, 4) == sum_sq({x, y, z, t}));
    REQUIRE(sum_sq({x, y, z, t, 2_dbl * x}, 3) == sum({sum_sq({x, y, z}), sum_sq({t, 2_dbl * x})}));
    REQUIRE(sum_sq({0_dbl, y, 0_dbl, t, 2_dbl * x}, 3) == sum_sq({y, t, 2_dbl * x}));
}

TEST_CASE("sum_sq s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = sum_sq({x, y, z});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sum_sq({x, y, z}));
}

TEST_CASE("sum_sq zero ignore")
{
    REQUIRE(sum_sq({1_dbl, 2_dbl, 0_dbl}) == sum_sq({1_dbl, 2_dbl}));
    REQUIRE(sum_sq({1_dbl, 0_dbl, 1_dbl}) == sum_sq({1_dbl, 1_dbl}));
    REQUIRE(sum_sq({0_dbl, 0_dbl, 0_dbl}) == 0_dbl);
    REQUIRE(sum_sq({0_dbl, -1_dbl, 0_dbl}) == 1_dbl);

    REQUIRE(sum_sq({0_dbl, 2_dbl, "x"_var}) == sum_sq({2_dbl, "x"_var}));
    REQUIRE(sum_sq({0_dbl, 2_dbl, "x"_var, 0_dbl}) == sum_sq({2_dbl, "x"_var}));
    REQUIRE(sum_sq({0_dbl, 2_dbl, 0_dbl, "x"_var, 0_dbl}) == sum_sq({2_dbl, "x"_var}));

    REQUIRE(std::get<func>(sum_sq({"y"_var, 0_dbl, "x"_var, -21_dbl}).value()).args()
            == std::vector{"y"_var, "x"_var, -21_dbl});
}
