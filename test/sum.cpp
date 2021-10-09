// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sum.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum");
    }

    {
        detail::sum_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == "()");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y)");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y + z)");
    }

    {
        std::ostringstream oss;

        oss << sum({x, y, z}, 2u);

        REQUIRE(oss.str() == "((x + y) + z)");
    }

    {
        std::ostringstream oss;

        oss << sum({x, y, z, x - y}, 2u);

        REQUIRE(oss.str() == "((x + y) + (z + (x - y)))");
    }

    {
        std::ostringstream oss;

        oss << sum({x, par[42], z, 4_dbl}, 2u);

        REQUIRE(boost::starts_with(oss.str(), "((x + p42) + (z + 4"));
    }
}

TEST_CASE("diff test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;
        REQUIRE(ss.gradient().empty());
    }

    {
        detail::sum_impl ss({x, y, z});
        REQUIRE(ss.gradient() == std::vector{1_dbl, 1_dbl, 1_dbl});
    }

    {
        REQUIRE(diff(sum({x, y, z}), "x") == 1_dbl);
        REQUIRE(diff(sum({x, x * x, z}), "x") == 1_dbl + 2_dbl * x);
        REQUIRE(diff(sum({x, x * x, -z}), "z") == -1_dbl);
    }
}

TEST_CASE("sum function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum({}) == 0_dbl);
    REQUIRE(sum({x}) == x);

    REQUIRE_THROWS_MATCHES(sum({x}, 0), std::invalid_argument,
                           Message("The 'split' value for a sum must be at least 2, but it is 0 instead"));
    REQUIRE_THROWS_MATCHES(sum({x}, 1), std::invalid_argument,
                           Message("The 'split' value for a sum must be at least 2, but it is 1 instead"));

    REQUIRE(sum({x, y, z, t}, 2) == sum({sum({x, y}), sum({z, t})}));
    REQUIRE(sum({x, y, z, t}, 3) == sum({sum({x, y, z}), sum({t})}));
    REQUIRE(sum({x, y, z, t}, 4) == sum({x, y, z, t}));
    REQUIRE(sum({x, y, z, t, 2_dbl * x}, 3) == sum({sum({x, y, z}), sum({t, 2_dbl * x})}));
}
