// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <heyoka/config.hpp>

// #include <cmath>
// #include <cstdint>
// #include <sstream>
// #include <variant>
#include <sstream>
#include <stdexcept>
#include <vector>

// #include <fmt/format.h>
// #include <fmt/ranges.h>

// #if defined(HEYOKA_HAVE_REAL128)

// #include <mp++/real128.hpp>

// #endif

#include <heyoka/expression.hpp>
// #include <heyoka/func.hpp>
// #include <heyoka/llvm_state.hpp>
// #include <heyoka/math/cos.hpp>
// #include <heyoka/math/kepE.hpp>
// #include <heyoka/math/pow.hpp>
// #include <heyoka/math/sin.hpp>
// #include <heyoka/math/sqrt.hpp>
// #include <heyoka/number.hpp>
// #include <heyoka/taylor.hpp>
#include <heyoka/math/sum.hpp>

#include "catch.hpp"
// #include "test_utils.hpp"

// #if defined(_MSC_VER) && !defined(__clang__)

// // NOTE: MSVC has issues with the other "using"
// // statement form.
// using namespace fmt::literals;

// #else

// using fmt::literals::operator""_format;

// #endif

using namespace heyoka;
// using namespace heyoka_test;

TEST_CASE("basic test")
{
    using Catch::Matchers::Message;

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

    REQUIRE_THROWS_MATCHES(detail::sum_impl{{par[0]}}, std::invalid_argument,
                           Message("The 'sum()' function accepts only variables or functions as arguments, "
                                   "but the expression 'p0' is neither"));
    REQUIRE_THROWS_AS((detail::sum_impl{{x, 0_dbl}}), std::invalid_argument);
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

    // TODO: test on expression too.
}
