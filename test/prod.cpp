// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <heyoka/config.hpp>

// #include <algorithm>
// #include <cmath>
#include <initializer_list>
// #include <limits>
// #include <random>
#include <sstream>
// #include <stdexcept>
// #include <tuple>
// #include <type_traits>
// #include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <fmt/core.h>

// #include <llvm/Config/llvm-config.h>

// #if defined(HEYOKA_HAVE_REAL128)

// #include <mp++/real128.hpp>

// #endif

// #if defined(HEYOKA_HAVE_REAL)

// #include <mp++/real.hpp>

// #endif

#include <heyoka/expression.hpp>
// #include <heyoka/func.hpp>
// #include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
// #include <heyoka/s11n.hpp>

#include "catch.hpp"
// #include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
// using namespace heyoka_test;

// const auto fp_types = std::tuple<double
// #if !defined(HEYOKA_ARCH_PPC)
//                                  ,
//                                  long double
// #endif
// #if defined(HEYOKA_HAVE_REAL128)
//                                  ,
//                                  mppp::real128
// #endif
//                                  >{};

// constexpr bool skip_batch_ld =
// #if LLVM_VERSION_MAJOR >= 13 && LLVM_VERSION_MAJOR <= 16
//     std::numeric_limits<long double>::digits == 64
// #else
//     false
// #endif
//     ;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::prod_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "prod");
    }

    {
        detail::prod_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::prod_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::prod_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{}", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{}", -1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_ldbl, x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-x");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y * z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y * z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, -1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{} / (x * y)", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, -1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("-{} / (x * y)", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x / y");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-x / y");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -2_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "x / y**2.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -2_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-x / y**2.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y) / z");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y) / z");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "(x * y) / z**3.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-(x * y) / z**3.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -2_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "x / (y**2.0000"));
        REQUIRE(boost::contains(oss.str(), "z**3.0000"));
        REQUIRE(boost::contains(oss.str(), "0000)"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -2_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-x / (y**2.0000"));
        REQUIRE(boost::contains(oss.str(), "z**3.0000"));
        REQUIRE(boost::contains(oss.str(), "0000)"));
    }
}

TEST_CASE("args simpl")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    // Flattening.
    REQUIRE(prod({prod({x, y}), z}) == prod({x, y, z}));

    // Gathering of common bases with numerical exponents.
    REQUIRE(prod({x, pow(y, 3.), pow(x, 2.), pow(y, 4.)}) == prod({pow(x, 3.), pow(y, 7.)}));
    REQUIRE(prod({pow(y, 3.), pow(x, 2.), x, pow(y, 4.)}) == prod({pow(x, 3.), pow(y, 7.)}));

    // Constant folding.
    REQUIRE(prod({3_dbl, 4_dbl}) == 12_dbl);
    REQUIRE(prod({3_dbl, 4_dbl, x, -2_dbl}) == prod({x, -24_dbl}));
    REQUIRE(prod({.5_dbl, 2_dbl, x}) == x);
    REQUIRE(prod({pow(y, 3.), pow(x, -1.), x, pow(y, -3.)}) == 1_dbl);
    REQUIRE(prod({pow(y, 3.), pow(x, -1.), x, pow(y, -3.), 0_dbl}) == 0_dbl);

    // Special cases.
    REQUIRE(prod({}) == 1_dbl);
    REQUIRE(prod({x}) == x);

    // Sorting.
    REQUIRE(prod({y, z, x, 1_dbl}) == prod({1_dbl, x, y, z}));
}
