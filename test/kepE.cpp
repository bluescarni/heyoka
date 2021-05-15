// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <sstream>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>

#include "catch.hpp"

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

using namespace heyoka;

TEST_CASE("kepE def ctor")
{
    detail::kepE_impl k;

    REQUIRE(k.args().size() == 2u);
    REQUIRE(k.args()[0] == 0_dbl);
    REQUIRE(k.args()[1] == 0_dbl);
}

TEST_CASE("kepE diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(kepE(x, y), x) == sin(kepE(x, y)) / (1_dbl - x * cos(kepE(x, y))));
    REQUIRE(diff(kepE(x, y), y) == 1_dbl / (1_dbl - x * cos(kepE(x, y))));
    auto E = kepE(x * x, x * y);
    REQUIRE(diff(E, x) == (2_dbl * x * sin(E) + y) / (1_dbl - x * x * cos(E)));
    REQUIRE(diff(E, y) == x / (1_dbl - x * x * cos(E)));
}

TEST_CASE("kepE decompose")
{
    {
        auto [u0, u1] = make_vars("u_0", "u_1");

        std::vector<std::pair<expression, std::vector<std::uint32_t>>> dec;
        dec.emplace_back("e"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("M"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(kepE(u0, u1), dec);

        REQUIRE(dec.size() == 6u);

        REQUIRE(dec[2].first == kepE(u0, u1));
        REQUIRE(dec[2].second == std::vector<std::uint32_t>{5, 3});

        REQUIRE(dec[3].first == sin("u_2"_var));
        REQUIRE(dec[3].second == std::vector<std::uint32_t>{4});

        REQUIRE(dec[4].first == cos("u_2"_var));
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{3});

        REQUIRE(dec[5].first == "u_0"_var * "u_4"_var);
        REQUIRE(dec[5].second.empty());
    }

    {
        auto [u0, u1] = make_vars("u_0", "u_1");

        std::vector<std::pair<expression, std::vector<std::uint32_t>>> dec;
        dec.emplace_back("e"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("M"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(kepE(u0 + u1, u1 - u0), dec);

        REQUIRE(dec.size() == 8u);

        REQUIRE(dec[2].first == "u_0"_var + "u_1"_var);
        REQUIRE(dec[2].second.empty());

        REQUIRE(dec[3].first == "u_1"_var - "u_0"_var);
        REQUIRE(dec[3].second.empty());

        REQUIRE(dec[4].first == kepE("u_2"_var, "u_3"_var));
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{7, 5});

        REQUIRE(dec[5].first == sin("u_4"_var));
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{6});

        REQUIRE(dec[6].first == cos("u_4"_var));
        REQUIRE(dec[6].second == std::vector<std::uint32_t>{5});

        REQUIRE(dec[7].first == "u_2"_var * "u_6"_var);
        REQUIRE(dec[7].second.empty());
    }
}

TEST_CASE("kepE overloads")
{
    auto k = kepE("x"_var, 1.1);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1});

    k = kepE("x"_var, 1.1l);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = kepE("x"_var, mppp::real128{"1.1"});
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{mppp::real128{"1.1"}});
#endif

    k = kepE(1.1, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1});

    k = kepE(1.1l, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = kepE(mppp::real128{"1.1"}, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{mppp::real128{"1.1"}});
#endif
}
