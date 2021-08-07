// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("atan2 diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(atan2(y, x), "x") == (-y) / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "y") == x / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "z") == 0_dbl);
    REQUIRE(diff(atan2(x * y, y / x), "x")
            == (y / x * y - (x * y) * (-y / (x * x))) / ((y / x) * (y / x) + (x * y) * (x * y)));
}

TEST_CASE("atan2 decompose")
{
    auto [u0, u1] = make_vars("u_0", "u_1");

    {
        taylor_dc_t dec;
        dec.emplace_back("y"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("x"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(atan2(u0, u1), dec);

        REQUIRE(dec.size() == 6u);

        REQUIRE(dec[2].first == u1 * u1);
        REQUIRE(dec[2].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[3].first == u0 * u0);
        REQUIRE(dec[3].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[4].first == "u_2"_var + "u_3"_var);
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[5].first == atan2(u0, u1));
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{4});
    }

    {
        taylor_dc_t dec;
        dec.emplace_back("y"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("x"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(atan2(u0 + u1, u1 - u0), dec);

        REQUIRE(dec.size() == 8u);

        REQUIRE(dec[2].first == u0 + u1);
        REQUIRE(dec[2].second.empty());

        REQUIRE(dec[3].first == u1 - u0);
        REQUIRE(dec[3].second.empty());

        REQUIRE(dec[4].first == "u_3"_var * "u_3"_var);
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[5].first == "u_2"_var * "u_2"_var);
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[6].first == "u_4"_var + "u_5"_var);
        REQUIRE(dec[6].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[7].first == atan2("u_2"_var, "u_3"_var));
        REQUIRE(dec[7].second == std::vector<std::uint32_t>{6});
    }
}
