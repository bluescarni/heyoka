// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <heyoka/expression.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("revdiff decompose")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(detail::revdiff_decompose(x) == std::vector{x, "u_0"_var});
    REQUIRE(detail::revdiff_decompose(par[0]) == std::vector{par[0], "u_0"_var});
    REQUIRE(detail::revdiff_decompose(par[0] + x) == std::vector{x, par[0], "u_1"_var + "u_0"_var, "u_2"_var});
    REQUIRE(detail::revdiff_decompose((par[1] + y) * (par[0] + x))
            == std::vector{x, y, par[0], par[1], ("u_2"_var + "u_0"_var), ("u_3"_var + "u_1"_var),
                           ("u_5"_var * "u_4"_var), "u_6"_var});

    REQUIRE(detail::revdiff_decompose(subs((par[1] + y) * (par[0] + x), {{y, 1_dbl}}))
            == std::vector{x, par[0], par[1], ("u_1"_var + "u_0"_var), subs("u_2"_var + y, {{y, 1_dbl}}),
                           ("u_4"_var * "u_3"_var), "u_5"_var});
}
