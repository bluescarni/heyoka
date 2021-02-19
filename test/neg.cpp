// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/neg.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("neg ostream")
{
    auto [x, y] = make_vars("x", "y");

    std::ostringstream oss;
    oss << neg(x + y);

    REQUIRE(oss.str() == "-(x + y)");

    oss.str("");
    oss << -x;

    REQUIRE(oss.str() == "-x");
}

TEST_CASE("unary minus simpl")
{
    REQUIRE(-1_dbl == expression{-1.});
    REQUIRE(-1.1_ldbl == expression{-1.1l});

    auto [x] = make_vars("x");

    REQUIRE(-x == neg(x));
}
