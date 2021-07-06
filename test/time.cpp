// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("time stream")
{
    std::ostringstream oss;

    oss << heyoka::time;

    REQUIRE(oss.str() == "t");
}

TEST_CASE("time diff")
{
    REQUIRE(diff(heyoka::time, "x") == 0_dbl);

    auto x = "x"_var;

    REQUIRE(diff(heyoka::time * cos(2. * x + 2. * heyoka::time), "x")
            == heyoka::time * (-2. * sin(2. * x + 2. * heyoka::time)));
}
