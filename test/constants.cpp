// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

class nc00_impl : public detail::constant_impl
{

public:
    nc00_impl() {}
};

class nc01_impl : public detail::constant_impl
{

public:
    nc01_impl() : detail::constant_impl("foo", number{0.}) {}
};

TEST_CASE("api test")
{
    using Catch::Matchers::Message;

    func f00{nc00_impl{}};

    REQUIRE(f00.get_name() == "null_constant");
    REQUIRE(f00.args().empty());

    REQUIRE_THROWS_MATCHES(
        func{nc01_impl{}}, std::invalid_argument,
        Message("A constant can be initialised only from a floating-point value with the maximum precision"));
}

TEST_CASE("pi stream")
{
    std::ostringstream oss;

    oss << heyoka::pi;

    REQUIRE(oss.str() == "pi");
}

TEST_CASE("pi diff")
{
    REQUIRE(diff(heyoka::pi, "x") == 0_dbl);

    auto x = "x"_var;

    REQUIRE(diff(heyoka::pi * cos(2. * x + 2. * heyoka::pi), "x")
            == heyoka::pi * (-2. * sin(2. * x + 2. * heyoka::pi)));
}

TEST_CASE("pi s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = heyoka::pi + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == heyoka::pi + x);
}
