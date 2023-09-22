// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("param basic")
{
    param p;

    REQUIRE(p.idx() == 0u);

    param p2{23};
    p2 = p;
    REQUIRE(p2.idx() == 0u);
    REQUIRE(p2 == p);
    REQUIRE(!(p2 != p));
}

TEST_CASE("param s11n")
{
    param p{42};

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << p;
    }

    p = param{0};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> p;
    }

    REQUIRE(p.idx() == 42u);
}
