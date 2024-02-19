// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("variable basics")
{
    variable var{"pippo"};

    REQUIRE(var.name() == "pippo");

    variable var2{"pluto"};

    REQUIRE(var != var2);
    REQUIRE(!(var == var2));
}

TEST_CASE("variable s11n")
{
    variable var{"pippo"};

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << var;
    }

    var = variable{"pluto"};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> var;
    }

    REQUIRE(var.name() == "pippo");
}
