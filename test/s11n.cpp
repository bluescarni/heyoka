// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <optional>
#include <sstream>
#include <string>

#include <heyoka/s11n.hpp>

#include "catch.hpp"

// NOTE: test some corner cases that
// do not show up in other testing.
TEST_CASE("optional s11n")
{
    std::stringstream ss;

    std::optional<std::string> opt;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << opt;
    }

    opt = "hello";

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> opt;
    }

    REQUIRE(!opt);
}
