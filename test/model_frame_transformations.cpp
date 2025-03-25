// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>

#include <heyoka/expression.hpp>
#include <heyoka/model/frame_transformations.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("rot_fk5j2000_icrs")
{
    const auto [x, y, z] = make_vars("x", "y", "z");
    const auto [icrs_x, icrs_y, icrs_z] = model::rot_fk5j2000_icrs({x, y, z});
    auto cf = cfunc<double>{{icrs_x, icrs_y, icrs_z}, {x, y, z}};

    {
        std::array<double, 3> in = {230940.10767585033, 230940.10767585033, 230940.10767585033}, out{};
        cf(out, in);

        REQUIRE(out[0] == approximately(230940.1435039826));
        REQUIRE(out[1] == approximately(230940.05975571528));
        REQUIRE(out[2] == approximately(230940.11976784476));
    }
}

TEST_CASE("rot_icrs_fk5j2000")
{
    const auto [x, y, z] = make_vars("x", "y", "z");
    const auto [icrs_x, icrs_y, icrs_z] = model::rot_icrs_fk5j2000({x, y, z});
    auto cf = cfunc<double>{{icrs_x, icrs_y, icrs_z}, {x, y, z}};

    {
        std::array<double, 3> in = {230940.1435039826, 230940.05975571528, 230940.11976784476}, out{};
        cf(out, in);

        REQUIRE(out[0] == approximately(230940.10767585033));
        REQUIRE(out[1] == approximately(230940.10767585033));
        REQUIRE(out[2] == approximately(230940.10767585033));
    }
}
