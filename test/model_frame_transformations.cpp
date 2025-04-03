// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>

#include <heyoka/detail/debug.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
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

// NOTE: in this test we are checking the rotation against the values provided here:
//
// https://hpiers.obspm.fr/eop-pc/index.php?index=rotation&lang=en
//
// We are checking that our implementation is correct around the cm level.
TEST_CASE("rot_itrs_icrs")
{
    heyoka::detail::edb_disabler ed;

    // NOTE: the online iers service uses the long-term data in the computation.
    const auto data = eop_data::fetch_latest_iers_long_term();

    const auto [x, y, z] = make_vars("x", "y", "z");
    const auto [icrs_x, icrs_y, icrs_z] = model::rot_itrs_icrs({x, y, z}, kw::thresh = 0., kw::eop_data = data);
    auto cf = cfunc<double>{{icrs_x, icrs_y, icrs_z}, {x, y, z}, kw::compact_mode = true};

    // Allow for an error of 2 cm.
    const auto tol = 2 * 1e-5;

    {
        std::array<double, 3> in = {3682.340016891433, 3682.340016891433, 3682.340016891433}, out{};
        cf(out, in, kw::time = 2.03386822825559e-08);

        REQUIRE(std::abs(out[0] - 4289.68738577) < tol);
        REQUIRE(std::abs(out[1] - -2952.56494217) < tol);
        REQUIRE(std::abs(out[2] - 3682.36695545) < tol);
    }

    {
        std::array<double, 3> in = {3682.340016891433 + 500, 3682.340016891433 + 250, 3682.340016891433}, out{};
        cf(out, in, kw::time = 0.050020554219585754);

        REQUIRE(std::abs(out[0] - 4669.92059844) < tol);
        REQUIRE(std::abs(out[1] - -3341.02241247) < tol);
        REQUIRE(std::abs(out[2] - 3680.25880463) < tol);
    }

    {
        std::array<double, 3> in = {-3682.340016891433 + 500, 3682.340016891433 + 250, -3682.340016891433}, out{};
        cf(out, in, kw::time = 0.1545117712322863);

        REQUIRE(std::abs(out[0] - 3458.39589769) < tol);
        REQUIRE(std::abs(out[1] - 3686.87030805) < tol);
        REQUIRE(std::abs(out[2] - -3687.37021982) < tol);
    }

    {
        std::array<double, 3> in = {3682.340016891433 + 500, -3682.340016891433 + 250, -3682.340016891433}, out{};
        cf(out, in, kw::time = 0.12111797671052296);

        REQUIRE(std::abs(out[0] - -4071.10690183) < tol);
        REQUIRE(std::abs(out[1] - -3568.60159061) < tol);
        REQUIRE(std::abs(out[2] - -3677.46179992) < tol);
    }

    {
        std::array<double, 3> in = {26, -6700, -123}, out{};
        cf(out, in, kw::time = -0.19888202804269012);

        REQUIRE(std::abs(out[0] - -6617.91806732) < tol);
        REQUIRE(std::abs(out[1] - 1044.27686493) < tol);
        REQUIRE(std::abs(out[2] - -135.83550179) < tol);
    }

    {
        std::array<double, 3> in = {6100, 15, 823}, out{};
        cf(out, in, kw::time = -0.05223132900347306);

        REQUIRE(std::abs(out[0] - 2116.13216603) < tol);
        REQUIRE(std::abs(out[1] - -5721.0838992) < tol);
        REQUIRE(std::abs(out[2] - 823.8553721) < tol);
    }
}
