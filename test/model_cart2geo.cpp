// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <stdexcept>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/cart2geo.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("impl")
{
    using Catch::Matchers::Message;

    const double a_earth = 6378137.0;
    const double b_earth = 6356752.314245;
    const double ecc2 = 1. - b_earth * b_earth / a_earth / a_earth;
    // First, we test malformed cases and their throws.
    // 1 - Size of the Cartesian state is wrong
    REQUIRE_THROWS_AS(
        model::detail::cart2geo_impl(std::vector<expression>{expression("x"), expression("y")}, ecc2, a_earth, 4u),
        std::invalid_argument);
    // 2 - Negative eccentricity
    REQUIRE_THROWS_AS(
        model::detail::cart2geo_impl(std::vector<expression>{expression("x"), expression("y"), expression("z")}, -2.3,
                                     a_earth, 4u),
        std::invalid_argument);
    // 3 - Negative planet equatorial radius
    REQUIRE_THROWS_AS(
        model::detail::cart2geo_impl(std::vector<expression>{expression("x"), expression("y"), expression("z")}, ecc2,
                                     -23233234.3423, 4u),
        std::invalid_argument);
    // 4 - No iterations
    REQUIRE_THROWS_AS(
        model::detail::cart2geo_impl(std::vector<expression>{expression("x"), expression("y"), expression("z")}, ecc2,
                                     a_earth, 0u),
        std::invalid_argument);

    // Then we test that for a few cases the numerical values are correct (approximately)
    // Init the symbolic variables.
    auto [x, y, z] = make_vars("x", "y", "z");
    auto geodetic_lph = model::cart2geo({x, y, z});
    cfunc<double> h_cf{{geodetic_lph[0]}, {x, y, z}};
    cfunc<double> phi_cf{{geodetic_lph[1]}, {x, y, z}};
    cfunc<double> lon_cf{{geodetic_lph[2]}, {x, y, z}};
    // Case 1
    {
        // Prepare the input-output buffers.
        std::array<double, 3> in{6000000, 6000000, 6000000};
        std::array<double, 1> out{};
        h_cf(out, in);
        REQUIRE(out[0] == approximately(4021307.660867557));
        phi_cf(out, in);
        REQUIRE(out[0] == approximately(0.6174213396277664));
        lon_cf(out, in);
        REQUIRE(out[0] == approximately(0.7853981633974483));
    }
    // Case 2
    {
        // Prepare the input-output buffers.
        std::array<double, 3> in{-2100.13123213, -1000.3764235678, 7324555.1224};
        std::array<double, 1> out{};
        h_cf(out, in);
        REQUIRE(out[0] == approximately(967803.1740983669));
        phi_cf(out, in);
        REQUIRE(out[0] == approximately(1.5704805814766054));
        lon_cf(out, in);
        REQUIRE(out[0] == approximately(-2.6970515993420663));
    }
    // And we repeat checking the n_iter and ecc2 values being not default
    // Case 1
    {
        geodetic_lph = model::cart2geo({x, y, z}, kw::ecc2 = 0.13, kw::R_eq = 60, kw::n_iters = 1);
        h_cf = cfunc<double>{{geodetic_lph[0]}, {x, y, z}};
        phi_cf = cfunc<double>{{geodetic_lph[1]}, {x, y, z}};
        lon_cf = cfunc<double>{{geodetic_lph[2]}, {x, y, z}};
        // Prepare the input-output buffers.
        std::array<double, 3> in{1., -1., 1.};
        std::array<double, 1> out{};
        h_cf(out, in);
        REQUIRE(out[0] == approximately(-59.791916138446254));
        phi_cf(out, in);
        REQUIRE(out[0] == approximately(-0.2053312550471871));
        lon_cf(out, in);
        REQUIRE(out[0] == approximately(-0.7853981633974483));
    }
    // Case 2
    {
        geodetic_lph = model::cart2geo(std::vector{x, y, z}, kw::ecc2 = 0.13, kw::R_eq = 60, kw::n_iters = 3);
        h_cf = cfunc<double>{{geodetic_lph[0]}, {x, y, z}};
        phi_cf = cfunc<double>{{geodetic_lph[1]}, {x, y, z}};
        lon_cf = cfunc<double>{{geodetic_lph[2]}, {x, y, z}};
        // Prepare the input-output buffers.
        std::array<double, 3> in{1., -1., 1.};
        std::array<double, 1> out{};
        h_cf(out, in);
        REQUIRE(out[0] == approximately(-58.66556686050428));
        phi_cf(out, in);
        REQUIRE(out[0] == approximately(-0.15741313107202423));
        lon_cf(out, in);
        REQUIRE(out[0] == approximately(-0.7853981633974483));
    }
}

TEST_CASE("igor_iface")
{
    auto [x, y, z] = make_vars("x", "y", "z");
    {
        auto igor_v = model::cart2geo({x, y, z}, kw::ecc2 = 12.12, kw::R_eq = 13.13, kw::n_iters = 3u);
        auto vanilla_v = model::detail::cart2geo_impl({x, y, z}, 12.12, 13.13, 3u);
        REQUIRE(igor_v == vanilla_v);
    }
}
