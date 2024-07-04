// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/sgp4.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto revday2radmin = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 1440.; };
const auto deg2rad = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 360.; };

TEST_CASE("basic")
{
    detail::edb_disabler ed;

    const auto outputs = model::sgp4();
    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");

    auto sgp4_cf = cfunc<double>(outputs, inputs);

    {
        std::vector<double> ins = {revday2radmin(15.50103472202482),
                                   0.0007417,
                                   deg2rad(51.6439),
                                   deg2rad(211.2001),
                                   deg2rad(17.6667),
                                   deg2rad(85.6398),
                                   .38792e-4,
                                   0.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(3469.947984145807, 10000.));
        REQUIRE(outs[1] == approximately(-2690.388430131083, 10000.));
        REQUIRE(outs[2] == approximately(5175.831924199492, 10000.));
        REQUIRE(outs[3] == approximately(5.810229142351453, 10000.));
        REQUIRE(outs[4] == approximately(4.802261184784617, 10000.));
        REQUIRE(outs[5] == approximately(-1.388280333072693, 10000.));
    }

    {
        std::vector<double> ins = {revday2radmin(15.50103472202482),
                                   0.0007417,
                                   deg2rad(51.6439),
                                   deg2rad(211.2001),
                                   deg2rad(17.6667),
                                   deg2rad(85.6398),
                                   .38792e-4,
                                   1440.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(-3591.82683131782, 10000.));
        REQUIRE(outs[1] == approximately(2723.666407193435, 10000.));
        REQUIRE(outs[2] == approximately(-5090.448264983512, 10000.));
        REQUIRE(outs[3] == approximately(-5.927709516654264, 10000.));
        REQUIRE(outs[4] == approximately(-4.496384419253211, 10000.));
        REQUIRE(outs[5] == approximately(1.785277174529374, 10000.));
    }

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738),
                                   0.75863e-3,
                                   0.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(2561.223660636298, 10000.));
        REQUIRE(outs[1] == approximately(3698.797144057697, 10000.));
        REQUIRE(outs[2] == approximately(5818.772215708888, 10000.));
        REQUIRE(outs[3] == approximately(-3.276142513618007, 10000.));
        REQUIRE(outs[4] == approximately(-4.806489082829041, 10000.));
        REQUIRE(outs[5] == approximately(4.511134501638151, 10000.));
    }

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738),
                                   0.75863e-3,
                                   1440.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(3134.2015620939, 10000.));
        REQUIRE(outs[1] == approximately(4604.963663328277, 10000.));
        REQUIRE(outs[2] == approximately(-4791.661126560278, 10000.));
        REQUIRE(outs[3] == approximately(2.732034613044249, 10000.));
        REQUIRE(outs[4] == approximately(3.952589777415254, 10000.));
        REQUIRE(outs[5] == approximately(5.588906721377138, 10000.));
    }
}

TEST_CASE("propagator")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    std::vector<double> ins = {revday2radmin(13.75091047972192),
                               revday2radmin(15.50103472202482),
                               0.0024963,
                               0.0007417,
                               deg2rad(90.2039),
                               deg2rad(51.6439),
                               deg2rad(55.5633),
                               deg2rad(211.2001),
                               deg2rad(320.5956),
                               deg2rad(17.6667),
                               deg2rad(91.4738),
                               deg2rad(85.6398),
                               0.75863e-3,
                               .38792e-4,
                               2460486.5,
                               2458826.5,
                               0.6478633000000116,
                               0.6933954099999937};

    prop_t prop{md_input_t{ins.data(), 9, 2}};

    auto tm = std::array{1440., 0.};
    prop_t::in_1d<double> tm_in{tm.data(), 2};

    std::vector<double> outs(12u);
    prop_t::out_2d out{outs.data(), 6, 2};

    prop(out, tm_in);

    REQUIRE(out(0, 0) == approximately(3134.2015620939, 10000.));
    REQUIRE(out(1, 0) == approximately(4604.963663328277, 10000.));
    REQUIRE(out(2, 0) == approximately(-4791.661126560278, 10000.));
    REQUIRE(out(3, 0) == approximately(2.732034613044249, 10000.));
    REQUIRE(out(4, 0) == approximately(3.952589777415254, 10000.));
    REQUIRE(out(5, 0) == approximately(5.588906721377138, 10000.));
    REQUIRE(out(0, 1) == approximately(3469.947984145807, 10000.));
    REQUIRE(out(1, 1) == approximately(-2690.388430131083, 10000.));
    REQUIRE(out(2, 1) == approximately(5175.831924199492, 10000.));
    REQUIRE(out(3, 1) == approximately(5.810229142351453, 10000.));
    REQUIRE(out(4, 1) == approximately(4.802261184784617, 10000.));
    REQUIRE(out(5, 1) == approximately(-1.388280333072693, 10000.));

    auto dates = std::array<prop_t::date, 2>{{{2460486.5 + 1, 0.6478633000000116}, {2458826.5, 0.6933954099999937}}};
    prop_t::in_1d<prop_t::date> date_in{dates.data(), 2};

    prop(out, date_in);

    REQUIRE(out(0, 0) == approximately(3134.2015620939, 10000.));
    REQUIRE(out(1, 0) == approximately(4604.963663328277, 10000.));
    REQUIRE(out(2, 0) == approximately(-4791.661126560278, 10000.));
    REQUIRE(out(3, 0) == approximately(2.732034613044249, 10000.));
    REQUIRE(out(4, 0) == approximately(3.952589777415254, 10000.));
    REQUIRE(out(5, 0) == approximately(5.588906721377138, 10000.));
    REQUIRE(out(0, 1) == approximately(3469.947984145807, 10000.));
    REQUIRE(out(1, 1) == approximately(-2690.388430131083, 10000.));
    REQUIRE(out(2, 1) == approximately(5175.831924199492, 10000.));
    REQUIRE(out(3, 1) == approximately(5.810229142351453, 10000.));
    REQUIRE(out(4, 1) == approximately(4.802261184784617, 10000.));
    REQUIRE(out(5, 1) == approximately(-1.388280333072693, 10000.));
}

TEST_CASE("propagator batch")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    std::vector<double> ins = {revday2radmin(13.75091047972192),
                               revday2radmin(15.50103472202482),
                               0.0024963,
                               0.0007417,
                               deg2rad(90.2039),
                               deg2rad(51.6439),
                               deg2rad(55.5633),
                               deg2rad(211.2001),
                               deg2rad(320.5956),
                               deg2rad(17.6667),
                               deg2rad(91.4738),
                               deg2rad(85.6398),
                               0.75863e-3,
                               .38792e-4,
                               2460486.5,
                               2458826.5,
                               0.6478633000000116,
                               0.6933954099999937};

    prop_t prop{md_input_t{ins.data(), 9, 2}};

    auto tm = std::array{1440., 0., 0., 1440.};
    prop_t::in_2d<double> tm_in{tm.data(), 2, 2};

    std::vector<double> outs(24u);
    prop_t::out_3d out{outs.data(), 2, 6, 2};

    prop(out, tm_in);

    REQUIRE(out(0, 0, 0) == approximately(3134.2015620939, 10000.));
    REQUIRE(out(0, 1, 0) == approximately(4604.963663328277, 10000.));
    REQUIRE(out(0, 2, 0) == approximately(-4791.661126560278, 10000.));
    REQUIRE(out(0, 3, 0) == approximately(2.732034613044249, 10000.));
    REQUIRE(out(0, 4, 0) == approximately(3.952589777415254, 10000.));
    REQUIRE(out(0, 5, 0) == approximately(5.588906721377138, 10000.));
    REQUIRE(out(0, 0, 1) == approximately(3469.947984145807, 10000.));
    REQUIRE(out(0, 1, 1) == approximately(-2690.388430131083, 10000.));
    REQUIRE(out(0, 2, 1) == approximately(5175.831924199492, 10000.));
    REQUIRE(out(0, 3, 1) == approximately(5.810229142351453, 10000.));
    REQUIRE(out(0, 4, 1) == approximately(4.802261184784617, 10000.));
    REQUIRE(out(0, 5, 1) == approximately(-1.388280333072693, 10000.));
    REQUIRE(out(1, 0, 0) == approximately(2561.223660636298, 10000.));
    REQUIRE(out(1, 1, 0) == approximately(3698.797144057697, 10000.));
    REQUIRE(out(1, 2, 0) == approximately(5818.772215708888, 10000.));
    REQUIRE(out(1, 3, 0) == approximately(-3.276142513618007, 10000.));
    REQUIRE(out(1, 4, 0) == approximately(-4.806489082829041, 10000.));
    REQUIRE(out(1, 5, 0) == approximately(4.511134501638151, 10000.));
    REQUIRE(out(1, 0, 1) == approximately(-3591.82683131782, 10000.));
    REQUIRE(out(1, 1, 1) == approximately(2723.666407193435, 10000.));
    REQUIRE(out(1, 2, 1) == approximately(-5090.448264983512, 10000.));
    REQUIRE(out(1, 3, 1) == approximately(-5.927709516654264, 10000.));
    REQUIRE(out(1, 4, 1) == approximately(-4.496384419253211, 10000.));
    REQUIRE(out(1, 5, 1) == approximately(1.785277174529374, 10000.));

    auto dates = std::array<prop_t::date, 4>{{{2460486.5 + 1, 0.6478633000000116},
                                              {2458826.5, 0.6933954099999937},
                                              {2460486.5, 0.6478633000000116},
                                              {2458826.5 + 1, 0.6933954099999937}}};
    prop_t::in_2d<prop_t::date> date_in{dates.data(), 2, 2};

    prop(out, date_in);

    REQUIRE(out(0, 0, 0) == approximately(3134.2015620939, 10000.));
    REQUIRE(out(0, 1, 0) == approximately(4604.963663328277, 10000.));
    REQUIRE(out(0, 2, 0) == approximately(-4791.661126560278, 10000.));
    REQUIRE(out(0, 3, 0) == approximately(2.732034613044249, 10000.));
    REQUIRE(out(0, 4, 0) == approximately(3.952589777415254, 10000.));
    REQUIRE(out(0, 5, 0) == approximately(5.588906721377138, 10000.));
    REQUIRE(out(0, 0, 1) == approximately(3469.947984145807, 10000.));
    REQUIRE(out(0, 1, 1) == approximately(-2690.388430131083, 10000.));
    REQUIRE(out(0, 2, 1) == approximately(5175.831924199492, 10000.));
    REQUIRE(out(0, 3, 1) == approximately(5.810229142351453, 10000.));
    REQUIRE(out(0, 4, 1) == approximately(4.802261184784617, 10000.));
    REQUIRE(out(0, 5, 1) == approximately(-1.388280333072693, 10000.));
    REQUIRE(out(1, 0, 0) == approximately(2561.223660636298, 10000.));
    REQUIRE(out(1, 1, 0) == approximately(3698.797144057697, 10000.));
    REQUIRE(out(1, 2, 0) == approximately(5818.772215708888, 10000.));
    REQUIRE(out(1, 3, 0) == approximately(-3.276142513618007, 10000.));
    REQUIRE(out(1, 4, 0) == approximately(-4.806489082829041, 10000.));
    REQUIRE(out(1, 5, 0) == approximately(4.511134501638151, 10000.));
    REQUIRE(out(1, 0, 1) == approximately(-3591.82683131782, 10000.));
    REQUIRE(out(1, 1, 1) == approximately(2723.666407193435, 10000.));
    REQUIRE(out(1, 2, 1) == approximately(-5090.448264983512, 10000.));
    REQUIRE(out(1, 3, 1) == approximately(-5.927709516654264, 10000.));
    REQUIRE(out(1, 4, 1) == approximately(-4.496384419253211, 10000.));
    REQUIRE(out(1, 5, 1) == approximately(1.785277174529374, 10000.));
}
