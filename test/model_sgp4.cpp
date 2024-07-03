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

        REQUIRE(outs[0] == approximately(3469.9479831138));
        REQUIRE(outs[1] == approximately(-2690.3884293306));
        REQUIRE(outs[2] == approximately(5175.8319226575));
        REQUIRE(outs[3] == approximately(5.810229140624679));
        REQUIRE(outs[4] == approximately(4.802261183356792));
        REQUIRE(outs[5] == approximately(-1.3882803326600321));
    }

    {
        std::vector<double> ins = {revday2radmin(15.50103472202482),
                                   0.0007417,
                                   deg2rad(51.6439),
                                   deg2rad(211.2001),
                                   deg2rad(17.6667),
                                   deg2rad(85.6398),
                                   .38792e-4,
                                   120.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(4086.3396316729295));
        REQUIRE(outs[1] == approximately(4796.928067515364));
        REQUIRE(outs[2] == approximately(-2562.5856280208386));
        REQUIRE(outs[3] == approximately(-5.287831886860389));
        REQUIRE(outs[4] == approximately(1.6959008859460454));
        REQUIRE(outs[5] == approximately(-5.265040553027974));
    }

    {
        std::vector<double> ins = {revday2radmin(16.05824518), 0.0086731,         deg2rad(72.8435), deg2rad(115.9689),
                                   deg2rad(52.6988),           deg2rad(110.5714), .66816e-4,        0.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(2328.969751931129));
        REQUIRE(outs[1] == approximately(-5995.220511600548));
        REQUIRE(outs[2] == approximately(1719.9729714023497));
        REQUIRE(outs[3] == approximately(2.912073280385506));
        REQUIRE(outs[4] == approximately(-0.9834179555026153));
        REQUIRE(outs[5] == approximately(-7.090816207952735));
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

        REQUIRE(outs[0] == approximately(2561.2236598772));
        REQUIRE(outs[1] == approximately(3698.7971429615));
        REQUIRE(outs[2] == approximately(5818.7722139807));
        REQUIRE(outs[3] == approximately(-3.2761425126425463));
        REQUIRE(outs[4] == approximately(-4.806489081397908));
        REQUIRE(outs[5] == approximately(4.511134500293972));
    }

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738),
                                   0.75863e-3,
                                   70.},
                            outs(6u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(1562.7283773281774));
        REQUIRE(outs[1] == approximately(2322.0639519617216));
        REQUIRE(outs[2] == approximately(-6796.866223211695));
        REQUIRE(outs[3] == approximately(3.8565580810018645));
        REQUIRE(outs[4] == approximately(5.607159620567683));
        REQUIRE(outs[5] == approximately(2.8124612700451466));
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

        REQUIRE(outs[0] == approximately(3134.2015610758267));
        REQUIRE(outs[1] == approximately(4604.9636618367485));
        REQUIRE(outs[2] == approximately(-4791.661125321369));
        REQUIRE(outs[3] == approximately(2.7320346123307453));
        REQUIRE(outs[4] == approximately(3.9525897763872484));
        REQUIRE(outs[5] == approximately(5.588906719554344));
    }
}

TEST_CASE("propagator")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 7, std::dynamic_extent>>;

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
                               .38792e-4

    };

    prop_t prop{md_input_t{ins.data(), 7, 2}};

    auto tm = std::array{1440., 120.};
    prop_t::in_1d tm_in{tm.data(), 2};

    std::vector<double> outs(12u);
    prop_t::out_2d out{outs.data(), 6, 2};

    prop(out, tm_in);

    REQUIRE(out(0, 0) == approximately(3134.2015610758267));
    REQUIRE(out(1, 0) == approximately(4604.9636618367485));
    REQUIRE(out(2, 0) == approximately(-4791.661125321369));
    REQUIRE(out(3, 0) == approximately(2.7320346123307453));
    REQUIRE(out(4, 0) == approximately(3.9525897763872484));
    REQUIRE(out(5, 0) == approximately(5.588906719554344));
    REQUIRE(out(0, 1) == approximately(4086.3396316729295));
    REQUIRE(out(1, 1) == approximately(4796.928067515364));
    REQUIRE(out(2, 1) == approximately(-2562.5856280208386));
    REQUIRE(out(3, 1) == approximately(-5.287831886860389));
    REQUIRE(out(4, 1) == approximately(1.6959008859460454));
    REQUIRE(out(5, 1) == approximately(-5.265040553027974));
}

TEST_CASE("propagator batch")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 7, std::dynamic_extent>>;

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
                               .38792e-4

    };

    prop_t prop{md_input_t{ins.data(), 7, 2}};

    auto tm = std::array{70., 0., 1440., 120.};
    prop_t::in_2d tm_in{tm.data(), 2, 2};

    std::vector<double> outs(24u);
    prop_t::out_3d out{outs.data(), 2, 6, 2};

    prop(out, tm_in);

    REQUIRE(out(0, 0, 0) == approximately(1562.7283773281774));
    REQUIRE(out(0, 1, 0) == approximately(2322.0639519617216));
    REQUIRE(out(0, 2, 0) == approximately(-6796.866223211695));
    REQUIRE(out(0, 3, 0) == approximately(3.8565580810018645));
    REQUIRE(out(0, 4, 0) == approximately(5.607159620567683));
    REQUIRE(out(0, 5, 0) == approximately(2.8124612700451466));
    REQUIRE(out(0, 0, 1) == approximately(3469.9479831138));
    REQUIRE(out(0, 1, 1) == approximately(-2690.3884293306));
    REQUIRE(out(0, 2, 1) == approximately(5175.8319226575));
    REQUIRE(out(0, 3, 1) == approximately(5.810229140624679));
    REQUIRE(out(0, 4, 1) == approximately(4.802261183356792));
    REQUIRE(out(0, 5, 1) == approximately(-1.3882803326600321));
    REQUIRE(out(1, 0, 0) == approximately(3134.2015610758267));
    REQUIRE(out(1, 1, 0) == approximately(4604.9636618367485));
    REQUIRE(out(1, 2, 0) == approximately(-4791.661125321369));
    REQUIRE(out(1, 3, 0) == approximately(2.7320346123307453));
    REQUIRE(out(1, 4, 0) == approximately(3.9525897763872484));
    REQUIRE(out(1, 5, 0) == approximately(5.588906719554344));
    REQUIRE(out(1, 0, 1) == approximately(4086.3396316729295));
    REQUIRE(out(1, 1, 1) == approximately(4796.928067515364));
    REQUIRE(out(1, 2, 1) == approximately(-2562.5856280208386));
    REQUIRE(out(1, 3, 1) == approximately(-5.287831886860389));
    REQUIRE(out(1, 4, 1) == approximately(1.6959008859460454));
    REQUIRE(out(1, 5, 1) == approximately(-5.265040553027974));
}
