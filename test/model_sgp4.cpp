// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/sgp4.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic")
{
    detail::edb_disabler ed;

    const auto revday2radmin = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 1440.; };
    const auto deg2rad = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 360.; };

    const auto [inputs, outputs] = model::sgp4();

    auto sgp4_cf = cfunc<double>(outputs, inputs);

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
