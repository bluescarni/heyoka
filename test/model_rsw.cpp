// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>

#include <heyoka/expression.hpp>
#include <heyoka/model/frame_transformations.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("to_rsw")
{
    const auto [pos_x, pos_y, pos_z] = make_vars("pos_x", "pos_y", "pos_z");
    const auto [vel_x, vel_y, vel_z] = make_vars("vel_x", "vel_y", "vel_z");
    const auto [x, y, z] = make_vars("x", "y", "z");
    const auto [vx, vy, vz] = make_vars("vx", "vy", "vz");

    const auto [pos_p, vel_p]
        = model::state_to_rsw({pos_x, pos_y, pos_z}, {vel_x, vel_y, vel_z}, {x, y, z}, {vx, vy, vz});

    const auto cf = cfunc<double>({pos_p[0], pos_p[1], pos_p[2], vel_p[0], vel_p[1], vel_p[2]},
                                  {pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, x, y, z, vx, vy, vz});

    std::array<double, 6> output{};

    // Simple test: transform the r/v defining the RSW frame (should result in all zeroes in the result).
    {
        const std::array<double, 12> input{1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0};

        cf(output, input);

        REQUIRE(std::ranges::all_of(output, [](const auto x) { return std::abs(x) < 1e-15; }));
    }

    // Simple test: transform the r defining the RSW frame and the negative v.
    {
        const std::array<double, 12> input{1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0};

        cf(output, input);

        REQUIRE(std::ranges::all_of(std::ranges::subrange(output.data(), output.data() + 3),
                                    [](const auto x) { return std::abs(x) < 1e-15; }));
        REQUIRE(std::abs(output[3]) < 1e-15);
        REQUIRE(std::abs(output[4] + 2) < 1e-15);
        REQUIRE(std::abs(output[5]) < 1e-15);
    }

    // Some tests comparing the results with the orekit library (see notebook in the 'tools' dir).
    {
        const std::array<double, 12> input{1.9901598679710650e+06,  -4.3171886347018217e+05, 6.7720686840924025e+06,
                                           -6.9420492298495501e+03, -2.1395918103707527e+03, 1.9120431913109790e+03,
                                           1.9990152502378467e+06,  -4.2466313738494366e+05, 6.7714722017919971e+06,
                                           -6.9397802817958964e+03, -2.1318724003511638e+03, 1.9205549571233923e+03};

        cf(output, input);

        const std::array state_orekit = {-1508.0546801836172, 10340.099985439594, 4400.563621644753,
                                         2.6516116955953297,  3.7183675363633046, 7.9607554111462075};

        for (auto i = 0u; i < 6u; ++i) {
            REQUIRE(std::abs((state_orekit[i] - output[i]) / state_orekit[i]) < 1e-12);
        }
    }
}

TEST_CASE("to_rsw_inertial")
{
    const auto [pos_x, pos_y, pos_z] = make_vars("pos_x", "pos_y", "pos_z");
    const auto [vel_x, vel_y, vel_z] = make_vars("vel_x", "vel_y", "vel_z");
    const auto [x, y, z] = make_vars("x", "y", "z");
    const auto [vx, vy, vz] = make_vars("vx", "vy", "vz");

    const auto [pos_p, vel_p]
        = model::state_to_rsw_inertial({pos_x, pos_y, pos_z}, {vel_x, vel_y, vel_z}, {x, y, z}, {vx, vy, vz});

    const auto cf = cfunc<double>({pos_p[0], pos_p[1], pos_p[2], vel_p[0], vel_p[1], vel_p[2]},
                                  {pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, x, y, z, vx, vy, vz});

    std::array<double, 6> output{};

    // Some tests comparing the results with the orekit library (see notebook in the 'tools' dir).
    {
        const std::array<double, 12> input{1.9901598679710650e+06,  -4.3171886347018217e+05, 6.7720686840924025e+06,
                                           -6.9420492298495501e+03, -2.1395918103707527e+03, 1.9120431913109790e+03,
                                           1.9990152502378467e+06,  -4.2466313738494366e+05, 6.7714722017919971e+06,
                                           -6.9397802817958964e+03, -2.1318724003511638e+03, 1.9205549571233923e+03};

        cf(output, input);

        const std::array state_orekit = {-1508.0546801836172, 10340.099985439594, 4400.563621644753,
                                         -3.0126937116365298, 7511.706686243073,  7.96075541114692};

        for (auto i = 0u; i < 6u; ++i) {
            REQUIRE(std::abs((state_orekit[i] - output[i]) / state_orekit[i]) < 1e-12);
        }
    }
}
