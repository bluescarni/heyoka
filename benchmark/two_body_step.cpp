// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <chrono>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main()
{
    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    const auto kep = std::array{1.5, .2, .3, .4, .5, .6};
    const auto [c_x, c_v] = kep_to_cart(kep, 1. / 4);

    std::vector init_state{c_v[0], -c_v[0], c_v[1], -c_v[1], c_v[2], -c_v[2],
                           c_x[0], -c_x[0], c_x[1], -c_x[1], c_x[2], -c_x[2]};

    taylor_adaptive<double> tad{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3,
                                 vx0, vx1, vy0, vy1, vz0, vz1},
                                std::move(init_state)};

    // Warm up.
    tad.step();

    auto start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < 4000; ++i) {
        tad.step();
    }

    const auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Elapsed time for a single timestep (double precision): " << elapsed / 4000 << "ns\n";

    return 0;
}
