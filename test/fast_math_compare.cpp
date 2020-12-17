// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Two-body problem energy from the state vector.
template <typename T>
T tbp_energy(const std::vector<T> &st)
{
    using std::sqrt;

    auto Dx = st[6] - st[7];
    auto Dy = st[8] - st[9];
    auto Dz = st[10] - st[11];
    auto dist = sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
    auto U = -1 / dist;

    auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
    auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];

    return T(1) / T(2) * (v2_0 + v2_1) + U;
}

TEST_CASE("two body fast math comparison")
{
    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    const auto kep = std::array{1.5, .2, .3, .4, .5, .6};
    const auto [c_x, c_v] = kep_to_cart(kep, 1. / 4);

    auto init_state = std::vector{c_v[0], -c_v[0], c_v[1], -c_v[1], c_v[2], -c_v[2],
                                  c_x[0], -c_x[0], c_x[1], -c_x[1], c_x[2], -c_x[2]};

    taylor_adaptive<double> ta_fm{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                   -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                  init_state};

    taylor_adaptive<double> ta_no_fm{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                      -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                     std::move(init_state),
                                     kw::fast_math = false};

    const auto en = tbp_energy(ta_fm.get_state());

    for (auto i = 0ul; i < 50000ul; ++i) {
        ta_fm.step();
        ta_no_fm.step();
    }

    std::cout << "Fast math:\n";
    std::cout << "Final time: " << ta_fm.get_time() << '\n';
    std::cout << "Relative error: " << std::abs((en - tbp_energy(ta_fm.get_state())) / en) << "\n\n";

    std::cout << "No fast math:\n";
    std::cout << "Final time: " << ta_no_fm.get_time() << '\n';
    std::cout << "Relative error: " << std::abs((en - tbp_energy(ta_no_fm.get_state())) / en) << "\n";
}
