// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <iostream>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

// A test to check for regressions in the
// expression simplification machinery.
TEST_CASE("n body")
{
    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    taylor_adaptive<double> ta{std::move(sys), std::vector<double>(36u), kw::compact_mode = true};

    REQUIRE(ta.get_decomposition().size() <= 384u);

    std::cout << "N-body: " << ta.get_decomposition().size() << '\n';
}

TEST_CASE("merc")
{
    auto [vx, vy, vz, x, y, z] = make_vars("vx", "vy", "vz", "x", "y", "z");

    auto mu = 0.01720209895 * 0.01720209895 * 365 * 365;
    auto eps = 2.5037803127808595e-10;

    auto v2 = vx * vx + vy * vy + vz * vz;
    auto r2 = x * x + y * y + z * z;
    auto r = sqrt(r2);

    auto Ham = 1. / 2 * v2 - mu / r + eps * (mu * mu / (2. * r2) - 1 / 8. * v2 * v2 - 3. / 2. * mu * v2 / r);

    auto ta = taylor_adaptive<double>{{prime(vx) = -diff(Ham, x), prime(vy) = -diff(Ham, y), prime(vz) = -diff(Ham, z),
                                       prime(x) = diff(Ham, vx), prime(y) = diff(Ham, vy), prime(z) = diff(Ham, vz)},
                                      std::vector<double>(6u)};

    REQUIRE(ta.get_decomposition().size() <= 113u);

    std::cout << "Mercury: " << ta.get_decomposition().size() << '\n';
}
