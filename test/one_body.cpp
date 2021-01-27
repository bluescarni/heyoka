// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("one_body")
{
    std::cout.precision(16);

    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    auto r_m3 = pow(x * x + y * y + z * z, -3_dbl / 2_dbl);

    auto [s_x, s_v] = kep_to_cart({1., .5, .0001, 0., 0., 0.}, 1.);

    taylor_adaptive<double> ta{{prime(x) = vx, prime(y) = vy, prime(z) = vz, prime(vx) = -x * r_m3,
                                prime(vy) = -y * r_m3, prime(vz) = -z * r_m3},
                               {s_x[0], s_x[1], s_x[2], s_v[0], s_v[1], s_v[2]},
                               kw::high_accuracy = true,
                               kw::tol = 1e-18};

    const auto &st = ta.get_state();

    while (true) {
        using std::abs;

        std::cout << st[0] << " " << st[1] << " " << st[2] << " " << st[3] << " " << st[4] << " " << st[5] << "\n";

        auto [res, _] = ta.step(2 * 3.141592653589793238460 - ta.get_time());

        REQUIRE((res == taylor_outcome::success || res == taylor_outcome::time_limit));

        if (abs(ta.get_time() - 2 * 3.141592653589793238460) < 1e-13) {
            std::cout << st[0] << " " << st[1] << " " << st[2] << " " << st[3] << " " << st[4] << " " << st[5] << "\n";

            break;
        }
    }
}
