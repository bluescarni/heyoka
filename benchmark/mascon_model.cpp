// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

// This benchmark builds an equation where the r.h.s. is the sum of N terms all depending on the state
// It can stress test the Taylor integrator for very long expressions \dot r = \sum_i f_i(r) where i goes to 10000
//
// As an example we write the equation for N fixed masses producing a gravitational field affecting on mass

using namespace heyoka;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    // assembling the r.h.s.
    for (double N = 3u; N < 1000; N += 10) {
        auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

        // Contributions to the x/y/z accelerations from each mass.
        std::vector<expression> x_acc, y_acc, z_acc;

        for (double i = -N; i < N; ++i) {
            auto xpos = expression{number(i)};
            auto r2 = (x - xpos) * (x - xpos) + y * y + z * z;

            x_acc.push_back((xpos - x) * pow(r2, expression{number{-3. / 2}}));
            y_acc.push_back(-y * pow(r2, expression{number{-3. / 2}}));
            z_acc.push_back(-z * pow(r2, expression{number{-3. / 2}}));
        }

        auto start = high_resolution_clock::now();
        taylor_adaptive<double> taylor{{prime(x) = vx, prime(y) = vy, prime(z) = vz, prime(vx) = pairwise_sum(x_acc),
                                        prime(vy) = pairwise_sum(y_acc), prime(vz) = pairwise_sum(z_acc)},
                                       {0.123, 0.123, 0.123, 0., 0., 0.},
                                       kw::compact_mode = true};
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << N << ": " << duration.count() / 1e6 << "s" << std::endl;

        // taylor.propagate_for(3000);
    }

    return 0;
}
