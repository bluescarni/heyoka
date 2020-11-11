// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

#include "data/mascon_itokawa.hpp"

// This benchmark builds a Taylor integrator for the motion around asteroid Itokawa.
// The mascon model for Itokawa was generated using a thetraedral mesh built upon
// the polyhedral surface model available

using namespace heyoka;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    auto dim = 5497u;

    // assembling the r.h.s.
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Contributions to the x/y/z accelerations from each mass.
    std::vector<expression> x_acc, y_acc, z_acc;

    for (decltype(dim) i = 0; i < dim; ++i) {
        auto x_masc = expression{number{mascon_points_itokawa[i][0]}};
        auto y_masc = expression{number{mascon_points_itokawa[i][1]}};
        auto z_masc = expression{number{mascon_points_itokawa[i][2]}};
        auto m_masc = expression{number{mascon_masses_itokawa[i]}};
        auto r2 = (x - x_masc) * (x - x_masc) + (y - y_masc) * (y - y_masc) + (z - z_masc) * (z - z_masc);

        x_acc.push_back(m_masc * (x_masc - x) * pow(r2, expression{number{-3. / 2}}));
        y_acc.push_back(m_masc * (y_masc - y) * pow(r2, expression{number{-3. / 2}}));
        z_acc.push_back(m_masc * (z_masc - z) * pow(r2, expression{number{-3. / 2}}));
    }

    auto start = high_resolution_clock::now();
    // When accumulating sums in the r.h.s. use pairwise summation to let llvm be way faster.
    taylor_adaptive<double> taylor{{prime(x) = vx, prime(y) = vy, prime(z) = vz, prime(vx) = pairwise_sum(x_acc),
                                    prime(vy) = pairwise_sum(y_acc), prime(vz) = pairwise_sum(z_acc)},
                                   {1., 0., 0., 0., 1., 0.},
                                   kw::compact_mode = true};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Duration (5497 masses): " << duration.count() / 1e6 << "s" << std::endl;

    double dt = 0.1;
    // for (auto i = 0u; i< 1000; ++i){
    taylor.propagate_for(dt);
    //auto state = taylor.get_state();
    // fmt::print("[{}, {}, {}, {}, {}, {}]\n", state[0], state[1], state[2], state[3], state[4], state[5]);
    //}
    return 0;
}
