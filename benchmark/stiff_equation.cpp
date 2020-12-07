// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <heyoka/taylor.hpp>

using namespace heyoka;
using namespace std::chrono;

int main()
{
    auto [y1, y2] = make_vars("y1", "y2");
    for (unsigned i = 1u; i < 6; ++i) {
        // Van der pool stiff equation
        auto mu = expression{number{std::pow(10, i)}};
        taylor_adaptive<double> taylor{{prime(y1) = y2, prime(y2) = mu * (1_dbl - y1 * y1) * y2 - y1}, {2., 0.}};
        std::cout << "Stiffness param " << mu << ": ";

        auto start = high_resolution_clock::now();
        taylor.propagate_for(3000);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        std::cout << duration.count() / 1e6 << "s" << std::endl;
    }

    return 0;
}
