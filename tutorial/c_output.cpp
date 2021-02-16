// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = -9.8 * sin(x)
                                      {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0.05, 0.025}};

    // Integrate for a single timestep, and store
    // the Taylor coefficients in the integrator.
    auto [_, h] = ta.step(true);
    std::cout << "Timestep size: " << h << '\n';

    // Fetch the Taylor coefficients of order 0
    // for x and v.
    std::cout << "TC of order 0 for x: " << ta.get_tc()[0] << '\n';
    std::cout << "TC of order 0 for v: " << ta.get_tc()[ta.get_order() + 1u] << '\n';

    // Compute and print the continuous output at t = 0.1.
    auto &co = ta.update_c_output(0.1);
    std::cout << "x(0.1) = " << co[0] << '\n';
    std::cout << "y(0.1) = " << co[1] << '\n';

    // Compute the continuous output at the end of the
    // previous timestep and compare it to the current
    // state vector.
    ta.update_c_output(h);
    const auto &st = ta.get_state();
    std::cout << "x rel. difference: " << (co[0] - st[0]) / st[0] << '\n';
    std::cout << "v rel. difference: " << (co[1] - st[1]) / st[1] << '\n';
}
