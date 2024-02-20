// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
    ta.step(true);

    // Fetch the Taylor coefficients of order 0
    // for x and v.
    std::cout << "TC of order 0 for x: " << ta.get_tc()[0] << '\n';
    std::cout << "TC of order 0 for v: " << ta.get_tc()[ta.get_order() + 1u] << '\n';

    // Compute and print the dense output at t = 0.1.
    auto &d_out = ta.update_d_output(0.1);
    std::cout << "x(0.1) = " << d_out[0] << '\n';
    std::cout << "y(0.1) = " << d_out[1] << '\n';

    // Compute the dense output at the end of the
    // previous timestep and compare it to the current
    // state vector.
    ta.update_d_output(ta.get_time());
    const auto &st = ta.get_state();
    std::cout << "x rel. difference: " << (d_out[0] - st[0]) / st[0] << '\n';
    std::cout << "v rel. difference: " << (d_out[1] - st[1]) / st[1] << '\n';

    // Reset time and state.
    ta.get_state_data()[0] = 0.05;
    ta.get_state_data()[1] = 0.025;
    ta.set_time(0);

    // Integrate up to t = 10, and request continuous output.
    auto c_out = std::get<4>(ta.propagate_until(10., kw::c_output = true));

    // Print to screen.
    std::cout << *c_out << '\n';

    // Compute and print to screen the system state
    // at different time coordinates.
    for (auto tm : {0., 1.5, 4.3, 6.7, 8.9, 10.}) {
        // Compute the state at tm.
        (*c_out)(tm);

        // Print it out:
        std::cout << "time=" << tm << ", x=" << c_out->get_output()[0] //
                  << ", v=" << c_out->get_output()[1] << '\n';
    }
}
