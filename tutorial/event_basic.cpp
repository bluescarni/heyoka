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

    // Create a vector to store the times
    // at which v goes to zero.
    std::vector<double> zero_vel_times;

    // Create the event object for the detection
    // of 'v == 0'.
    nt_event<double> ev(
        // The left-hand side of the event equation
        v,
        // The callback.
        [&zero_vel_times](taylor_adaptive<double> &ta, double time) {
            // Compute the state of the system when the
            // event triggered and print the value of x.
            ta.update_d_output(time);
            std::cout << "Value of x when v is zero: " << ta.get_d_output()[0] << '\n';

            // Add the event time to zero_vel_times.
            zero_vel_times.push_back(time);
        });

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = -9.8 * sin(x)
                                      {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {-0.05, 0.},
                                      // Non-terminal events.
                                      kw::nt_events = {ev}};

    // Enable full precision printing.
    std::cout.precision(16);

    // Propagate for a few time units.
    ta.propagate_until(5);

    // Print the event times.
    for (auto t : zero_vel_times) {
        std::cout << "Event detection time: " << t << '\n';
    }
}
