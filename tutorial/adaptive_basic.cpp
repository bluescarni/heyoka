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

    // Let's print to screen the integrator object.
    std::cout << ta << '\n';

    // Perform a single step.
    auto [oc, h] = ta.step();

    // Print the outcome flag and the timestep used.
    std::cout << "Outcome : " << oc << '\n';
    std::cout << "Timestep: " << h << "\n\n";

    // Print again the integrator object to screen.
    std::cout << ta << '\n';

    // Perform a step backward.
    std::tie(oc, h) = ta.step_backward();

    // Print the outcome flag and the timestep used.
    std::cout << "Outcome : " << oc << '\n';
    std::cout << "Timestep: " << h << "\n\n";

    // Perform a step forward in time, clamping
    // the timestep size to 0.01.
    std::tie(oc, h) = ta.step(0.01);

    // Print the outcome flag and the timestep used.
    std::cout << "Outcome : " << oc << '\n';
    std::cout << "Timestep: " << h << "\n\n";

    // Perform a step backward in time, clamping
    // the timestep size to 0.02.
    std::tie(oc, h) = ta.step(-0.02);

    // Print the outcome flag and the timestep used.
    std::cout << "Outcome : " << oc << '\n';
    std::cout << "Timestep: " << h << "\n\n";

    // Print the current time.
    std::cout << "Current time: " << ta.get_time() << '\n';

    // Print out the current value of the x variable.
    std::cout << "Current x value: " << ta.get_state()[0] << "\n\n";

    // Reset the time and state to the initial values.
    ta.set_time(0.);
    ta.get_state_data()[0] = 0.05;
    ta.get_state_data()[1] = 0.025;

    // Propagate for 5 time units.
    auto [status, min_h, max_h, nsteps] = ta.propagate_for(5.);

    std::cout << "Outcome      : " << status << '\n';
    std::cout << "Min. timestep: " << min_h << '\n';
    std::cout << "Max. timestep: " << max_h << '\n';
    std::cout << "Num. of steps: " << nsteps << '\n';
    std::cout << "Current time : " << ta.get_time() << "\n\n";

    // Propagate until t = 20.
    std::tie(status, min_h, max_h, nsteps) = ta.propagate_until(20.);

    std::cout << "Outcome      : " << status << '\n';
    std::cout << "Min. timestep: " << min_h << '\n';
    std::cout << "Max. timestep: " << max_h << '\n';
    std::cout << "Num. of steps: " << nsteps << '\n';
    std::cout << "Current time : " << ta.get_time() << "\n\n";

    // Propagate back to t = 0.
    std::tie(status, min_h, max_h, nsteps) = ta.propagate_until(0.);

    std::cout << "Outcome      : " << status << '\n';
    std::cout << "Min. timestep: " << min_h << '\n';
    std::cout << "Max. timestep: " << max_h << '\n';
    std::cout << "Num. of steps: " << nsteps << '\n';
    std::cout << "Current time : " << ta.get_time() << "\n\n";

    std::cout << ta << '\n';
}
