// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
        [&zero_vel_times](taylor_adaptive<double> &ta, double time, int) {
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

    // Redefine ev to detect only events
    // in the positive direction.
    ev = nt_event<double>(
        v, [&zero_vel_times](taylor_adaptive<double> &, double time, int) { zero_vel_times.push_back(time); },
        // Specify the direction.
        kw::direction = event_direction::positive);

    // Reset zero_vel_times and the integrator.
    zero_vel_times.clear();
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {-0.05, 0.}, kw::nt_events = {ev}};

    // Propagate for a few time units.
    ta.propagate_until(5);

    // Print the event times.
    for (auto t : zero_vel_times) {
        std::cout << "Event detection time: " << t << '\n';
    }

    // Define two close non-terminal events.
    nt_event<double> ev0(v, [](taylor_adaptive<double> &, double time, int) {
        std::cout << "Event 0 triggering at t=" << time << '\n';
    });
    nt_event<double> ev1(v * v - 1e-12, [](taylor_adaptive<double> &, double time, int) {
        std::cout << "Event 1 triggering at t=" << time << '\n';
    });

    // Reset the integrator.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {-0.05, 0.}, kw::nt_events = {ev0, ev1}};

    // Propagate for a few time units.
    ta.propagate_until(5);

    // Define a terminal event that turns air drag on/off
    // whenever the velocity goes to zero.
    t_event<double> t_ev(
        // The event equation.
        v,
        // The callback.
        kw::callback = [](taylor_adaptive<double> &ta, [[maybe_unused]] bool mr, int) {
            // NOTE: the value of the drag coefficient
            // is stored as the first (and only) runtime parameter
            // of the integrator.
            if (ta.get_pars()[0] == 0) {
                ta.get_pars_data()[0] = 1;
            } else {
                ta.get_pars_data()[0] = 0;
            }

            // Do not stop the integration.
            return true;
        });

    // Construct the damped pendulum integrator.
    ta = taylor_adaptive<double>{{prime(x) = v,
                                  // NOTE: alpha is represented as
                                  // the first (and only) runtime
                                  // parameter: par[0].
                                  prime(v) = -9.8 * sin(x) - par[0] * v},
                                 {0.05, 0.025},
                                 // The list of terminal events.
                                 kw::t_events = {t_ev}};

    // Propagate step-by-step until the event triggers.
    taylor_outcome oc;
    do {
        oc = std::get<0>(ta.step());
    } while (oc == taylor_outcome::success);

    // Print the outcome to screen.
    std::cout << "Integration outcome: " << oc << '\n';
    std::cout << "Event index        : " << static_cast<std::int64_t>(oc) << '\n';

    // Integrate over a time grid.
    ta.propagate_until(1.);
    auto out = ta.propagate_grid({1., 2., 3., 4., 5., 6., 7., 8., 9., 10.});

    // Let's print the values of the state vector
    // over the time grid.
    for (auto i = 0u; i < 10u; ++i) {
        std::cout << "[" << std::get<4>(out)[i * 2u] << ", " << std::get<4>(out)[i * 2u + 1u] << "]\n";
    }

    std::cout << "\nFinal time: " << ta.get_time() << '\n';
}
