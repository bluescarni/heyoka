// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/heyoka.hpp>

using namespace heyoka;
namespace hy = heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // We will be using a batch size of 4.
    const auto batch_size = 4;

    // Create flat 1D vectors to hold the system state
    // and the runtime parameters.
    std::vector<double> state(2 * batch_size), pars(1 * batch_size);

    // Create xtensor adaptors on the state and
    // parameters vectors for ease of indexing.
    auto s_arr = xt::adapt(state.data(), {2, batch_size});
    auto p_arr = xt::adapt(pars.data(), {1, batch_size});

    // Setup the initial conditions.
    xt::view(s_arr, 0, xt::all()) = xt::xarray<double>{0.01, 0.02, 0.03, 0.04}; // x_0
    xt::view(s_arr, 1, xt::all()) = xt::xarray<double>{1.85, 1.86, 1.87, 1.88}; // v_0
    std::cout << "State array:\n" << s_arr << "\n\n";

    // Setup the parameter values.
    xt::view(p_arr, 0, xt::all()) = xt::xarray<double>{0.10, 0.11, 0.12, 0.13}; // alpha
    std::cout << "Parameters array:\n" << p_arr << "\n\n";

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive_batch<double>{// Definition of the ODE system:
                                            // x' = v
                                            // v' = cos(t) - alpha*v - sin(x)
                                            {prime(x) = v, prime(v) = cos(hy::time) - par[0] * v - sin(x)},
                                            // Initial conditions
                                            // for x and v.
                                            std::move(state),
                                            // The batch size.
                                            batch_size,
                                            // The vector of parameters.
                                            kw::pars = std::move(pars)};

    // Create an adaptor for the time array and
    // print its contents.
    auto t_arr = xt::adapt(ta.get_time_data(), {batch_size});
    std::cout << "Time array: " << t_arr << "\n\n";

    // Perform a single step forward in time.
    ta.step();

    // Iterate over the vector of outcomes
    // and print them to screen.
    for (auto i = 0u; i < batch_size; ++i) {
        auto [oc, h] = ta.get_step_res()[i];

        std::cout << "Batch index " << i << ": (" << oc << ", " << h << ")\n";
    }

    std::cout << "\nState array:\n" << s_arr << "\n\n";
    std::cout << "Time array:\n" << t_arr << "\n\n";

    // Perform a single step forward in time
    // clamping the maximum absolute values
    // of the timesteps.
    ta.step({0.010, 0.011, 0.012, 0.013});

    for (auto i = 0u; i < batch_size; ++i) {
        auto [oc, h] = ta.get_step_res()[i];

        std::cout << "Batch index " << i << ": (" << oc << ", " << h << ")\n";
    }

    std::cout << "\nState array:\n" << s_arr << "\n\n";
    std::cout << "Time array:\n" << t_arr << "\n\n";

    // Propagate for different time intervals.
    ta.propagate_for({10., 11., 12., 13.});
    for (auto i = 0u; i < batch_size; ++i) {
        auto [oc, min_h, max_h, nsteps] = ta.get_propagate_res()[i];

        std::cout << "Batch index " << i << ": (" << oc << ", " << min_h << ", " << max_h << ", " << nsteps << ")\n";
    }
    std::cout << "\nState array:\n" << s_arr << "\n\n";
    std::cout << "Time array:\n" << t_arr << "\n\n";

    // Propagate up to different time coordinates.
    ta.propagate_until({20., 21., 22., 23.});
    for (auto i = 0u; i < batch_size; ++i) {
        auto [oc, min_h, max_h, nsteps] = ta.get_propagate_res()[i];

        std::cout << "Batch index " << i << ": (" << oc << ", " << min_h << ", " << max_h << ", " << nsteps << ")\n";
    }
    std::cout << "\nState array:\n" << s_arr << "\n\n";
    std::cout << "Time array:\n" << t_arr << "\n\n";

    // Propagate for another timestep, making
    // sure the Taylor coefficients are recorded.
    ta.step(true);

    // Create an xtensor adaptor over the
    // vector of Taylor coefficients.
    auto tc_arr = xt::adapt(ta.get_tc(), {// First dimension: number of state variables.
                                          2,
                                          // Second dimension: total number of orders for
                                          // the Taylor coefficients.
                                          int(ta.get_order()) + 1,
                                          // Third dimension: batch size.
                                          batch_size});

    std::cout << "Array of Taylor coefficients:\n" << tc_arr << "\n\n";

    std::cout << "Order-0 x: " << xt::view(tc_arr, 0, 0, xt::all()) << '\n';
    std::cout << "Order-0 v: " << xt::view(tc_arr, 1, 0, xt::all()) << '\n';

    // Compute the dense output at different time coordinates,
    // and create an xtensor adaptor on the dense output
    // for ease of indexing.
    auto d_out_arr = xt::adapt(ta.update_d_output({20.1, 21.1, 22.1, 23.1}), {2, batch_size});

    std::cout << "\nDense output:\n" << d_out_arr << "\n\n";

    // Create the event object for the detection
    // of 'v == 0'.
    nt_event_batch<double> ev(
        // The left-hand side of the event equation
        v,
        // The callback.
        [](auto &ta_, double time, int, std::uint32_t batch_idx) {
            // Compute the state of the system when the event triggered and
            // print the value of t and x for the batch element batch_idx.
            ta_.update_d_output({time, time, time, time});

            std::cout << "Zero velocity time and angle for batch element " << batch_idx << ": " << time << ", "
                      << ta_.get_d_output()[batch_idx] << '\n';
        });

    // Create the integrator object.
    ta = taylor_adaptive_batch<double>{// Definition of the ODE system:
                                       // x' = v
                                       // v' = -9.8 * sin(x)
                                       {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       // Batches of initial conditions
                                       // for x and v.
                                       {0, 0.01, 0.02, 0.03, .25, .26, .27, .28},
                                       // The batch size.
                                       batch_size,
                                       // The non-terminal events.
                                       kw::nt_events = {ev}};

    // Propagate all batch elements for
    // a few time units.
    ta.propagate_for({5., 5., 5., 5.});
}
