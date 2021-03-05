// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

    std::cout << "State array:\n" << s_arr << "\n\n";
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
}
