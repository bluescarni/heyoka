// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

using namespace std::chrono;
using namespace heyoka;

int main()
{
    // System Dimension
    auto n_neurons = 50u;
    auto n_in = 4u;
    auto n_out = 2u;

    // System state
    std::vector<expression> x;
    for (auto i = 0u; i < n_in; ++i) {
        x.emplace_back(variable{"x" + std::to_string(i)});
    }

    // Network weights and biases
    auto n_w = (n_in + 1) * n_neurons + (n_neurons + 1) * n_out;
    std::vector<expression> w;
    for (auto i = 0u; i < n_w; ++i) {
        w.emplace_back(variable{"w" + std::to_string(i)});
    }

    // We compute the outputs of the first (and only) layer of neurons
    std::vector<expression> hidden;
    for (auto i = 0u; i < n_neurons; ++i) {
        // the bias
        hidden.push_back(w[i * (n_in + 1)]);
    }
    // The weighted sum of inputs
    for (decltype(hidden.size()) i = 0u; i < hidden.size(); ++i) {
        for (auto j = 0u; j < n_in; ++j) {
            auto ji = i * (n_in + 1) + (j + 1);
            hidden[i] += w[ji] * x[j];
        }
        // the non linearity
        hidden[i] = sin(hidden[i]);
    }

    // We compute the outputs of the output layer
    auto offset = n_neurons * (n_in + 1);
    std::vector<expression> out;
    for (auto i = 0u; i < n_out; ++i) {
        // the bias
        out.push_back(w[offset + i * (n_neurons + 1)]);
    }
    for (auto i = 0u; i < n_out; ++i) {
        // the weighted sum of inputs
        for (auto j = 0u; j < n_neurons; ++j) {
            // compute the weight number
            auto ji = offset + i * (n_neurons + 1) + (j + 1);
            // add the weighted input
            out[i] += w[ji] * hidden[j];
        }
        // the non linearity
        out[i] = sin(out[i]);
    }

    // Assembling the dynamics (weights and biases derivatives are zero)
    std::vector<expression> dynamics;
    // kinematics
    dynamics.push_back(x[2]);
    dynamics.push_back(x[3]);
    // dynamics
    dynamics.push_back(out[0]);
    dynamics.push_back(out[1]);
    // parameters
    for (decltype(w.size()) i = 0u; i < w.size(); ++i) {
        dynamics.push_back(0_dbl);
    }

    // Setting the initial conditions (random weights and biases initialization)
    detail::random_engine_type engine(123u);
    std::vector<double> ic = {0., 0., 0., 0.};
    for (decltype(w.size()) i = 0u; i < w.size(); ++i) {
        ic.push_back(std::uniform_real_distribution<>(-1., 1.)(engine));
    }

    // Defining the integrator
    std::cout << "Calling LLVM: " << std::endl;
    auto start = high_resolution_clock::now();
    taylor_adaptive_dbl neural_network_ode{dynamics, ic};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Microseconds: " << duration.count() << std::endl;

    // Calling the integrator
    std::cout << "Calling the integrator: " << std::endl;
    start = high_resolution_clock::now();
    neural_network_ode.propagate_until(1000.);
    neural_network_ode.propagate_until(0.);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Microseconds: " << duration.count() << std::endl;
    std::cout << "Error in x:" << neural_network_ode.get_state()[0] << std::endl;
    std::cout << "Error in y:" << neural_network_ode.get_state()[1] << std::endl;
    std::cout << "Error in vx:" << neural_network_ode.get_state()[2] << std::endl;
    std::cout << "Error in vy:" << neural_network_ode.get_state()[3] << std::endl;

    return 0;
}
