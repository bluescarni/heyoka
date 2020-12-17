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

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/splitmix64.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

using namespace std::chrono;
using namespace heyoka;

expression sigmoid(const expression &in)
{
    return 1._dbl / (1._dbl + exp(-in));
}

// In this benchmark we integrate the two dimensional motion of a spacecraft perturbed by a conservative field
// defined by a neural network. The propagation is made forward and then backward to 0. The error is thus computed.

int main()
{
    // System Dimension
    auto n_neurons = 10u;
    auto n_in = 2u;
    auto n_out = 1u;
    unsigned offset_w_names = 1000u;

    // System state (letter a is used to make sure the state comes before the weights w)
    std::vector<expression> x;
    for (auto i = 0u; i < 4u; ++i) {
        x.emplace_back(variable{"a" + std::to_string(i)});
    }

    // Network paramterers: weights and biases (w0001-w0002... guarantees correct alphabetical order up to
    // offset_w_names - 1) parameters)
    auto n_w = (n_in + 1) * n_neurons + (n_neurons + 1) * n_out;
    std::vector<expression> w;
    for (auto i = 0u; i < n_w; ++i) {
        w.emplace_back(variable{"w" + std::to_string(i + offset_w_names)});
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
        hidden[i] = sigmoid(hidden[i]);
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
        out[i] = sigmoid(out[i]);
    }

    // Assembling the dynamics (weights and biases derivatives are zero)
    std::vector<expression> dynamics;
    // kinematics
    dynamics.push_back(x[2]);
    dynamics.push_back(x[3]);
    // dynamics
    auto f0 = diff(out[0] + 1_dbl / (pow(x[0] * x[0] + x[1] * x[1], 0.5_dbl)), "a0");
    auto f1 = diff(out[0] + 1_dbl / (pow(x[0] * x[0] + x[1] * x[1], 0.5_dbl)), "a1");
    dynamics.push_back(f0);
    dynamics.push_back(f1);
    // parameters are constants -> derivative to zero
    for (decltype(w.size()) i = 0u; i < w.size(); ++i) {
        dynamics.push_back(0_dbl);
    }

    // Setting the initial conditions (random weights and biases initialization)
    splitmix64 engine(123u);
    std::vector<double> ic = {1., 0., 0., 1.};
    for (decltype(w.size()) i = 0u; i < w.size(); ++i) {
        ic.push_back(std::normal_distribution<>(0., 1.)(engine));
    }

    // Defining the integrator
    std::cout << "\nCompiling the Taylor Integrator (" << std::to_string(w.size()) << " parameters)." << std::endl;
    auto start = high_resolution_clock::now();
    taylor_adaptive_dbl neural_network_ode{dynamics, ic};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Microseconds: " << duration.count() << std::endl;

    // Calling the integrator
    std::cout << "\nCalling the Taylor Integrator." << std::endl;
    start = high_resolution_clock::now();
    // Longer times result in reaching limit cycles and thus loss of precision
    auto state = neural_network_ode.get_state();

    // Uncomment these lines to print the state on screen during the first steps
    // std::unordered_map<std::string, double> eval_map;
    // eval_map["a0"] = state[0];
    // eval_map["a1"] = state[1];
    // for (decltype(w.size()) i = 0u; i < w.size(); ++i) {
    //    eval_map["w" + std::to_string(i + offset_w_names)] = ic[4 + i];
    //}
    // auto V = eval_dbl(-out[0] - 1_dbl / pow(x[0] * x[0] + x[1] * x[1], 0.5_dbl), eval_map);
    // auto E0 = V + 0.5 * (state[2] * state[2] + state[3] * state[3]);
    // for (auto i = 0u; i < 100; ++i) {
    //    auto res = neural_network_ode.step();
    //    state = neural_network_ode.get_state();
    //    eval_map["a0"] = state[0];
    //    eval_map["a1"] = state[1];
    //    V = eval_dbl(-out[0] - 1_dbl / pow(x[0] * x[0] + x[1] * x[1], 0.5_dbl), eval_map);
    //    auto E = V + 0.5 * (state[2] * state[2] + state[3] * state[3]);
    //    std::cout << neural_network_ode.get_time() << "," << state[0] << "," << state[1] << "," << E - E0 << ", "
    //              << static_cast<int>(std::get<0>(res)) << ", " << V << ", " << std::endl;
    //}

    neural_network_ode.propagate_until(100.);
    neural_network_ode.propagate_until(0.);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Microseconds: " << duration.count() << std::endl;
    auto error = neural_network_ode.get_state();
    std::transform(error.begin(), error.begin() + n_in, ic.begin(), error.begin(),
                   [](auto a, auto b) { return std::abs(a - b); });
    std::cout << "Error:" << *std::max_element(error.begin(), error.begin() + n_in) << std::endl;
    return 0;
}
