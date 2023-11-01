// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/ffnn.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{
std::vector<expression> compute_layer(std::uint32_t layer_id, const std::vector<expression> &inputs,
                                      const std::vector<std::uint32_t> &n_neurons,
                                      const std::function<expression(const expression &)> &activation,
                                      const std::vector<expression> &nn_wb, std::uint32_t n_net_w,
                                      std::uint32_t &wcounter, std::uint32_t &bcounter)
{
    assert(layer_id > 0);
    auto n_neurons_prev_layer = boost::numeric_cast<std::uint32_t>(inputs.size());
    auto n_neurons_curr_layer = n_neurons[layer_id];

    std::vector<expression> retval(n_neurons_curr_layer, 0_dbl);
    for (std::uint32_t i = 0u; i < n_neurons_curr_layer; ++i) {
        for (std::uint32_t j = 0u; j < n_neurons_prev_layer; ++j) {

            // Add the weight and update the weight counter
            retval[i] += nn_wb[wcounter] * inputs[j];
            ++wcounter;
        }

        // Add the bias and update the counter
        retval[i] += nn_wb[bcounter + n_net_w];
        ++bcounter;
        // Activation function
        retval[i] = activation(retval[i]);
    }
    return retval;
}

std::vector<expression> ffnn_impl(const std::vector<expression> &in, const std::vector<std::uint32_t> &nn_hidden,
                                  std::uint32_t n_out,
                                  const std::vector<std::function<expression(const expression &)>> &activations,
                                  const std::vector<expression> &nn_wb)
{
    // Sanity checks
    if (nn_hidden.size() + 1 != activations.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of hidden layers, as detected from the inputs, was {}, while "
            "the number of activation function supplied was {}. A FFNN needs exactly one more activation function "
            "than the number of hidden layers.",
            nn_hidden.size(), activations.size()));
    }
    if (in.empty()) {
        throw std::invalid_argument("The inputs provided to the FFNN is an empty vector.");
    }
    if (n_out == 0) {
        throw std::invalid_argument("The number of network outputs cannot be zero.");
    }
    if (!std::all_of(nn_hidden.begin(), nn_hidden.end(), [](std::uint32_t item) { return item > 0; })) {
        throw std::invalid_argument("The number of neurons for each hidden layer must be greater than zero!");
    }

    // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons)
    auto n_hidden_layers = boost::numeric_cast<std::uint32_t>(nn_hidden.size());
    // Number of neuronal layers (counting input and output)
    auto n_layers = n_hidden_layers + 2;
    // Number of inputs
    auto n_in = boost::numeric_cast<std::uint32_t>(in.size());
    // Number of neurons per neuronal layer
    std::vector<std::uint32_t> n_neurons = nn_hidden;
    n_neurons.insert(n_neurons.begin(), n_in);
    n_neurons.insert(n_neurons.end(), n_out);
    // Number of network parameters (wb: weights and biases, w: only weights)
    std::uint32_t n_net_wb = 0u;
    std::uint32_t n_net_w = 0u;
    for (std::uint32_t i = 1u; i < n_layers; ++i) {
        n_net_wb += n_neurons[i - 1] * n_neurons[i];
        n_net_w += n_neurons[i - 1] * n_neurons[i];
        n_net_wb += n_neurons[i];
    }
    // Sanity check
    if (nn_wb.size() != n_net_wb) {
        throw std::invalid_argument(fmt::format(
            "The number of network parameters, detected from its structure to be {}, does not match the size of "
            "the corresponding expressions: {}.",
            n_net_wb, nn_wb.size()));
    }

    // Now we build the expressions recursively transvering from layer to layer (L = f(Wx+b)))
    std::vector<expression> retval = in;
    std::uint32_t wcounter = 0;
    std::uint32_t bcounter = 0;
    for (std::uint32_t i = 1u; i < n_layers; ++i) {
        retval = detail::compute_layer(i, retval, n_neurons, activations[i - 1], nn_wb, n_net_w, wcounter, bcounter);
    }
    return retval;
}
} // namespace model
HEYOKA_END_NAMESPACE