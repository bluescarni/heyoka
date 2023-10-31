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

namespace model
{
namespace detail
{
// From the i-th neuron of the layer_id and the incoming j-th connection, returns the corresponding weight position in
// the flattened structure. NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<expression>::size_type flattenw(std::uint32_t i, std::uint32_t j,
                                            const std::vector<std::uint32_t> &n_neurons, std::uint32_t layer_id)
{
    assert(layer_id > 0);
    // The weight for the jth-neuron of the ith layer will be placed after all previous layers.
    // We start counting how many in flattened.
    std::uint32_t counter = 0;
    for (std::uint32_t k = 1; k < layer_id; ++k) {
        counter += n_neurons[k] * n_neurons[k - 1];
    }
    // We then add the weights used for the previous neurons on the same layer.
    counter += i * n_neurons[layer_id - 1];
    // And return the index corresponding to the j-th weight.
    return counter + j;
}

// From the i-th neuron of the layer_id, returns the corresponding bias position in the
// flattened structure.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<expression>::size_type flattenb(std::uint32_t i, const std::vector<std::uint32_t> &n_neurons,
                                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                            std::uint32_t layer_id, std::uint32_t n_net_w)
{
    assert(layer_id > 0);
    // The weight for the jth-neuron of the ith layer will be placed after all previous layers.
    // We start counting how many in flattened.
    std::uint32_t counter = 0;
    for (std::uint32_t k = 1; k < layer_id; ++k) {
        counter += n_neurons[k];
    }
    // And return the index corresponding to the i-th bias.
    return counter + i + n_net_w;
}

std::vector<expression> compute_layer(std::uint32_t layer_id, const std::vector<expression> &inputs,
                                      const std::vector<std::uint32_t> &n_neurons,
                                      const std::function<expression(const expression &)> &activation,
                                      const std::vector<expression> &net_wb, std::uint32_t n_net_w)
{
    assert(layer_id > 0);
    auto n_neurons_prev_layer = boost::numeric_cast<std::uint32_t>(inputs.size());
    auto n_neurons_curr_layer = n_neurons[layer_id];

    std::vector<expression> retval(n_neurons_curr_layer, 0_dbl);
    fmt::print("nneurons: {}", n_neurons_prev_layer);
    std::cout << std::endl;

    for (std::uint32_t i = 0u; i < n_neurons_curr_layer; ++i) {
        for (std::uint32_t j = 0u; j < n_neurons_prev_layer; ++j) {
            fmt::print("layer, i, j: {}, {}, {}", layer_id, i, j);
            std::cout << std::endl;
            retval[i] += net_wb[flattenw(i, j, n_neurons, layer_id)] * inputs[j];
        }
        retval[i] += net_wb[flattenb(i, n_neurons, layer_id, n_net_w)];
        retval[i] = activation(retval[i]);
    }
    return retval;
}
} // namespace detail

std::vector<expression> ffnn_impl(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<expression> &in, std::uint32_t n_out,
    const std::vector<std::uint32_t> &n_neurons_per_hidden_layer,
    const std::vector<std::function<expression(const expression &)>> &activations,
    const std::vector<expression> &net_wb)
{
    // Sanity checks
    if (n_neurons_per_hidden_layer.size() + 1 != activations.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of hidden layers, as detected from the inputs, was {}, while"
            "the number of activation function supplied was {}. A FFNN needs exactly one more activation function "
            "than the number of hidden layers.",
            n_neurons_per_hidden_layer.size(), activations.size()));
    }
    if (in.empty()) {
        throw std::invalid_argument("The inputs provided to the ffnn seem to be an empty vector.");
    }
    if (n_out == 0) {
        throw std::invalid_argument("The number of network outputs cannot be zero.");
    }
    if (!std::all_of(n_neurons_per_hidden_layer.begin(), n_neurons_per_hidden_layer.end(),
                     [](std::uint32_t item) { return item > 0; })) {
        throw std::invalid_argument("The number of neurons for each hidden layer must be greater than zero!");
    }
    if (n_neurons_per_hidden_layer.empty()) { // TODO(darioizzo): maybe this is actually a wanted corner case, remove?
        throw std::invalid_argument("The number of hidden layers cannot be zero.");
    }

    // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons)
    auto n_hidden_layers = boost::numeric_cast<std::uint32_t>(n_neurons_per_hidden_layer.size());
    // Number of neuronal layers (counting input and output)
    auto n_layers = n_hidden_layers + 2;
    // Number of inputs
    auto n_in = boost::numeric_cast<std::uint32_t>(in.size());
    // Number of neurons per neuronal layer
    std::vector<std::uint32_t> n_neurons = n_neurons_per_hidden_layer;
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
    if (net_wb.size() != n_net_wb) {
        throw std::invalid_argument(fmt::format(
            "The number of network parameters, detected from its structure to be {}, does not match the size of"
            "the corresponding expressions {} ",
            n_net_wb, net_wb.size()));
    }

    // Now we build the expressions recursively going from layer to layer (L = f(Wx+b)))

    std::vector<expression> retval = in;
    for (std::uint32_t i = 1u; i < n_layers; ++i) {
        fmt::print("{},{}", i, n_neurons[i]);
        std::cout << std::endl;
        retval = detail::compute_layer(i, retval, n_neurons, activations[i], net_wb, n_net_w);
    }
    return retval;
}
} // namespace model
HEYOKA_END_NAMESPACE