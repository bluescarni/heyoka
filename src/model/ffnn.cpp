// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <vector>

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
    for (std::uint32_t i = 0u; i < n_neurons_curr_layer; ++i) {
        for (std::uint32_t j = 0u; j < n_neurons_prev_layer; ++j) {
            retval[i] += net_wb[flattenw(i, j, n_neurons, layer_id)] * inputs[j];
        }
        retval[i] += net_wb[flattenb(i, n_neurons, layer_id, n_net_w)];
        retval[i] = activation(retval[i]);
    }
    return retval;
}
} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<expression> ffnn_impl(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<expression> &in, std::uint32_t n_out,
    const std::vector<std::uint32_t> &n_neurons_per_hidden_layer,
    const std::vector<std::function<expression(const expression &)>> &activations,
    const std::vector<expression> &net_wb)
{
    // Sanity check (should be a throw check?)
    assert(n_neurons_per_hidden_layer.size() + 1 == activations.size());

    // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons)
    auto n_hidden_layers = boost::numeric_cast<std::uint32_t>(n_neurons_per_hidden_layer.size());
    // Number of neuronal layers (counting input and output)
    auto n_layers = n_hidden_layers + 2;
    // Number o
    auto n_in = boost::numeric_cast<std::uint32_t>(in.size());
    // Number of neurons per neuronal layer
    std::vector<std::uint32_t> n_neurons = n_neurons_per_hidden_layer;
    n_neurons.insert(n_neurons.begin(), n_in);
    n_neurons.insert(n_neurons.end(), n_out);
    // Number of network parameters
    std::uint32_t n_net_wb = 0u;
    std::uint32_t n_net_w = 0u;
    for (std::uint32_t i = 1u; i < n_layers; ++i) {
        n_net_wb += n_neurons[i - 1] * n_neurons[i];
        n_net_w += n_neurons[i - 1] * n_neurons[i];
        n_net_wb += n_neurons[i];
    }
    // Sanity check (should be a throw check?)
    assert(net_wb.size() == n_net_wb);
    std::vector<expression> retval{};
    for (std::uint32_t i = 1u; i < n_layers; ++i) {
        retval = detail::compute_layer(i, retval, n_neurons, activations[i], net_wb, n_net_w);
    }
    return retval;
}
} // namespace model
HEYOKA_END_NAMESPACE