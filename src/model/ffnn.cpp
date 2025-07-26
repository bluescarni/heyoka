// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/ffnn.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// Safe counterpart to std::uint32_t in order to avoid
// overflows when manipulating indices and sizes.
using su32 = boost::safe_numerics::safe<std::uint32_t>;

std::vector<expression> compute_layer(su32 layer_id, const std::vector<expression> &inputs,
                                      const std::vector<su32> &n_neurons,
                                      const std::function<expression(const expression &)> &activation,
                                      const std::vector<expression> &nn_wb, su32 n_net_w, su32 &wcounter,
                                      su32 &bcounter)
{
    assert(layer_id > 0u);
    auto n_neurons_prev_layer = su32(inputs.size());
    auto n_neurons_curr_layer = n_neurons[layer_id];

    std::vector<expression> retval, tmp_sum;
    retval.reserve(n_neurons_curr_layer);

    for (su32 i = 0; i < n_neurons_curr_layer; ++i) {
        // Clear the summation terms.
        tmp_sum.clear();

        for (su32 j = 0; j < n_neurons_prev_layer; ++j) {

            // Add the weight and update the weight counter.
            tmp_sum.push_back(nn_wb[wcounter] * inputs[j]);
            ++wcounter;
        }

        // Add the bias and update the counter.
        tmp_sum.push_back(nn_wb[bcounter + n_net_w]);
        ++bcounter;

        // Activation function.
        retval.push_back(activation(sum(tmp_sum)));
    }

    return retval;
}

} // namespace

// This function returns the symbolic expressions of the `n_out` output neurons in a feed forward neural network, as a
// function of the `n_in` input expressions.
//
// The expression will contain the weights and biases of the neural network flattened into `nn_wb` with the following
// conventions: from the left to right layer of parameters: [W01, W12,W23, ..., B1,B2,B3,....] where the weight matrices
// Wij are to be considered as flattened (row first) and so are the bias vectors.
std::vector<expression> ffnn_impl(const std::vector<expression> &in, const std::vector<std::uint32_t> &nn_hidden,
                                  const std::uint32_t n_out,
                                  const std::vector<std::function<expression(const expression &)>> &activations,
                                  const std::vector<expression> &nn_wb)
{
    // Sanity checks.
    if (activations.empty()) {
        throw std::invalid_argument("Cannot create a FFNN with an empty list of activation functions");
    }
    if (nn_hidden.size() != activations.size() - 1u) {
        throw std::invalid_argument(fmt::format(
            "The number of hidden layers, as detected from the inputs, was {}, while "
            "the number of activation function supplied was {}. A FFNN needs exactly one more activation function "
            "than the number of hidden layers.",
            nn_hidden.size(), activations.size()));
    }
    if (in.empty()) {
        throw std::invalid_argument("The inputs provided to the FFNN is an empty vector.");
    }
    if (n_out == 0u) {
        throw std::invalid_argument("The number of network outputs cannot be zero.");
    }
    if (!std::ranges::all_of(nn_hidden, [](auto item) { return item > 0u; })) {
        throw std::invalid_argument("The number of neurons for each hidden layer must be greater than zero!");
    }
    if (std::ranges::any_of(activations, [](const auto &func) { return !func; })) {
        throw std::invalid_argument("The list of activation functions cannot contain empty functions");
    }

    // From now on, always use safe arithmetics to compute/manipulate indices and sizes.
    using detail::su32;

    // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons).
    auto n_hidden_layers = su32(nn_hidden.size());
    // Number of neuronal layers (counting input and output).
    auto n_layers = n_hidden_layers + 2;
    // Number of inputs.
    auto n_in = su32(in.size());
    // Number of neurons per neuronal layer.
    std::vector<su32> n_neurons{n_in};
    n_neurons.insert(n_neurons.end(), nn_hidden.begin(), nn_hidden.end());
    n_neurons.insert(n_neurons.end(), n_out);
    // Number of network parameters (wb: weights and biases, w: only weights).
    su32 n_net_wb = 0, n_net_w = 0;
    for (su32 i = 1; i < n_layers; ++i) {
        n_net_wb += n_neurons[i - 1u] * n_neurons[i];
        n_net_w += n_neurons[i - 1u] * n_neurons[i];
        n_net_wb += n_neurons[i];
    }
    // Sanity check.
    if (nn_wb.size() != n_net_wb) {
        throw std::invalid_argument(fmt::format(
            "The number of network parameters, detected from its structure to be {}, does not match the size of "
            "the corresponding expressions: {}.",
            static_cast<std::uint32_t>(n_net_wb), nn_wb.size()));
    }

    // Now we build the expressions recursively transvering from layer to layer (L = f(Wx+b))).
    std::vector<expression> retval = in;
    su32 wcounter = 0, bcounter = 0;
    for (su32 i = 1; i < n_layers; ++i) {
        retval = detail::compute_layer(i, retval, n_neurons, activations[i - 1u], nn_wb, n_net_w, wcounter, bcounter);
    }
    return retval;
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
