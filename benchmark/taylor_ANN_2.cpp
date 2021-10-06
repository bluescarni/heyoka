// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <boost/lexical_cast.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/splitmix64.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

using namespace std::chrono;
using namespace heyoka;

// In this benchmark we construct a feed forward neural network (does not really matter with what values for weights and
// biases) and we build a taylor integrator with it to check the dimension of the resulting simplified decomposition
// w.r.t. the dimension of the actual expression represententing the network output.
//
// It turns out that the decomposion grows mildly and linearly, while the expression complexity grows exponentially and
// soon gets intractable. This highlights the need to have a dedicated NN function in the heyoka machinery, one able to
// build the decomposition directly without passing through the tree construction.

namespace heyoka
{
// TEST ME PLEASE!!!!
// This helper struct allows to build the full expression tree of a neural network
// surely a bad idea as its complexity grows exponentially with the number of hidden layers
struct ffnn {
    ffnn(unsigned n_in, unsigned n_out, unsigned n_hidden, unsigned n_neurons_per_layer)
        : m_n_in(n_in), m_n_out(n_out), m_n_layers(n_hidden + 2u), m_n_neurons(n_hidden + 2u, n_neurons_per_layer)
    {
        m_n_neurons[0] = m_n_in;
        m_n_neurons.back() = m_n_out;
        m_n_w = 0u;
        m_n_b = 0u;
        for (auto i = 0u; i < m_n_layers - 1u; i++) {
            // weights
            m_n_w += m_n_neurons[i] * m_n_neurons[i + 1];
            // biases
            m_n_b += m_n_neurons[i + 1];
        }
    };

    std::vector<expression> operator()(const std::vector<expression> &input)
    {
        assert(input.size() == m_n_in);
        std::vector<expression> retval = input;
        for (auto i = 1u; i < m_n_layers; ++i) {
            retval = compute_layer(i, retval);
        }
        return retval;
    }

    // from the layer the neuron and the input
    // returns the flattened position in par of the corresponding weight
    unsigned flattenw(unsigned layer, unsigned neuron, unsigned input)
    {
        assert(layer > 0);
        unsigned counter = 0u;
        for (auto k = 1u; k < layer; ++k) {
            counter += m_n_neurons[k] * m_n_neurons[k - 1];
        }
        counter += neuron * m_n_neurons[layer - 1];
        return counter + input;
    }

    // from the layer the neuron and the input
    // returns the flattened index in par of the corresponding bias
    unsigned flattenb(unsigned layer, unsigned neuron)
    {
        assert(layer > 0);
        unsigned counter = 0u;
        for (auto k = 1u; k < layer; ++k) {
            counter += m_n_neurons[k];
        }
        return counter + neuron + m_n_w;
    }

    std::vector<expression> compute_layer(unsigned layer, const std::vector<expression> &in)
    {
        assert(layer > 0);
        assert(in.size() == m_n_neurons[layer - 1]);
        std::vector<expression> retval(m_n_neurons[layer]);
        for (auto neuron = 0u; neuron < m_n_neurons[layer]; ++neuron) {
            // wij xj
            std::vector<expression> tmp;
            for (auto input = 0u; input < m_n_neurons[layer - 1]; ++input) {
                tmp.push_back(par[flattenw(layer, neuron, input)] * in[input]);
                // retval[neuron] += par[flattenw(layer, neuron, input)] * in[input];
            }
            // b
            tmp.push_back(par[flattenb(layer, neuron)]);
            // wij xj + bi
            retval[neuron] = pairwise_sum(tmp);
            // non linearity
            retval[neuron] = sin(retval[neuron]);
        }
        return retval;
    }

    unsigned m_n_in;
    unsigned m_n_out;
    unsigned m_n_layers;
    std::vector<unsigned> m_n_neurons;
    unsigned m_n_w;
    unsigned m_n_b;
};
} // namespace heyoka

int main()
{
    for (auto i = 2u; i < 15; ++i) {
        std::cout << "\nLayers: " << std::to_string(i) << std::endl;
        // A FFNN 2 in, 2 out 4 extra layers, ten neurons
        ffnn eclipsenet(2, 2, i, 10);
        // We build the system dx = NN(x)
        auto [x1, x2] = make_vars("x1", "x2");

        auto start = high_resolution_clock::now();
        auto out = eclipsenet(std::vector<expression>{x1, x2});
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "\tTime to compute the NN output expressions: " << duration.count() / 1e6 << "s" << std::endl;

        start = high_resolution_clock::now();
        taylor_adaptive<double> ta{out, {0.1, 0.2}, kw::compact_mode = true};
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        std::cout << "\tTime to build the Taylor integrator: " << duration.count() / 1e6 << "s" << std::endl;

        // We show here the scaling
        std::cout << "\tSize of decomposion: " << ta.get_decomposition().size() << std::endl;
        std::cout << "\tSize of output expression: " << get_n_nodes(out[0]) << std::endl;
    }
    return 0;
}
