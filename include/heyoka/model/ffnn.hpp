// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FFNN_HPP
#define HEYOKA_MODEL_FFNN_HPP

#include <atomic>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{
namespace detail
{
template <typename... KwArgs>
auto ffnn_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Network inputs. Mandatory
    auto inputs = [&p]() {
        if constexpr (p.has(kw::inputs)) {
            return std::vector<expression>{p(kw::inputs)};
        } else {
            static_assert(::heyoka::detail::always_false_v<KwArgs...>,
                          "The 'inputs' keyword argument is necessary but it was not provided");
        }
    }();

    // Number of hidden neurons per hidden layer. Mandatory
    auto nn_hidden = [&p]() {
        if constexpr (p.has(kw::nn_hidden)) {
            return std::vector<std::uint32_t>{p(kw::nn_hidden)};
        } else {
            static_assert(::heyoka::detail::always_false_v<KwArgs...>,
                          "The 'nn_hidden' keyword argument is necessary but it was not provided");
        }
    }();

    // Number of network outputs. Mandatory
    auto n_out = [&p]() {
        if constexpr (p.has(kw::n_out)) {
            return std::uint32_t{p(kw::n_out)};
        } else {
            static_assert(::heyoka::detail::always_false_v<KwArgs...>,
                          "The 'n_out' keyword argument is necessary but it was not provided");
        }
    }();

    // Network activation functions. Mandatory
    auto activations = [&p]() {
        if constexpr (p.has(kw::activations)) {
            return std::vector<std::function<expression(const expression &)>>{p(kw::activations)};
        } else {
            static_assert(::heyoka::detail::always_false_v<KwArgs...>,
                          "The 'activations' keyword argument is necessary but it was not provided");
        }
    }();

    // Network weights and biases. Defaults to heyoka parameters.
    auto nn_wb = [&p, &nn_hidden, &inputs, n_out]() {
        if constexpr (p.has(kw::nn_wb)) {
            return std::vector<expression> {p(kw::nn_wb)};
        } else {
            // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons)
            auto n_hidden_layers = boost::numeric_cast<std::uint32_t>(nn_hidden.size());
            // Number of neuronal layers (counting input and output)
            auto n_layers = n_hidden_layers + 2;
            // Number of inputs
            auto n_in = boost::numeric_cast<std::uint32_t>(inputs.size());
            // Number of neurons per neuronal layer
            std::vector<std::uint32_t> n_neurons = nn_hidden;
            n_neurons.insert(n_neurons.begin(), n_in);
            n_neurons.insert(n_neurons.end(), n_out);
            // Number of network parameters (wb: weights and biases, w: only weights)
            std::uint32_t n_wb = 0u;
            for (std::uint32_t i = 1u; i < n_layers; ++i) {
                n_wb += n_neurons[i - 1] * n_neurons[i];
                n_wb += n_neurons[i];
            }
            std::vector<expression> retval(n_wb);
            for (decltype(retval.size()) i = 0; i < retval.size(); ++i) {
                retval[i] = heyoka::par[i];
            }
            return retval;
        }
    }();

    return std::tuple{std::move(inputs), std::move(nn_hidden), std::move(n_out), std::move(activations),
                      std::move(nn_wb)};
}

// This c++ function returns the symbolic expressions of the `n_out` output neurons in a feed forward neural network,
// as a function of the `n_in` input expressions.
//
// The expression will contain the weights and biases of the neural network flattened into `pars` with the following
// conventions:
//
// from the left to right layer of parameters: [W01, W12,W23, ..., B1,B2,B3,....] where the weight matrices Wij are
// to be considered as flattened (row first) and so are the bias vectors.
//
HEYOKA_DLL_PUBLIC std::vector<expression> ffnn_impl(const std::vector<expression> &, const std::vector<std::uint32_t> &,
                                                    std::uint32_t,
                                                    const std::vector<std::function<expression(const expression &)>> &,
                                                    const std::vector<expression> &);
} // namespace detail

inline constexpr auto ffnn = [](const auto &...kw_args) -> std::vector<expression> {
    return std::apply(detail::ffnn_impl, detail::ffnn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
