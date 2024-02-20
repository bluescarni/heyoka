// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FFNN_HPP
#define HEYOKA_MODEL_FFNN_HPP

#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{
namespace detail
{

template <typename... KwArgs>
auto ffnn_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Network inputs. Mandatory.
    // The kw::inputs argument must be a range of values from which
    // an expression can be constructed.
    std::vector<expression> inputs;
    if constexpr (p.has(kw::inputs)) {
        for (const auto &val : p(kw::inputs)) {
            inputs.emplace_back(val);
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                      "The 'inputs' keyword argument is necessary but it was not provided");
    }

    // Number of hidden neurons per hidden layer. Mandatory.
    // The kw::nn_hidden argument must be a range containing
    // integral values.
    std::vector<std::uint32_t> nn_hidden;
    if constexpr (p.has(kw::nn_hidden)) {
        for (const auto &nval : p(kw::nn_hidden)) {
            nn_hidden.push_back(boost::numeric_cast<std::uint32_t>(nval));
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                      "The 'nn_hidden' keyword argument is necessary but it was not provided");
    }

    // Number of network outputs. Mandatory.
    // The kw::n_out argument must be of integral type.
    auto n_out = [&p]() {
        if constexpr (p.has(kw::n_out)) {
            return boost::numeric_cast<std::uint32_t>(p(kw::n_out));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'n_out' keyword argument is necessary but it was not provided");
        }
    }();

    // Network activation functions. Mandatory.
    // The kw::activations argument must be a range containing values
    // from which a std::function can be constructed.
    std::vector<std::function<expression(const expression &)>> activations;
    if constexpr (p.has(kw::activations)) {
        for (const auto &f : p(kw::activations)) {
            activations.emplace_back(f);
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                      "The 'activations' keyword argument is necessary but it was not provided");
    }

    // Network weights and biases. Optional, defaults to heyoka parameters.
    // The kw::nn_wb argument, if present, must be a range of values from which
    // expressions can be constructed.
    std::vector<expression> nn_wb;
    if constexpr (p.has(kw::nn_wb)) {
        for (const auto &val : p(kw::nn_wb)) {
            nn_wb.emplace_back(val);
        }
    } else {
        // Safe counterpart to std::uint32_t in order to avoid
        // overflows when manipulating indices and sizes.
        using su32 = boost::safe_numerics::safe<std::uint32_t>;

        // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons).
        auto n_hidden_layers = su32(nn_hidden.size());
        // Number of neuronal layers (counting input and output).
        auto n_layers = n_hidden_layers + 2;
        // Number of inputs.
        auto n_in = su32(inputs.size());
        // Number of neurons per neuronal layer.
        std::vector<su32> n_neurons{n_in};
        n_neurons.insert(n_neurons.end(), nn_hidden.begin(), nn_hidden.end());
        n_neurons.insert(n_neurons.end(), n_out);

        // Number of network parameters (wb: weights and biases, w: only weights).
        su32 n_wb = 0;
        for (su32 i = 1; i < n_layers; ++i) {
            n_wb += n_neurons[i - 1] * n_neurons[i];
            n_wb += n_neurons[i];
        }
        nn_wb.resize(n_wb);
        for (decltype(nn_wb.size()) i = 0; i < nn_wb.size(); ++i) {
            nn_wb[i] = par[boost::numeric_cast<std::uint32_t>(i)];
        }
    }

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
