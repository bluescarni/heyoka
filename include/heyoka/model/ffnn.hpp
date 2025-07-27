// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FFNN_HPP
#define HEYOKA_MODEL_FFNN_HPP

#include <concepts>
#include <cstdint>
#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/ranges_to.hpp>
#include <heyoka/detail/safe_integer.hpp>
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
    using heyoka::detail::ranges_to;

    const igor::parser p{kw_args...};

    // Network inputs.
    auto inputs = ranges_to<std::vector<expression>>(p(kw::inputs));

    // Number of hidden neurons per hidden layer.
    //
    // NOTE: turn the kwarg into an lvalue otherwise we cannot pipe it into the range transformation if it is an
    // initializer list.
    auto &&nnh_arg = p(kw::nn_hidden);
    auto nn_hidden = ranges_to<std::vector<std::uint32_t>>(
        nnh_arg | std::views::transform([](const auto n) { return boost::numeric_cast<std::uint32_t>(n); }));

    // Number of network outputs.
    const auto n_out = boost::numeric_cast<std::uint32_t>(p(kw::n_out));

    // Network activation functions.
    auto activations = ranges_to<std::vector<std::function<expression(const expression &)>>>(p(kw::activations));

    // Network weights and biases. Optional, defaults to heyoka parameters.
    std::vector<expression> nn_wb;
    if constexpr (p.has(kw::nn_wb)) {
        nn_wb = ranges_to<std::vector<expression>>(p(kw::nn_wb));
    } else {
        // Safe counterpart to std::uint32_t in order to avoid
        // overflows when manipulating indices and sizes.
        using su32 = boost::safe_numerics::safe<std::uint32_t>;

        // Number of hidden layers (defined as all neuronal columns that are nor input nor output neurons).
        const auto n_hidden_layers = su32(nn_hidden.size());
        // Number of neuronal layers (counting input and output).
        const auto n_layers = n_hidden_layers + 2;
        // Number of inputs.
        const auto n_in = su32(inputs.size());
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

    return std::tuple{std::move(inputs), std::move(nn_hidden), n_out, std::move(activations), std::move(nn_wb)};
}

HEYOKA_DLL_PUBLIC std::vector<expression> ffnn_impl(const std::vector<expression> &, const std::vector<std::uint32_t> &,
                                                    std::uint32_t,
                                                    const std::vector<std::function<expression(const expression &)>> &,
                                                    const std::vector<expression> &);
} // namespace detail

inline constexpr auto ffnn_kw_cfg = igor::config<
    kw::descr::constructible_input_range<kw::inputs, expression, true>,
    igor::descr<kw::nn_hidden,
                []<typename U>() {
                    return requires {
                        requires std::ranges::input_range<U>;
                        requires std::integral<std::remove_cvref_t<std::ranges::range_reference_t<U>>>;
                    };
                }>{.required = true},
    kw::descr::integral<kw::n_out, true>,
    kw::descr::constructible_input_range<kw::activations, std::function<expression(const expression &)>, true>,
    kw::descr::constructible_input_range<kw::nn_wb, expression>>{};

inline constexpr auto ffnn = []<typename... KwArgs>
    requires igor::validate<ffnn_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> std::vector<expression> {
    return std::apply(detail::ffnn_impl, detail::ffnn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
