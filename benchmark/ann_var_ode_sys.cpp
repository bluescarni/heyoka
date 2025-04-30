// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>

#include <fmt/core.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/tanh.hpp>
#include <heyoka/model/ffnn.hpp>
#include <heyoka/var_ode_sys.hpp>

// NOTE: this is a benchmark to investigate the performance of var_ode_sys with
// neural networks.

using namespace heyoka;

int main(int, char *[])
{
    set_logger_level_trace();

    const auto n = 6;
    const auto h = 40;
    const auto nh = 4;

    // Create the state variables.
    std::vector<expression> state;
    state.reserve(n);
    for (auto i = 0; i < n; ++i) {
        state.emplace_back(fmt::format("x{}", i));
    }

    // Create the NN model.
    const auto linear = [](const expression &x) { return x; };
    std::vector<std::function<expression(const expression &)>> activations;
    activations.reserve(nh + 1);
    for (auto i = 0; i < nh; ++i) {
        activations.emplace_back([](const expression &x) { return tanh(x); });
    }
    activations.emplace_back(linear);
    const auto ffnn = model::ffnn(kw::inputs = state, kw::nn_hidden = std::vector(nh, h), kw::n_out = n,
                                  kw::activations = activations);

    // Create the dynamics.
    std::vector<std::pair<expression, expression>> dyn;
    dyn.reserve(n);
    for (auto i = 0; i < n; ++i) {
        dyn.emplace_back(state[i], ffnn[i]);
    }

    auto vsys = var_ode_sys(dyn, var_args::params);
}
