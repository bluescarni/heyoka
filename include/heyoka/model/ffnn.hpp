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
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{
// This c++ function returns the symbolic expressions of the `n_out` output neurons in a feed forward neural network,
// as a function of the `n_in` input expressions.
//
// The expression will contain the weights and biases of the neural network flattened into `pars` with the following
// conventions:
//
// from the left to right layer of parameters: [W01, W12,W23, ..., B1,B2,B3,....] where the weight matrices Wij are
// to be considered as flattened (row first) and so are the bias vectors.
//
HEYOKA_DLL_PUBLIC std::vector<expression> ffnn_impl(const std::vector<expression> &, std::uint32_t,
                                                    const std::vector<std::uint32_t> &,
                                                    const std::vector<std::function<expression(const expression &)>> &,
                                                    const std::vector<expression> &);
} // namespace model

HEYOKA_END_NAMESPACE

#endif
