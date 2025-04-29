// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ANALYTICAL_THEORIES_HELPERS_HPP
#define HEYOKA_DETAIL_ANALYTICAL_THEORIES_HELPERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>

// NOTE: this header contains utilities for the implementation of analytical theories
// of celestial mechanics.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Create the expression for the evaluation of the polynomial with coefficients
// stored (in dense form) in cfs according to Horner's scheme.
template <typename T, std::size_t N>
expression horner_eval(const std::array<T, N> &cfs, const expression &x)
{
    static_assert(N > 0u);

    auto ret = expression(cfs[N - 1u]);

    for (std::size_t i = 1; i < N; ++i) {
        ret = cfs[N - i - 1u] + ret * x;
    }

    return ret;
} // LCOV_EXCL_LINE

std::array<expression, 2> ex_cmul(const std::array<expression, 2> &, const std::array<expression, 2> &);
std::array<expression, 2> ex_cinv(const std::array<expression, 2> &);

// Dictionary to map an integral exponent to the corresponding integral power of a complex expression.
using pow_dict_t = boost::unordered_flat_map<std::int8_t, std::array<expression, 2>>;

// Dictionary to map an expression "ex" to a a dictionary of integral powers of
// cos(ex) + im * sin(ex).
using trig_eval_dict_t = boost::unordered_flat_map<expression, pow_dict_t, std::hash<expression>>;

std::array<expression, 2> ccpow(const expression &, trig_eval_dict_t &, std::int8_t);
std::array<expression, 2> pairwise_cmul(std::vector<std::array<expression, 2>> &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
