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
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <ranges>
#include <type_traits>
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
template <typename R>
    requires std::ranges::contiguous_range<R> && std::ranges::sized_range<R>
             && (std::same_as<expression, std::remove_cvref_t<std::ranges::range_reference_t<R>>>
                 || std::same_as<double, std::remove_cvref_t<std::ranges::range_reference_t<R>>>)
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
expression horner_eval(R &&cfs, const expression &x)
{
    const auto sz = std::ranges::size(cfs);
    if (sz == 0) {
        return 0_dbl;
    }

    const auto *ptr = std::ranges::data(cfs);
    auto ret = expression(ptr[sz - 1]);
    for (std::ranges::range_size_t<R> i = 1; i < sz; ++i) {
        ret = ptr[sz - i - 1] + ret * x;
    }

    return ret;
} // LCOV_EXCL_LINE

expression horner_eval(std::initializer_list<expression>, const expression &);

std::array<expression, 2> ex_cmul(const std::array<expression, 2> &, const std::array<expression, 2> &);
std::array<expression, 2> ex_cinv(const std::array<expression, 2> &);

// Dictionary to map an integral exponent to the corresponding integral power of a complex expression.
using pow_dict_t = boost::unordered_flat_map<std::int8_t, std::array<expression, 2>>;

// Dictionary to map an expression "ex" to a a dictionary of integral powers of
// cos(ex) + im * sin(ex).
// NOTE: use std::map (rather than an unordered map) for the usual reason that comparison-based
// containers can perform better than hashing on account of the fact that comparison does not
// need to traverse the entire expression.
using trig_eval_dict_t = std::map<expression, pow_dict_t>;

std::array<expression, 2> ccpow(const expression &, trig_eval_dict_t &, std::int8_t);
std::array<expression, 2> pairwise_cmul(std::vector<std::array<expression, 2>> &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
