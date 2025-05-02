// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_UDF_SPLIT_HPP
#define HEYOKA_DETAIL_UDF_SPLIT_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <cassert>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// This function accepts in input an expression e assumed to contain a function.
// If the function contained in e is of type UDF, then the function will be
// transformed in a sequence of nested invocations of that same function that
// will have no more than 'split' arguments per invocation. Otherwise, e will
// be returned unchanged.
//
// It is assumed that the function being splitted is generally associative
// (e.g., sum, prod, ...), so that reorganising it in a sequence of nested invocations
// will produce a mathematically-equivalent result. It is also assumed that
// UDF(arg) == arg.
//
// This helper is intended to re-organise long sums and products into a nested
// sequence of shorter sums and products.
//
// NOTE: perhaps in the future we can consider switching to a non-recursive implementation.
//
// NOLINTNEXTLINE(misc-no-recursion)
template <typename UDF>
expression udf_split(const expression &e, std::uint32_t split)
{
    assert(split >= 2u);
    assert(std::holds_alternative<func>(e.value()));

    const auto *udf_ptr = std::get<func>(e.value()).template extract<UDF>();

    // NOTE: return 'e' unchanged if it is not of type UDF,
    // or if it is of type UDF but it does not need to be split.
    // The latter condition is also used to terminate the
    // recursion.
    if (udf_ptr == nullptr || udf_ptr->args().size() <= split) {
        return e;
    }

    // NOTE: ret_seq will be a list of UDFs each containing 'split' terms.
    // tmp is a temporary vector used to accumulate the arguments for each
    // UDF in ret_seq.
    std::vector<expression> ret_seq, tmp;
    for (const auto &arg : udf_ptr->args()) {
        tmp.push_back(arg);

        if (tmp.size() == split) {
            ret_seq.emplace_back(func{UDF(std::move(tmp))});

            // NOTE: tmp is practically guaranteed to be empty, but let's
            // be paranoid.
            tmp.clear();
        }
    }

    // NOTE: tmp is not empty if 'split' does not divide
    // exactly udf_ptr->args().size(). In such a case, we need to do the
    // last iteration manually.
    if (!tmp.empty()) {
        // NOTE: contrary to the previous loop, here we could
        // in principle end up creating a UDF with only one
        // term. In such a case, return arg directly.
        if (tmp.size() == 1u) {
            ret_seq.push_back(std::move(tmp[0]));
        } else {
            ret_seq.emplace_back(func{UDF(std::move(tmp))});
        }
    }

    // Recurse to split further, if needed.
    return udf_split<UDF>(expression{func{UDF(std::move(ret_seq))}}, split);
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
