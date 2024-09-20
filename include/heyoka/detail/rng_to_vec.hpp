// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_RNG_TO_VEC_HPP
#define HEYOKA_DETAIL_RNG_TO_VEC_HPP

#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to convert an input range R into a vector.
// The value type is deduced from the reference type of R.
template <typename R>
auto rng_to_vec(R &&r)
{
    // Deduce the value type.
    using value_type = std::remove_cvref_t<std::ranges::range_reference_t<R>>;

    // Prepare the return value, reserving the appropriate
    // size if R is a sized range.
    std::vector<value_type> retval;
    if constexpr (std::ranges::sized_range<R>) {
        retval.reserve(static_cast<decltype(retval.size())>(std::ranges::size(r)));
    }

    // Add r's values into retval.
    for (auto &&val : r) {
        retval.push_back(std::forward<decltype(val)>(val));
    }

    return retval;
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
