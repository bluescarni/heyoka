// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_RANGES_TO
#define HEYOKA_DETAIL_RANGES_TO

#include <concepts>
#include <initializer_list>
#include <ranges>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: these are wrappers around std::ranges::to() which add the following features:
//
// - invocability with an rvalue std::initializer_list as input,
// - invocability when the reference type of R is not convertible to the value type of C (but the value type of C must
//   be constructible from the reference type of R).
//
// NOTE: at the moment we are requiring both C and R to be input ranges.
template <typename C, typename R, typename... Args>
    requires std::ranges::input_range<C> && std::ranges::input_range<R>
constexpr decltype(auto) ranges_to(R &&r, Args &&...args)
{
    using C_value_t = std::ranges::range_value_t<C>;
    using R_ref_t = std::ranges::range_reference_t<R>;

    if constexpr (std::convertible_to<R_ref_t, C_value_t>) {
        return std::ranges::to<C>(std::forward<R>(r), std::forward<Args>(args)...);
    } else if constexpr (std::constructible_from<C_value_t, R_ref_t> && std::movable<C_value_t>) {
        return std::ranges::to<C>(
            std::forward<R>(r) | std::views::transform([]<typename T>(T &&x) { return C_value_t(std::forward<T>(x)); }),
            std::forward<Args>(args)...);
    } else {
        static_assert(always_false_v<C>,
                      "Unable to convert the range reference type to the container value type in ranges_to().");
    }
}

template <typename C, typename T, typename... Args>
constexpr auto ranges_to(std::initializer_list<T> ilist, Args &&...args)
    -> decltype(ranges_to<C>(std::ranges::subrange(ilist.begin(), ilist.end()), std::forward<Args>(args)...))
{
    return ranges_to<C>(std::ranges::subrange(ilist.begin(), ilist.end()), std::forward<Args>(args)...);
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
