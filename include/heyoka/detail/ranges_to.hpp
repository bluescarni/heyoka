// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_RANGES_TO
#define HEYOKA_DETAIL_RANGES_TO

#include <initializer_list>
#include <ranges>
#include <utility>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: these are wrappers around std::ranges::to() to allow usage with std::initializer_list as input range.
template <typename C, typename R, typename... Args>
constexpr auto ranges_to(R &&r, Args &&...args)
    -> decltype(std::ranges::to<C>(std::forward<R>(r), std::forward<Args>(args)...))
{
    return std::ranges::to<C>(std::forward<R>(r), std::forward<Args>(args)...);
}

template <typename C, typename T, typename... Args>
constexpr auto ranges_to(std::initializer_list<T> ilist, Args &&...args)
    -> decltype(std::ranges::to<C>(std::ranges::subrange(ilist.begin(), ilist.end()), std::forward<Args>(args)...))
{
    return std::ranges::to<C>(std::ranges::subrange(ilist.begin(), ilist.end()), std::forward<Args>(args)...);
}

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
