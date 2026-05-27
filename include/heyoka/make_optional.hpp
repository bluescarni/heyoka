// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MAKE_OPTIONAL_HPP
#define HEYOKA_MAKE_OPTIONAL_HPP

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Concept to detect if Args is a pack of exactly 1 type which:
//
// - after the removal of cvref qualifiers, is the same as std::optional<T>, and
// - can be used to construct a std::optional<T>.
//
// NOTE: the latter condition is essentially equivalent to requiring that std::optional<T> is copy or move
// constructible, depending on the exact type of Args.
template <typename T, typename... Args>
concept is_optional_passthrough
    = sizeof...(Args) == 1u && (... && std::same_as<std::optional<T>, std::remove_cvref_t<Args>>)
      && std::constructible_from<std::optional<T>, Args...>;

} // namespace detail

template <typename T, typename... Args>
concept can_make_optional = (!std::same_as<T, void>) && std::same_as<std::remove_cvref_t<T>, T>
                            && (detail::is_optional_passthrough<T, Args...> || std::constructible_from<T, Args...>);

// NOTE: this is similar in spirit to std::make_optional, with the added ability to pass through an existing
// std::optional<T> via copy/move.
template <typename T, typename... Args>
    requires can_make_optional<T, Args &&...>
std::optional<T> make_optional(Args &&...args)
{
    if constexpr (detail::is_optional_passthrough<T, Args &&...>) {
        return std::optional<T>(std::forward<Args>(args)...);
    } else {
        return std::optional<T>(std::in_place, std::forward<Args>(args)...);
    }
}

HEYOKA_END_NAMESPACE

#endif
