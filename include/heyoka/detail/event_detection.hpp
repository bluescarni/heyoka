// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EVENT_DETECTION_HPP
#define HEYOKA_DETAIL_EVENT_DETECTION_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>

namespace heyoka::detail
{

template <typename T>
inline T taylor_deduce_cooldown(T)
{
    static_assert(always_false_v<T>, "Unhandled type");
}

template <>
double taylor_deduce_cooldown(double);

template <>
long double taylor_deduce_cooldown(long double);

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 taylor_deduce_cooldown(mppp::real128);

#endif

template <typename T>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, T, bool, event_direction>> &,
                          std::vector<std::tuple<std::uint32_t, T, event_direction>> &,
                          const std::vector<detail::t_event_impl<T>> &, const std::vector<detail::nt_event_impl<T>> &,
                          const std::vector<std::optional<std::pair<T, T>>> &, T, const std::vector<T> &, std::uint32_t,
                          std::uint32_t)
{
    static_assert(always_false_v<T>, "Unhandled type");
}

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, double, bool, event_direction>> &,
                          std::vector<std::tuple<std::uint32_t, double, event_direction>> &,
                          const std::vector<detail::t_event_impl<double>> &,
                          const std::vector<detail::nt_event_impl<double>> &,
                          const std::vector<std::optional<std::pair<double, double>>> &, double,
                          const std::vector<double> &, std::uint32_t, std::uint32_t);

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, long double, bool, event_direction>> &,
                          std::vector<std::tuple<std::uint32_t, long double, event_direction>> &,
                          const std::vector<detail::t_event_impl<long double>> &,
                          const std::vector<detail::nt_event_impl<long double>> &,
                          const std::vector<std::optional<std::pair<long double, long double>>> &, long double,
                          const std::vector<long double> &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, mppp::real128, bool, event_direction>> &,
                          std::vector<std::tuple<std::uint32_t, mppp::real128, event_direction>> &,
                          const std::vector<detail::t_event_impl<mppp::real128>> &,
                          const std::vector<detail::nt_event_impl<mppp::real128>> &,
                          const std::vector<std::optional<std::pair<mppp::real128, mppp::real128>>> &, mppp::real128,
                          const std::vector<mppp::real128> &, std::uint32_t, std::uint32_t);

#endif

} // namespace heyoka::detail

#endif
