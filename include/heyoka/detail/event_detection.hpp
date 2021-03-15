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
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, T, bool>> &,
                          std::vector<std::tuple<std::uint32_t, T>> &, const std::vector<t_event<T>> &,
                          const std::vector<nt_event<T>> &, const std::vector<std::optional<std::pair<T, T>>> &, T,
                          const std::vector<T> &, std::uint32_t, std::uint32_t)
{
    static_assert(always_false_v<T>, "Unhandled type");
}

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, double, bool>> &,
                          std::vector<std::tuple<std::uint32_t, double>> &, const std::vector<t_event<double>> &,
                          const std::vector<nt_event<double>> &,
                          const std::vector<std::optional<std::pair<double, double>>> &, double,
                          const std::vector<double> &, std::uint32_t, std::uint32_t);

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, long double, bool>> &,
                          std::vector<std::tuple<std::uint32_t, long double>> &,
                          const std::vector<t_event<long double>> &, const std::vector<nt_event<long double>> &,
                          const std::vector<std::optional<std::pair<long double, long double>>> &, long double,
                          const std::vector<long double> &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

template <>
void taylor_detect_events(std::vector<std::tuple<std::uint32_t, mppp::real128, bool>> &,
                          std::vector<std::tuple<std::uint32_t, mppp::real128>> &,
                          const std::vector<t_event<mppp::real128>> &, const std::vector<nt_event<mppp::real128>> &,
                          const std::vector<std::optional<std::pair<mppp::real128, mppp::real128>>> &, mppp::real128,
                          const std::vector<mppp::real128> &, std::uint32_t, std::uint32_t);

#endif

} // namespace heyoka::detail

#endif
