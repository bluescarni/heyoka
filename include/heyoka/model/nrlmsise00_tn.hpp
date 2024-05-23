// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_NRLMSISE00_TN_HPP
#define HEYOKA_MODEL_NRLMSISE00_TN_HPP

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Common options for the nrlmsise00_tn functions.
template <typename... KwArgs>
auto nrlmsise00_tn_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    // Geodetic coordinates. Mandatory.
    std::vector<expression> geodetic;
    if constexpr (p.has(kw::geodetic)) {
        for (const auto &val : p(kw::geodetic)) {
            geodetic.emplace_back(val);
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'geodetic' keyword argument is necessary but it was not provided");
    }

    // f107a index. Mandatory.
    auto f107a = [&p]() -> expression {
        if constexpr (p.has(kw::f107a)) {
            return p(kw::f107a);
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'f107a' keyword argument is necessary but it was not provided");
        }
    }();

    // f107a index. Mandatory.
    auto f107 = [&p]() -> expression {
        if constexpr (p.has(kw::f107)) {
            return p(kw::f107);
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'f107' keyword argument is necessary but it was not provided");
        }
    }();

    // ap index. Mandatory.
    auto ap = [&p]() -> expression {
        if constexpr (p.has(kw::ap)) {
            return p(kw::ap);
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'ap' keyword argument is necessary but it was not provided");
        }
    }();

    // Time expression. Mandatory. (In this case the fractional number of days elapsed since the last 1st of January 00:00:00 UTC)
    auto doy_expr = [&p]() -> expression {
        if constexpr (p.has(kw::time)) {
            return p(kw::time);
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'time' keyword argument is necessary but it was not provided. Note that for thermonets this is expected to be an "
                          "expression of the heyoka::time returning the number of days elapsed since the last 1st of January 00:00:00.");
        }
    }();

    return std::tuple{std::move(geodetic), std::move(f107a), std::move(f107), std::move(ap), std::move(doy_expr)};
}

// This c++ function returns the symbolic expressions of the thermospheric density at a certain geodetic coordinate,
// having the f107a, f107, ap indexes and from a time expression returning the days elapsed since the last 1st of January 00:00:00.
HEYOKA_DLL_PUBLIC expression nrlmsise00_tn_impl(const std::vector<expression>&, const expression&, const expression&, const expression&,
                                                const expression&);

} // namespace detail

inline constexpr auto nrlmsise00_tn = [](const auto &...kw_args) -> expression {
    return std::apply(detail::nrlmsise00_tn_impl, detail::nrlmsise00_tn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
