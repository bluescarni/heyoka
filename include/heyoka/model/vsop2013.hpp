// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_VSOP2013_HPP
#define HEYOKA_MODEL_VSOP2013_HPP

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

// Common options for the vsop2013 functions.
template <typename... KwArgs>
auto vsop2013_common_opts(double def_thresh, const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = [&p]() -> expression {
        if constexpr (p.has(kw::time_expr)) {
            return p(kw::time_expr);
        } else {
            return heyoka::time;
        }
    }();

    // Threshold value.
    auto thresh = [&]() -> double {
        if constexpr (p.has(kw::thresh)) {
            return p(kw::thresh);
        } else {
            return def_thresh;
        }
    }();

    return std::tuple{std::move(time_expr), thresh};
}

HEYOKA_DLL_PUBLIC expression vsop2013_elliptic_impl(std::uint32_t, std::uint32_t, expression, double);
HEYOKA_DLL_PUBLIC std::vector<expression> vsop2013_cartesian_impl(std::uint32_t, expression, double);
HEYOKA_DLL_PUBLIC std::vector<expression> vsop2013_cartesian_icrf_impl(std::uint32_t, expression, double);

} // namespace detail

template <typename... KwArgs>
expression vsop2013_elliptic(std::uint32_t pl_idx, std::uint32_t var_idx, const KwArgs &...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_elliptic_impl(pl_idx, var_idx, std::move(time_expr), thresh);
}

template <typename... KwArgs>
std::vector<expression> vsop2013_cartesian(std::uint32_t pl_idx, const KwArgs &...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_cartesian_impl(pl_idx, std::move(time_expr), thresh);
}

template <typename... KwArgs>
std::vector<expression> vsop2013_cartesian_icrf(std::uint32_t pl_idx, const KwArgs &...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_cartesian_icrf_impl(pl_idx, std::move(time_expr), thresh);
}

HEYOKA_DLL_PUBLIC std::array<double, 10> get_vsop2013_mus();

} // namespace model

HEYOKA_END_NAMESPACE

#endif
