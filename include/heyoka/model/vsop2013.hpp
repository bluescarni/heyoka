// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/detail/igor.hpp>
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
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // Threshold value.
    const auto thresh = static_cast<double>(p(kw::thresh, def_thresh));

    return std::tuple{std::move(time_expr), thresh};
}

HEYOKA_DLL_PUBLIC expression vsop2013_elliptic_impl(std::uint32_t, std::uint32_t, expression, double);
HEYOKA_DLL_PUBLIC std::vector<expression> vsop2013_cartesian_impl(std::uint32_t, expression, double);
HEYOKA_DLL_PUBLIC std::vector<expression> vsop2013_cartesian_icrf_impl(std::uint32_t, expression, double);

} // namespace detail

inline constexpr auto vsop2013_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                                                     kw::descr::convertible_to<kw::thresh, double>>{};

template <typename... KwArgs>
    requires igor::validate<vsop2013_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
expression vsop2013_elliptic(std::uint32_t pl_idx, std::uint32_t var_idx, KwArgs &&...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_elliptic_impl(pl_idx, var_idx, std::move(time_expr), thresh);
}

template <typename... KwArgs>
    requires igor::validate<vsop2013_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> vsop2013_cartesian(std::uint32_t pl_idx, KwArgs &&...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_cartesian_impl(pl_idx, std::move(time_expr), thresh);
}

template <typename... KwArgs>
    requires igor::validate<vsop2013_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::vector<expression> vsop2013_cartesian_icrf(std::uint32_t pl_idx, KwArgs &&...kw_args)
{
    auto [time_expr, thresh] = detail::vsop2013_common_opts(1e-9, kw_args...);

    return detail::vsop2013_cartesian_icrf_impl(pl_idx, std::move(time_expr), thresh);
}

HEYOKA_DLL_PUBLIC std::array<double, 10> get_vsop2013_mus();

} // namespace model

HEYOKA_END_NAMESPACE

#endif
