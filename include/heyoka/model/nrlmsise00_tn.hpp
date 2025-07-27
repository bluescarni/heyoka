// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_NRLMSISE00_TN_HPP
#define HEYOKA_MODEL_NRLMSISE00_TN_HPP

#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/ranges_to.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Macro to reduce typing when handling kwargs.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define HEYOKA_MODEL_NRLMSISE00_KWARG(name) auto name = expression(p(kw::name));

// Common options for the nrlmsise00_tn functions.
template <typename... KwArgs>
auto nrlmsise00_tn_common_opts(const KwArgs &...kw_args)
{
    using heyoka::detail::ranges_to;

    const igor::parser p{kw_args...};

    // Geodetic coordinates. Mandatory.
    auto geodetic = ranges_to<std::vector<expression>>(p(kw::geodetic));

    HEYOKA_MODEL_NRLMSISE00_KWARG(f107);
    HEYOKA_MODEL_NRLMSISE00_KWARG(f107a);
    HEYOKA_MODEL_NRLMSISE00_KWARG(ap);
    // NOTE: the time in this case is the fractional number of days elapsed since the last 1st of January
    // 00:00:00 UTC).
    HEYOKA_MODEL_NRLMSISE00_KWARG(time_expr);

    return std::tuple{std::move(geodetic), std::move(f107), std::move(f107a), std::move(ap), std::move(time_expr)};
}

#undef HEYOKA_MODEL_NRLMSISE00_KWARG

HEYOKA_DLL_PUBLIC expression nrlmsise00_tn_impl(const std::vector<expression> &, const expression &, const expression &,
                                                const expression &, const expression &);

} // namespace detail

// Macro to reduce typing when handling kwargs descriptors.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR(name) kw::descr::constructible_from<expression, kw::name, true>

inline constexpr auto nrlmsise00_tn_kw_cfg
    = igor::config<kw::descr::constructible_input_range<kw::geodetic, expression>,
                   HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR(f107), HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR(f107a),
                   HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR(ap), HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR(time_expr)>{};

#undef HEYOKA_MODEL_NRLMSISE00_KWARG_DESCR

inline constexpr auto nrlmsise00_tn = []<typename... KwArgs>
    requires igor::validate<nrlmsise00_tn_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::nrlmsise00_tn_impl, detail::nrlmsise00_tn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
