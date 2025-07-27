// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_JB08_TN_HPP
#define HEYOKA_MODEL_JB08_TN_HPP

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
#define HEYOKA_MODEL_JB08_KWARG(name) auto name = expression(p(kw::name));

// Common options for the jb08_tn functions.
template <typename... KwArgs>
auto jb08_tn_common_opts(const KwArgs &...kw_args)
{
    using heyoka::detail::ranges_to;

    const igor::parser p{kw_args...};

    // Geodetic coordinates. Mandatory.
    auto geodetic = ranges_to<std::vector<expression>>(p(kw::geodetic));

    HEYOKA_MODEL_JB08_KWARG(f107a);
    HEYOKA_MODEL_JB08_KWARG(f107);
    HEYOKA_MODEL_JB08_KWARG(s107a);
    HEYOKA_MODEL_JB08_KWARG(s107);
    HEYOKA_MODEL_JB08_KWARG(m107a);
    HEYOKA_MODEL_JB08_KWARG(m107);
    HEYOKA_MODEL_JB08_KWARG(y107a);
    HEYOKA_MODEL_JB08_KWARG(y107);
    HEYOKA_MODEL_JB08_KWARG(dDstdT);
    // NOTE: the time in this case is the fractional number of days elapsed since the last 1st of January
    // 00:00:00 UTC).
    HEYOKA_MODEL_JB08_KWARG(time_expr);

    return std::tuple{std::move(geodetic), std::move(f107a),  std::move(f107),     std::move(s107a),
                      std::move(s107),     std::move(m107a),  std::move(m107),     std::move(y107a),
                      std::move(y107),     std::move(dDstdT), std::move(time_expr)};
}

#undef HEYOKA_MODEL_JB08_KWARG

HEYOKA_DLL_PUBLIC expression jb08_tn_impl(const std::vector<expression> &, const expression &, const expression &,
                                          const expression &, const expression &, const expression &,
                                          const expression &, const expression &, const expression &,
                                          const expression &, const expression &);

} // namespace detail

// Macro to reduce typing when handling kwargs descriptors.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define HEYOKA_MODEL_JB08_KWARG_DESCR(name) kw::descr::constructible_from<expression, kw::name, true>

inline constexpr auto jb08_tn_kw_cfg
    = igor::config<kw::descr::constructible_input_range<kw::geodetic, expression, true>,
                   HEYOKA_MODEL_JB08_KWARG_DESCR(f107a), HEYOKA_MODEL_JB08_KWARG_DESCR(f107),
                   HEYOKA_MODEL_JB08_KWARG_DESCR(s107a), HEYOKA_MODEL_JB08_KWARG_DESCR(s107),
                   HEYOKA_MODEL_JB08_KWARG_DESCR(m107a), HEYOKA_MODEL_JB08_KWARG_DESCR(m107),
                   HEYOKA_MODEL_JB08_KWARG_DESCR(y107a), HEYOKA_MODEL_JB08_KWARG_DESCR(y107),
                   HEYOKA_MODEL_JB08_KWARG_DESCR(dDstdT), HEYOKA_MODEL_JB08_KWARG_DESCR(time_expr)>{};

#undef HEYOKA_MODEL_JB08_KWARG_DESCR

inline constexpr auto jb08_tn = []<typename... KwArgs>
    requires igor::validate<jb08_tn_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::jb08_tn_impl, detail::jb08_tn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
