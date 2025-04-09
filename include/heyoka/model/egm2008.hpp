// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_EGM2008_HPP
#define HEYOKA_MODEL_EGM2008_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Default values of the gravitational parameter 'mu' and Earth radius 'a' for the egm2008_*() functions.
// NOTE: these are in SI units, taken from the official documentation of EGM2008.
inline constexpr double egm2008_default_mu = 3986004.415e8;
inline constexpr double egm2008_default_a = 6378136.3;

// Common options for the egm2008_*() functions.
template <typename... KwArgs>
auto egm2008_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    // Gravitational parameter.
    auto mu = [&p]() -> expression {
        if constexpr (p.has(kw::mu)) {
            return p(kw::mu);
        } else {
            return expression{egm2008_default_mu};
        }
    }();

    // Earth radius.
    auto a = [&]() -> expression {
        if constexpr (p.has(kw::a)) {
            return p(kw::a);
        } else {
            return expression{egm2008_default_a};
        }
    }();

    return std::tuple{std::move(mu), std::move(a)};
}

[[nodiscard]] HEYOKA_DLL_PUBLIC expression egm2008_pot_impl(const std::array<expression, 3> &, std::uint32_t,
                                                            std::uint32_t, const expression &, const expression &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> egm2008_acc_impl(const std::array<expression, 3> &,
                                                                           std::uint32_t, std::uint32_t,
                                                                           const expression &, const expression &);

} // namespace detail

inline constexpr auto egm2008_pot = []<typename... KwArgs>
    requires(!igor::has_unnamed_arguments<KwArgs...>())
(const std::array<expression, 3> &xyz, std::uint32_t n, std::uint32_t m, const KwArgs &...kw_args) {
    return std::apply(detail::egm2008_pot_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz), n, m), detail::egm2008_common_opts(kw_args...)));
};

inline constexpr auto egm2008_acc = []<typename... KwArgs>
    requires(!igor::has_unnamed_arguments<KwArgs...>())
(const std::array<expression, 3> &xyz, std::uint32_t n, std::uint32_t m, const KwArgs &...kw_args) {
    return std::apply(detail::egm2008_acc_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz), n, m), detail::egm2008_common_opts(kw_args...)));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
