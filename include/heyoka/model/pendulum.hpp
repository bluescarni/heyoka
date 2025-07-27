// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_PENDULUM_HPP
#define HEYOKA_MODEL_PENDULUM_HPP

#include <tuple>
#include <utility>
#include <vector>

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

template <typename... KwArgs>
auto pendulum_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Gravitational constant (defaults to 1).
    auto gconst = expression(p(kw::gconst, 1.));

    // Length (defaults to 1).
    auto l = expression(p(kw::length, 1.));

    return std::tuple{std::move(gconst), std::move(l)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> pendulum_impl(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression pendulum_energy_impl(const expression &, const expression &);

} // namespace detail

inline constexpr auto pendulum_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::gconst>,
                                                     kw::descr::constructible_from<expression, kw::length>>{};

inline constexpr auto pendulum = []<typename... KwArgs>
    requires igor::validate<pendulum_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::pendulum_impl, detail::pendulum_common_opts(kw_args...));
};

// NOTE: this returns a specific energy.
inline constexpr auto pendulum_energy = []<typename... KwArgs>
    requires igor::validate<pendulum_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::pendulum_energy_impl, detail::pendulum_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
