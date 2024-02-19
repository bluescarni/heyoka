// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Gravitational constant (defaults to 1).
    auto gconst = [&p]() {
        if constexpr (p.has(kw::gconst)) {
            return expression{p(kw::gconst)};
        } else {
            return 1_dbl;
        }
    }();

    // Length (defaults to 1).
    auto l = [&p]() {
        if constexpr (p.has(kw::length)) {
            return expression{p(kw::length)};
        } else {
            return 1_dbl;
        }
    }();

    return std::tuple{std::move(gconst), std::move(l)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> pendulum_impl(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression pendulum_energy_impl(const expression &, const expression &);

} // namespace detail

inline constexpr auto pendulum = [](const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::pendulum_impl, detail::pendulum_common_opts(kw_args...));
};

// NOTE: this returns a specific energy.
inline constexpr auto pendulum_energy = [](const auto &...kw_args) -> expression {
    return std::apply(detail::pendulum_energy_impl, detail::pendulum_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
