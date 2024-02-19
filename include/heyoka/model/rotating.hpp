// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_ROTATING_HPP
#define HEYOKA_MODEL_ROTATING_HPP

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
auto rotating_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    std::vector<expression> omega;
    if constexpr (p.has(kw::omega)) {
        for (const auto &comp : p(kw::omega)) {
            omega.emplace_back(comp);
        }
    }

    return std::tuple{std::move(omega)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> rotating_impl(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression rotating_energy_impl(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression rotating_potential_impl(const std::vector<expression> &);

} // namespace detail

// NOTE: dynamics of a free particle in a reference frame rotating with uniform angular
// velocity omega. Accounts for the centrifugal and Coriolis accelerations (but not the
// Euler acceleration as omega is assumed to be constant).
inline constexpr auto rotating = [](const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::rotating_impl, detail::rotating_common_opts(kw_args...));
};

// NOTE: this returns a specific energy.
inline constexpr auto rotating_energy = [](const auto &...kw_args) -> expression {
    return std::apply(detail::rotating_energy_impl, detail::rotating_common_opts(kw_args...));
};

// NOTE: this is the generalised potential originating from the centrifugal
// and Coriolis accelerations.
inline constexpr auto rotating_potential = [](const auto &...kw_args) -> expression {
    return std::apply(detail::rotating_potential_impl, detail::rotating_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
