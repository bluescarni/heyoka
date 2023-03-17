// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
auto rotating_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");
    static_assert(!p.has_other_than(kw::omega), "This function accepts only the 'omega' named argument.");

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

} // namespace detail

inline constexpr auto rotating = [](auto &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::rotating_impl, detail::rotating_common_opts(std::forward<decltype(kw_args)>(kw_args)...));
};

inline constexpr auto rotating_energy = [](auto &&...kw_args) -> expression {
    return std::apply(detail::rotating_energy_impl,
                      detail::rotating_common_opts(std::forward<decltype(kw_args)>(kw_args)...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
