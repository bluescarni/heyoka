// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_CR3BP_HPP
#define HEYOKA_MODEL_CR3BP_HPP

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
auto cr3bp_common_opts(KwArgs &&...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    expression mu{1e-3};
    if constexpr (p.has(kw::mu)) {
        mu = expression{std::forward<decltype(p(kw::mu))>(p(kw::mu))};
    }

    return std::tuple{std::move(mu)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> cr3bp_impl(const expression &);

HEYOKA_DLL_PUBLIC expression cr3bp_jacobi_impl(const expression &);

} // namespace detail

// NOTE: non-dimensional c3bp dynamics in the usual rotating (synodic) reference frame.
// Expressed in terms of canonical state variables.
inline constexpr auto cr3bp = [](auto &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::cr3bp_impl, detail::cr3bp_common_opts(std::forward<decltype(kw_args)>(kw_args)...));
};

inline constexpr auto cr3bp_jacobi = [](auto &&...kw_args) -> expression {
    return std::apply(detail::cr3bp_jacobi_impl,
                      detail::cr3bp_common_opts(std::forward<decltype(kw_args)>(kw_args)...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
