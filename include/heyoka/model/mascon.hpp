// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_MASCON_HPP
#define HEYOKA_MODEL_MASCON_HPP

#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/model/rotating.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

template <typename... KwArgs>
auto mascon_common_opts(const KwArgs &...kw_args)
{
    return std::tuple_cat(fixed_centres_common_opts(kw_args...), rotating_common_opts(kw_args...));
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> mascon_impl(const expression &,
                                                                             const std::vector<expression> &,
                                                                             const std::vector<expression> &,
                                                                             const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression mascon_energy_impl(const expression &, const std::vector<expression> &,
                                                const std::vector<expression> &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression mascon_potential_impl(const expression &, const std::vector<expression> &,
                                                   const std::vector<expression> &, const std::vector<expression> &);

} // namespace detail

inline constexpr auto mascon = [](const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::mascon_impl, detail::mascon_common_opts(kw_args...));
};

// NOTE: this is the specific energy.
inline constexpr auto mascon_energy = [](const auto &...kw_args) -> expression {
    return std::apply(detail::mascon_energy_impl, detail::mascon_common_opts(kw_args...));
};

// NOTE: this is the potential.
inline constexpr auto mascon_potential = [](const auto &...kw_args) -> expression {
    return std::apply(detail::mascon_potential_impl, detail::mascon_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
