// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_FIXED_CENTRES_HPP
#define HEYOKA_MODEL_FIXED_CENTRES_HPP

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
auto fixed_centres_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(),
                  "Unnamed arguments cannot be passed in the variadic pack to this function.");

    // G constant (defaults to 1).
    auto Gconst = [&p]() {
        if constexpr (p.has(kw::Gconst)) {
            return expression{p(kw::Gconst)};
        } else {
            return 1_dbl;
        }
    }();

    // The vector of masses.
    std::vector<expression> masses_vec;
    if constexpr (p.has(kw::masses)) {
        for (const auto &mass_value : p(kw::masses)) {
            masses_vec.emplace_back(mass_value);
        }
    }

    // The vector of positions.
    std::vector<expression> positions_vec;
    if constexpr (p.has(kw::positions)) {
        for (const auto &pos_value : p(kw::positions)) {
            positions_vec.emplace_back(pos_value);
        }
    }

    return std::tuple{std::move(Gconst), std::move(masses_vec), std::move(positions_vec)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>>
fixed_centres_impl(const expression &, const std::vector<expression> &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression fixed_centres_energy_impl(const expression &, const std::vector<expression> &,
                                                       const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression fixed_centres_potential_impl(const expression &, const std::vector<expression> &,
                                                          const std::vector<expression> &);

} // namespace detail

inline constexpr auto fixed_centres = [](const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::fixed_centres_impl, detail::fixed_centres_common_opts(kw_args...));
};

// NOTE: these return specific energy and potential.

inline constexpr auto fixed_centres_energy = [](const auto &...kw_args) -> expression {
    return std::apply(detail::fixed_centres_energy_impl, detail::fixed_centres_common_opts(kw_args...));
};

inline constexpr auto fixed_centres_potential = [](const auto &...kw_args) -> expression {
    return std::apply(detail::fixed_centres_potential_impl, detail::fixed_centres_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
