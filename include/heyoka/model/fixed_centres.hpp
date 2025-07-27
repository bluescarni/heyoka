// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/ranges_to.hpp>
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
    using heyoka::detail::ranges_to;

    const igor::parser p{kw_args...};

    // G constant (defaults to 1).
    auto Gconst = expression(p(kw::Gconst, 1.));

    // The vector of masses.
    std::vector<expression> masses_vec;
    if constexpr (p.has(kw::masses)) {
        masses_vec = ranges_to<std::vector<expression>>(p(kw::masses));
    }

    // The vector of positions.
    std::vector<expression> positions_vec;
    if constexpr (p.has(kw::positions)) {
        positions_vec = ranges_to<std::vector<expression>>(p(kw::positions));
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

inline constexpr auto fixed_centres_kw_cfg
    = igor::config<kw::descr::constructible_from<expression, kw::Gconst>,
                   kw::descr::constructible_input_range<kw::masses, expression>,
                   kw::descr::constructible_input_range<kw::positions, expression>>{};

inline constexpr auto fixed_centres = []<typename... KwArgs>
    requires igor::validate<fixed_centres_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::fixed_centres_impl, detail::fixed_centres_common_opts(kw_args...));
};

// NOTE: these return specific energy and potential.
inline constexpr auto fixed_centres_energy = []<typename... KwArgs>
    requires igor::validate<fixed_centres_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::fixed_centres_energy_impl, detail::fixed_centres_common_opts(kw_args...));
};

inline constexpr auto fixed_centres_potential = []<typename... KwArgs>
    requires igor::validate<fixed_centres_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::fixed_centres_potential_impl, detail::fixed_centres_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
