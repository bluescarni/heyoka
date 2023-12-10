// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_NBODY_HPP
#define HEYOKA_MODEL_NBODY_HPP

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

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
auto nbody_common_opts(std::uint32_t n, const KwArgs &...kw_args)
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
    } else {
        // If no masses are provided, fix all masses to 1.
        masses_vec.resize(boost::numeric_cast<decltype(masses_vec.size())>(n), 1_dbl);
    }

    return std::tuple{n, std::move(Gconst), std::move(masses_vec)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> nbody_impl(std::uint32_t, const expression &,
                                                                            const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression nbody_energy_impl(std::uint32_t, const expression &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression nbody_potential_impl(std::uint32_t, const expression &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> np1body_impl(std::uint32_t, const expression &,
                                                                              const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression np1body_energy_impl(std::uint32_t, const expression &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression np1body_potential_impl(std::uint32_t, const expression &, const std::vector<expression> &);

} // namespace detail

// NOTE: these return energies and potential energies.

inline constexpr auto nbody
    = [](std::uint32_t n, const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::nbody_impl, detail::nbody_common_opts(n, kw_args...));
};

inline constexpr auto nbody_energy = [](std::uint32_t n, const auto &...kw_args) -> expression {
    return std::apply(detail::nbody_energy_impl, detail::nbody_common_opts(n, kw_args...));
};

inline constexpr auto nbody_potential = [](std::uint32_t n, const auto &...kw_args) -> expression {
    return std::apply(detail::nbody_potential_impl, detail::nbody_common_opts(n, kw_args...));
};

inline constexpr auto np1body
    = [](std::uint32_t n, const auto &...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::np1body_impl, detail::nbody_common_opts(n, kw_args...));
};

inline constexpr auto np1body_energy = [](std::uint32_t n, const auto &...kw_args) -> expression {
    return std::apply(detail::np1body_energy_impl, detail::nbody_common_opts(n, kw_args...));
};

inline constexpr auto np1body_potential = [](std::uint32_t n, const auto &...kw_args) -> expression {
    return std::apply(detail::np1body_potential_impl, detail::nbody_common_opts(n, kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
