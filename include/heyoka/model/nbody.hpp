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
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

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
auto nbody_common_opts(std::uint32_t n, KwArgs &&...kw_args)
{
    if (n < 2u) {
        throw std::invalid_argument(
            fmt::format("Cannot construct an N-body system with N == {}: at least 2 bodies are needed", n));
    }

    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(),
                  "Unnamed arguments cannot be passed in the variadic pack to this function.");
    static_assert(!p.has_other_than(kw::Gconst, kw::masses),
                  "This function accepts only the 'Gconst' and 'masses' named arguments.");

    // G constant (defaults to 1).
    auto Gconst = [&p]() {
        if constexpr (p.has(kw::Gconst)) {
            return expression{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))};
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

    if (masses_vec.size() > n) {
        throw std::invalid_argument(fmt::format("In an N-body system the number of particles with mass ({}) cannot be "
                                                "greater than the total number of particles ({})",
                                                masses_vec.size(), n));
    }

    return std::tuple{n, std::move(Gconst), std::move(masses_vec)};
}

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> nbody_impl(std::uint32_t, const expression &,
                                                                            const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression nbody_energy_impl(std::uint32_t, const expression &, const std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>> np1body_impl(std::uint32_t, const expression &,
                                                                              const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression np1body_energy_impl(std::uint32_t, const expression &, const std::vector<expression> &);

} // namespace detail

inline constexpr auto nbody = [](std::uint32_t n, auto &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::nbody_impl, detail::nbody_common_opts(n, std::forward<decltype(kw_args)>(kw_args)...));
};

inline constexpr auto nbody_energy = [](std::uint32_t n, auto &&...kw_args) -> expression {
    return std::apply(detail::nbody_energy_impl,
                      detail::nbody_common_opts(n, std::forward<decltype(kw_args)>(kw_args)...));
};

inline constexpr auto np1body
    = [](std::uint32_t n, auto &&...kw_args) -> std::vector<std::pair<expression, expression>> {
    return std::apply(detail::np1body_impl, detail::nbody_common_opts(n, std::forward<decltype(kw_args)>(kw_args)...));
};

inline constexpr auto np1body_energy = [](std::uint32_t n, auto &&...kw_args) -> expression {
    return std::apply(detail::np1body_energy_impl,
                      detail::nbody_common_opts(n, std::forward<decltype(kw_args)>(kw_args)...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
