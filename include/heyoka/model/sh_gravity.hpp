// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_SH_GRAVITY_HPP
#define HEYOKA_MODEL_SH_GRAVITY_HPP

#include <array>
#include <cstdint>
#include <functional>
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

// Common options for the sh_gravity_*() functions.
template <typename... KwArgs>
auto sh_gravity_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Gravitational parameter.
    auto mu = expression(p(kw::mu));

    // Reference radius.
    auto a = expression(p(kw::a));

    // SH coefficients.
    auto sh_coefficients = heyoka::detail::ranges_to<std::vector<expression>>(p(kw::sh_coefficients));

    return std::tuple{std::move(mu), std::move(a), std::move(sh_coefficients)};
}

// Function used to get the S/C coefficients in the sh_gravity_*_impl() primitives.
using sh_gravity_sc_getter_t = std::function<std::array<expression, 2>(std::uint32_t, std::uint32_t)>;

[[nodiscard]] HEYOKA_DLL_PUBLIC expression sh_gravity_pot_impl(const std::array<expression, 3> &, std::uint32_t,
                                                               std::uint32_t, const expression &, const expression &,
                                                               const sh_gravity_sc_getter_t &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> sh_gravity_acc_impl(const std::array<expression, 3> &,
                                                                              std::uint32_t, std::uint32_t,
                                                                              const expression &, const expression &,
                                                                              const sh_gravity_sc_getter_t &);

// Factory function for an implementation of sh_gravity_sc_getter_t which fetches the coefficients from a flattened list
// of S/C coefficients beginning from n=2.
[[nodiscard]] HEYOKA_DLL_PUBLIC sh_gravity_sc_getter_t sh_gravity_sc_getter_from_list(const std::vector<expression> &);

} // namespace detail

// kwargs configuration for the common options of the sh_gravity_*() functions.
inline constexpr auto sh_gravity_kw_cfg
    = igor::config<kw::descr::constructible_from<expression, kw::mu, true>,
                   kw::descr::constructible_from<expression, kw::a, true>,
                   kw::descr::constructible_input_range<kw::sh_coefficients, expression, true>>{};

// NOTE: in these implementations we accept the kwargs as forwarding references in order to highlight that they cannot
// be reused in other invocations.
inline constexpr auto sh_gravity_pot = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m, KwArgs &&...kw_args) {
    const auto [mu, a, sh_coefficients] = detail::sh_gravity_common_opts(kw_args...);

    const auto sc_getter = sh_gravity_sc_getter_from_array(sh_coefficients);

    return detail::sh_gravity_pot_impl(xyz, n, m, mu, a, sc_getter);
};

inline constexpr auto sh_gravity_acct = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m, KwArgs &&...kw_args) {
    const auto [mu, a, sh_coefficients] = detail::sh_gravity_common_opts(kw_args...);

    const auto sc_getter = sh_gravity_sc_getter_from_array(sh_coefficients);

    return detail::sh_gravity_acc_impl(xyz, n, m, mu, a, sc_getter);
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
