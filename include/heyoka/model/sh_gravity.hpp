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
#include <ranges>
#include <tuple>
#include <type_traits>
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

// Type trait to detect a 2-elements C array or std::array of any type.
template <typename>
struct is_array2_like : std::false_type {
};

template <typename T>
struct is_array2_like<T[2]> : std::true_type {
};

template <typename T>
struct is_array2_like<std::array<T, 2>> : std::true_type {
};

// Concept to detect if T is a possibly cvref-qualified array-like type whose perfectly-forwarded element type can be
// used to construct an expression.
template <typename T>
concept expression_array_ctible = requires(T val) { expression{std::forward_like<T>(val[0])}; };

// Concept to detect if R is an input range from which a list of C/S coefficients for a custom spherical harmonics
// gravity model can be constructed.
template <typename R>
concept sh_gravity_cs_input_range
    = std::ranges::input_range<R> && is_array2_like<std::remove_cvref_t<std::ranges::range_reference_t<R>>>::value
      && expression_array_ctible<std::ranges::range_reference_t<R>>;

// Kwarg descriptor for sh_gravity_cs_input_range.
template <auto NArg>
    requires igor::any_named_argument<NArg>
inline constexpr auto sh_gravity_cs_input_range_descr
    = igor::descr<NArg, []<typename U>() { return sh_gravity_cs_input_range<U>; }>{.required = true};

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
    auto cs_rng
        // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
        = p(kw::sh_coefficients) | std::views::transform([]<typename T>(T &&val) {
              return std::array{expression{std::forward_like<T>(val[0])}, expression{std::forward_like<T>(val[1])}};
          });
    auto sh_coefficients = std::vector(std::ranges::begin(cs_rng), std::ranges::end(cs_rng));

    return std::tuple{std::move(mu), std::move(a), std::move(sh_coefficients)};
}

// Function type used to get the C/S coefficients in the sh_gravity_*_impl() primitives.
using sh_gravity_cs_getter_t = std::function<std::array<expression, 2>(std::uint32_t, std::uint32_t)>;

[[nodiscard]] HEYOKA_DLL_PUBLIC expression sh_gravity_pot_impl(const std::array<expression, 3> &, std::uint32_t,
                                                               std::uint32_t, const expression &, const expression &,
                                                               const sh_gravity_cs_getter_t &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> sh_gravity_acc_impl(const std::array<expression, 3> &,
                                                                              std::uint32_t, std::uint32_t,
                                                                              const expression &, const expression &,
                                                                              const sh_gravity_cs_getter_t &);

[[nodiscard]] HEYOKA_DLL_PUBLIC sh_gravity_cs_getter_t
sh_gravity_cs_getter_from_list(const std::vector<std::array<expression, 2>> &);

} // namespace detail

// kwargs configuration for the common options of the sh_gravity_*() functions.
inline constexpr auto sh_gravity_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::mu, true>,
                                                       kw::descr::constructible_from<expression, kw::a, true>,
                                                       detail::sh_gravity_cs_input_range_descr<kw::sh_coefficients>>{};

// NOTE: in these implementations we accept the kwargs as forwarding references in order to highlight that they cannot
// be reused in other invocations.
inline constexpr auto sh_gravity_pot = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m, KwArgs &&...kw_args) {
    const auto [mu, a, sh_coefficients] = detail::sh_gravity_common_opts(kw_args...);

    const auto cs_getter = sh_gravity_cs_getter_from_list(sh_coefficients);

    return detail::sh_gravity_pot_impl(xyz, n, m, mu, a, cs_getter);
};

inline constexpr auto sh_gravity_acc = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m, KwArgs &&...kw_args) {
    const auto [mu, a, sh_coefficients] = detail::sh_gravity_common_opts(kw_args...);

    const auto cs_getter = sh_gravity_cs_getter_from_list(sh_coefficients);

    return detail::sh_gravity_acc_impl(xyz, n, m, mu, a, cs_getter);
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
