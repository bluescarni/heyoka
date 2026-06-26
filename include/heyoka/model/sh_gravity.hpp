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
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

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

// Type-trait to detect if T is an integral or a std::optional of an integral.
template <typename T>
struct is_optional_integral : std::is_integral<T> {
};

template <typename T>
struct is_optional_integral<std::optional<T>> : std::is_integral<T> {
};

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

    // Maximum degree/order. These can optionally be provided - if they are, only a subset of the spherical harmonics
    // model is used, otherwise the full model is used.
    const auto max_degree_order_factory = [&p]<auto K>() -> std::optional<std::uint32_t> {
        if constexpr (p.has(K)) {
            if constexpr (std::is_integral_v<std::remove_cvref_t<decltype(p(K))>>) {
                return boost::numeric_cast<std::uint32_t>(p(K));
            } else {
                if (p(K)) {
                    return boost::numeric_cast<std::uint32_t>(*p(K));
                }
            }
        }
        return {};
    };
    auto max_degree = max_degree_order_factory.template operator()<kw::max_degree>();
    auto max_order = max_degree_order_factory.template operator()<kw::max_order>();

    return std::tuple{std::move(mu), std::move(a), std::move(sh_coefficients), std::move(max_degree),
                      std::move(max_order)};
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

[[nodiscard]] HEYOKA_DLL_PUBLIC std::pair<sh_gravity_cs_getter_t, std::uint32_t>
sh_gravity_cs_getter_from_list(const std::vector<std::array<expression, 2>> &);

[[nodiscard]] HEYOKA_DLL_PUBLIC std::pair<std::uint32_t, std::uint32_t>
sh_gravity_resolve_n_m(const std::optional<std::uint32_t> &, const std::optional<std::uint32_t> &, std::uint32_t);

} // namespace detail

// kwargs configuration for the common options of the sh_gravity_*() functions.
inline constexpr auto sh_gravity_kw_cfg = igor::config<
    kw::descr::constructible_from<expression, kw::mu, true>, kw::descr::constructible_from<expression, kw::a, true>,
    igor::descr<kw::max_degree,
                []<typename U>() { return detail::is_optional_integral<std::remove_cvref_t<U>>::value; }>{},
    igor::descr<kw::max_order,
                []<typename U>() { return detail::is_optional_integral<std::remove_cvref_t<U>>::value; }>{},
    igor::descr<kw::sh_coefficients, []<typename U>() { return detail::sh_gravity_cs_input_range<U>; }>{.required
                                                                                                        = true}>{};

// NOTE: in these implementations we accept the kwargs as forwarding references in order to highlight that they cannot
// be reused in other invocations.
inline constexpr auto sh_gravity_pot = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    // Parse the common options.
    const auto [mu, a, sh_coefficients, max_degree, max_order] = detail::sh_gravity_common_opts(kw_args...);

    // Build the cs_getter and infer the maximum degree from the list of C/S coefficients.
    const auto [cs_getter, max_cs_n] = detail::sh_gravity_cs_getter_from_list(sh_coefficients);

    // Compute the degree/order values to pass to the impl functions.
    const auto [n, m] = detail::sh_gravity_resolve_n_m(max_degree, max_order, max_cs_n);

    return detail::sh_gravity_pot_impl(xyz, n, m, mu, a, cs_getter);
};

inline constexpr auto sh_gravity_acc = []<typename... KwArgs>
    requires igor::validate<sh_gravity_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    // Parse the common options.
    const auto [mu, a, sh_coefficients, max_degree, max_order] = detail::sh_gravity_common_opts(kw_args...);

    // Build the cs_getter and infer the maximum degree from the list of C/S coefficients.
    const auto [cs_getter, max_cs_n] = detail::sh_gravity_cs_getter_from_list(sh_coefficients);

    // Compute the degree/order values to pass to the impl functions.
    const auto [n, m] = detail::sh_gravity_resolve_n_m(max_degree, max_order, max_cs_n);

    return detail::sh_gravity_acc_impl(xyz, n, m, mu, a, cs_getter);
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
