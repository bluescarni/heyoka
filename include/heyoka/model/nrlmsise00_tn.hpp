// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_NRLMSISE00_TN_HPP
#define HEYOKA_MODEL_NRLMSISE00_TN_HPP

#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Macro to reduce typing when handling kwargs.
#define HEYOKA_MODEL_NRLMSISE00_KWARG(name)                                                                            \
    auto name = [&p]() {                                                                                               \
        if constexpr (p.has(kw::name) && std::constructible_from<expression, decltype(p(kw::name))>) {                 \
            return expression{p(kw::name)};                                                                            \
        } else {                                                                                                       \
            static_assert(heyoka::detail::always_false_v<KwArgs...>,                                                   \
                          "The '" #name "' keyword argument is necessary but either it was not provided, or it is of " \
                          "the wrong type");                                                                           \
        }                                                                                                              \
    }()

// Common options for the nrlmsise00_tn functions.
template <typename... KwArgs>
auto nrlmsise00_tn_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Geodetic coordinates. Mandatory.
    std::vector<expression> geodetic;
    geodetic.reserve(3);
    if constexpr (p.has(kw::geodetic)) {
        using geodetic_t = decltype(p(kw::geodetic));

        if constexpr (requires() {
                          requires std::ranges::input_range<geodetic_t>;
                          requires std::constructible_from<expression, std::ranges::range_reference_t<geodetic_t>>;
                      }) {
            for (auto &&v : p(kw::geodetic)) {
                geodetic.emplace_back(std::forward<decltype(v)>(v));
            }
        } else {
            static_assert(
                heyoka::detail::always_false_v<KwArgs...>,
                "The 'geodetic' keyword argument is of the wrong type (it must be an input range whose reference "
                "type can be used to construct an expression)");
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                      "The 'geodetic' keyword argument is necessary but it was not provided");
    }

    HEYOKA_MODEL_NRLMSISE00_KWARG(f107);
    HEYOKA_MODEL_NRLMSISE00_KWARG(f107a);
    HEYOKA_MODEL_NRLMSISE00_KWARG(ap);
    // NOTE: the time in this case is the fractional number of days elapsed since the last 1st of January
    // 00:00:00 UTC)
    HEYOKA_MODEL_NRLMSISE00_KWARG(time_expr);

    return std::tuple{std::move(geodetic), std::move(f107), std::move(f107a), std::move(ap), std::move(time_expr)};
}

#undef HEYOKA_MODEL_NRLMSISE00_KWARG

// This c++ function returns the symbolic expressions of the thermospheric density at a certain geodetic coordinate,
// having the f107a, f107, ap indexes and from a time expression returning the days elapsed since the last 1st of
// January 00:00:00.
HEYOKA_DLL_PUBLIC expression nrlmsise00_tn_impl(const std::vector<expression> &, const expression &, const expression &,
                                                const expression &, const expression &);

} // namespace detail

inline constexpr auto nrlmsise00_tn = [](const auto &...kw_args) -> expression {
    return std::apply(detail::nrlmsise00_tn_impl, detail::nrlmsise00_tn_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
