// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_CART2GEO_HPP
#define HEYOKA_MODEL_CART2GEO_HPP

#include <concepts>
#include <initializer_list>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

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

// NOTE: keep these outside the function to re-use in the Python exposition.
inline constexpr auto a_earth = 6378137.0;

inline constexpr auto b_earth = 6356752.314245;

template <typename... KwArgs>
auto cart2geo_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // The eccentricity squared. Optional. Defaults to the WGS84 value (Earth)
    // The kw::ecc2 argument must be a double
    double ecc2 = 1 - b_earth * b_earth / a_earth / a_earth;
    if constexpr (p.has(kw::ecc2)) {
        if constexpr (std::convertible_to<decltype(p(kw::ecc2)), double>) {
            ecc2 = static_cast<double>(p(kw::ecc2));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'ecc2' argument to cart2geo() is of the wrong type (it must be convertible to double).");
        }
    }

    // Planet equatorial radius ('a' in the geodetic classical notation)
    // The kw::R_eq argument must be a double
    double R_eq = a_earth;
    if constexpr (p.has(kw::R_eq)) {
        if constexpr (std::convertible_to<decltype(p(kw::R_eq)), double>) {
            R_eq = static_cast<double>(p(kw::R_eq));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'R_eq' argument to cart2geo() is of the wrong type (it must be convertible to double).");
        }
    }

    // Number of iterations. Optional. Defaults to 4 (guarantees that in the thermosphere the error is below cm)
    unsigned n_iters = 4u;
    if constexpr (p.has(kw::n_iters)) {
        if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::n_iters))>>) {
            n_iters = boost::numeric_cast<unsigned>(p(kw::n_iters));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'n_iters' argument to cart2geo() is of the wrong type (it must be of integral type).");
        }
    }

    return std::tuple{ecc2, R_eq, n_iters};
}

// This c++ function returns the symbolic expressions of the geodetic coordinates (h,lat,lon) as a function of the
// Cartesian coordinates in the ECRF (Earth Centered Reference Frame)
HEYOKA_DLL_PUBLIC std::vector<expression> cart2geo_impl(const std::vector<expression> &, double, double, unsigned);

// Implementation of the cart2geo function object.
// Annoyingly, we need two overloads to accept also std::initializer_list in input.
struct cart2geo_functor {
    template <typename R, typename... KwArgs>
        requires std::ranges::input_range<R> && std::constructible_from<expression, std::ranges::range_reference_t<R>>
    auto operator()(R &&r, const KwArgs &...kw_args) const
    {
        std::vector<expression> xyz;
        xyz.reserve(3);
        for (auto &&val : r) {
            xyz.emplace_back(std::forward<decltype(val)>(val));
        }

        return std::apply(cart2geo_impl, std::tuple_cat(std::tuple{std::move(xyz)}, cart2geo_common_opts(kw_args...)));
    }
    template <typename T, typename... KwArgs>
        requires std::constructible_from<expression, const T &>
    auto operator()(std::initializer_list<T> r, const KwArgs &...kw_args) const
    {
        return operator()(std::ranges::subrange(r.begin(), r.end()), kw_args...);
    }
};

} // namespace detail

inline constexpr auto cart2geo = detail::cart2geo_functor{};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
