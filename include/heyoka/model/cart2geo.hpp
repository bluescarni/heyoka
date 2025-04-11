// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_CART2GEO_HPP
#define HEYOKA_MODEL_CART2GEO_HPP

#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>

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

template <bool WithNIters, typename... KwArgs>
auto cart2geo_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // The eccentricity squared. Optional. Defaults to the WGS84 value.
    // The kw::ecc2 argument must be a double.
    double ecc2 = 1 - (b_earth * b_earth / a_earth / a_earth);
    if constexpr (p.has(kw::ecc2)) {
        if constexpr (std::convertible_to<decltype(p(kw::ecc2)), double>) {
            ecc2 = static_cast<double>(p(kw::ecc2));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'ecc2' argument to cart2geo() is of the wrong type (it must be convertible to double).");
        }
    }

    // Planet equatorial radius ('a' in the geodetic classical notation).
    // The kw::R_eq argument must be a double.
    double R_eq = a_earth;
    if constexpr (p.has(kw::R_eq)) {
        if constexpr (std::convertible_to<decltype(p(kw::R_eq)), double>) {
            R_eq = static_cast<double>(p(kw::R_eq));
        } else {
            static_assert(heyoka::detail::always_false_v<KwArgs...>,
                          "The 'R_eq' argument to cart2geo() is of the wrong type (it must be convertible to double).");
        }
    }

    if constexpr (WithNIters) {
        // Number of iterations. Optional. Defaults to 4 (guarantees an error below the cm level on the Earth).
        unsigned n_iters = 4u;
        if constexpr (p.has(kw::n_iters)) {
            if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::n_iters))>>) {
                n_iters = boost::numeric_cast<unsigned>(p(kw::n_iters));
            } else {
                static_assert(
                    heyoka::detail::always_false_v<KwArgs...>,
                    "The 'n_iters' argument to cart2geo() is of the wrong type (it must be of integral type).");
            }
        }

        return std::tuple{ecc2, R_eq, n_iters};
    } else {
        return std::tuple{ecc2, R_eq};
    }
}

// This c++ function returns the symbolic expressions of the geodetic coordinates (h,lat,lon) as a function of the
// Cartesian coordinates in the ECEF (Earth-Centered Earth-Fixed reference Frame).
HEYOKA_DLL_PUBLIC std::array<expression, 3> cart2geo_impl(const std::array<expression, 3> &, double, double, unsigned);

// Inverse of cart2geo_impl().
HEYOKA_DLL_PUBLIC std::array<expression, 3> geo2cart_impl(const std::array<expression, 3> &, double, double);

} // namespace detail

inline constexpr auto cart2geo = []<typename... KwArgs>
    requires(!igor::has_unnamed_arguments<KwArgs...>())
(const std::array<expression, 3> &xyz, const KwArgs &...kw_args) {
    return std::apply(detail::cart2geo_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::cart2geo_common_opts<true>(kw_args...)));
};

inline constexpr auto geo2cart = []<typename... KwArgs>
    requires(!igor::has_unnamed_arguments<KwArgs...>())
(const std::array<expression, 3> &xyz, const KwArgs &...kw_args) {
    return std::apply(detail::geo2cart_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::cart2geo_common_opts<false>(kw_args...)));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
