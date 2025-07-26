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
#include <tuple>

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

// NOTE: keep these outside the function to re-use in the Python exposition.
inline constexpr auto a_earth = 6378137.0;
inline constexpr auto b_earth = 6356752.314245;

template <bool WithNIters, typename... KwArgs>
auto cart2geo_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // The eccentricity squared. Optional. Defaults to the WGS84 value.
    const double ecc2 = p(kw::ecc2, 1 - (b_earth * b_earth / a_earth / a_earth));

    // Planet equatorial radius ('a' in the geodetic classical notation). Defaults to a_earth.
    const double R_eq = p(kw::R_eq, a_earth);

    if constexpr (WithNIters) {
        // Number of iterations. Optional. Defaults to 4 (guarantees an error below the cm level on the Earth).
        const auto n_iters = boost::numeric_cast<unsigned>(p(kw::n_iters, 4));

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

// kwargs config for cart2geo.
inline constexpr auto cart2geo_kw_cfg
    = igor::config<kw::descr::convertible_to<kw::ecc2, double>, kw::descr::convertible_to<kw::R_eq, double>,
                   kw::descr::integral<kw::n_iters>>{};

// kwargs config for geo2cart.
inline constexpr auto geo2cart_kw_cfg
    = igor::config<kw::descr::convertible_to<kw::ecc2, double>, kw::descr::convertible_to<kw::R_eq, double>>{};

// NOTE: in these implementations we accept the kwargs as forwarding references in order to highlight that they cannot
// be reused in other invocations.
inline constexpr auto cart2geo = []<typename... KwArgs>
    requires igor::validate<cart2geo_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::cart2geo_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::cart2geo_common_opts<true>(kw_args...)));
};

inline constexpr auto geo2cart = []<typename... KwArgs>
    requires igor::validate<geo2cart_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(const std::array<expression, 3> &xyz, KwArgs &&...kw_args) {
    return std::apply(detail::geo2cart_impl,
                      std::tuple_cat(std::make_tuple(std::cref(xyz)), detail::cart2geo_common_opts<false>(kw_args...)));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
