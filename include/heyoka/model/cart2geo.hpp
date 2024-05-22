// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_CART2GEO_HPP
#define HEYOKA_MODEL_CART2GEO_HPP

#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{
namespace detail
{

template <typename... KwArgs>
auto cart2geo_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Cartesian coordinates. Mandatory.
    // The kw::xyz argument must be a range of values from which
    // an expression can be constructed.
    std::vector<expression> xyz;
    if constexpr (p.has(kw::xyz)) {
        for (const auto &val : p(kw::xyz)) {
            xyz.emplace_back(val);
        }
    } else {
        static_assert(heyoka::detail::always_false_v<KwArgs...>,
                      "The 'xyz' keyword argument is necessary but it was not provided");
    }

    const double a_earth = 6378137.0;
    const double b_earth = 6356752.314245;
    // The eccentricity squared. Optional. Defaults to the WGS84 value (Earth)
    // The kw::ecc2 argument must be a double
    double ecc2 = 1 - b_earth * b_earth / a_earth / a_earth;
    if constexpr (p.has(kw::ecc2)) {
        ecc2 = p(kw::ecc2);
    }

    // Planet equatorial radius ('a' in the geodetic classical notation)
    // The kw::R_eq argument must be a double
    double R_eq = a_earth;
    if constexpr (p.has(kw::R_eq)) {
        R_eq = p(kw::R_eq);
    }

    // Number of iterations. Optional. Defaults to 4 (guarantees that in the thermosphere the error is below cm)
    unsigned n_iters = 4u;
    if constexpr (p.has(kw::n_iters)) {
        n_iters = boost::numeric_cast<unsigned>(p(kw::n_iters));
    }
    return std::tuple{std::move(xyz), ecc2, R_eq, n_iters};
}

// This c++ function returns the symbolic expressions of the geodetic coordinates (h,lat,lon) as a function of the Cartesian
// coordinates in the ECRF (Earth Centered Reference Frame)
HEYOKA_DLL_PUBLIC std::vector<expression> cart2geo_impl(const std::vector<expression> &, double, double, unsigned);
} // namespace detail

inline constexpr auto cart2geo = [](const auto &...kw_args) -> std::vector<expression> {
    return std::apply(detail::cart2geo_impl, detail::cart2geo_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

#endif
