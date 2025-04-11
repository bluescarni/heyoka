// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <stdexcept>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/atan.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/model/cart2geo.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void common_cart2geo_checks(double ecc2, double R_eq, unsigned n_iters)
{
    if (!std::isfinite(ecc2) || ecc2 < 0.) [[unlikely]] {
        throw std::invalid_argument("The ecc2 argument must be finite and non-negative");
    }
    if (!std::isfinite(R_eq) || R_eq <= 0.) [[unlikely]] {
        throw std::invalid_argument("The R_eq argument must be finite and positive");
    }
    if (n_iters == 0u) [[unlikely]] {
        throw std::invalid_argument("The n_iters argument must be strictly positive");
    }
}

} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> cart2geo_impl(const std::array<expression, 3> &xyz, double ecc2, double R_eq,
                                        unsigned n_iters)
{
    // Sanity checks.
    common_cart2geo_checks(ecc2, R_eq, n_iters);

    const expression lon = atan2(xyz[1], xyz[0]);
    const expression p = sqrt(pow(xyz[0], 2.) + pow(xyz[1], 2.));
    expression phi = atan(xyz[2] / (p * (1. - ecc2)));
    expression h, N;
    // We iterate to improve the solution.
    for (auto i = 0u; i < n_iters; ++i) {
        N = R_eq * pow(1. - ecc2 * pow(sin(phi), 2.), -.5);
        h = p / cos(phi) - N;
        phi = atan(xyz[2] / (p * (1. - ecc2 * N / (N + h))));
    }
    return {h, phi, lon};
}

// NOTE: see https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> geo2cart_impl(const std::array<expression, 3> &geo, double ecc2, double R_eq)
{
    // Sanity checks.
    common_cart2geo_checks(ecc2, R_eq, 1u);

    // Fetch the geodetic coordinates.
    const auto &[h, phi, lon] = geo;

    // Compute N.
    const auto cos_phi = cos(phi), sin_phi = sin(phi);
    const auto N = R_eq / sqrt(1. - ecc2 * pow(sin_phi, 2.));

    // Assemble the return value.
    const auto Nph_cphi = (N + h) * cos_phi;
    const auto x = Nph_cphi * cos(lon);
    const auto y = Nph_cphi * sin(lon);
    const auto z = ((1. - ecc2) * N + h) * sin_phi;

    return {x, y, z};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
