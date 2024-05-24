// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <stdexcept>
#include <vector>

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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<expression> cart2geo_impl(const std::vector<expression> &xyz, double ecc2, double R_eq, unsigned n_iters)
{
    // Sanity checks.
    if (xyz.size() != 3u) {
        throw std::invalid_argument("The xyz argument value must have exactly the size 3");
    }
    if (!std::isfinite(ecc2) || ecc2 < 0.) {
        throw std::invalid_argument("The ecc2 argument must be finite and positive");
    }
    if (!std::isfinite(R_eq) || R_eq < 0.) {
        throw std::invalid_argument("The R_eq argument must be finite and positive");
    }
    if (n_iters == 0u) {
        throw std::invalid_argument("The n_iters argument must be strictly positive");
    }

    const expression lon = atan2(xyz[1], xyz[0]);
    const expression p = sqrt(pow(xyz[0], 2.) + pow(xyz[1], 2.));
    expression phi = atan(xyz[2] / (p * (1. - ecc2)));
    expression h, N;
    // we iterate to improve the solution
    for (auto i = 0u; i < n_iters; ++i) {
        N = R_eq * pow(1. - ecc2 * pow(sin(phi), 2.), -.5);
        h = p / cos(phi) - N;
        phi = atan(xyz[2] / (p * (1. - ecc2 * N / (N + h))));
    }
    return {h, phi, lon};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
