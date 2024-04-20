// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/model/cr3bp.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

void cr3bp_check_mu(const expression &mu)
{
    if (const auto *nptr = std::get_if<number>(&mu.value())) {
        std::visit(
            [](const auto &v) {
                using std::isfinite;

                if (!isfinite(v) || v <= 0 || v >= .5) {
                    throw std::invalid_argument(fmt::format("The 'mu' parameter in a CR3BP must be in the range (0, "
                                                            "0.5), but a value of {} was provided instead",
                                                            v));
                }
            },
            nptr->value());
    }
}

} // namespace

std::vector<std::pair<expression, expression>> cr3bp_impl(const expression &mu)
{
    cr3bp_check_mu(mu);

    // Init the state variables,
    auto [px, py, pz, x, y, z] = make_vars("px", "py", "pz", "x", "y", "z");

    // x - mu.
    const auto x_m_mu = x - mu;
    // x - mu + 1.
    const auto x_m_mu_p1 = x_m_mu + 1.;
    // y**2 + z**2.
    const auto y_p_z_2 = pow(y, 2_dbl) + pow(z, 2_dbl);
    // rp1**2.
    const auto rp1_2 = pow(x_m_mu, 2_dbl) + y_p_z_2;
    // rp2**2.
    const auto rp2_2 = pow(x_m_mu_p1, 2_dbl) + y_p_z_2;
    // (1 - mu) / rp1**3.
    const auto g1 = (1. - mu) * pow(rp1_2, -3. / 2);
    // mu / rp2**3.
    const auto g2 = mu * pow(rp2_2, -3. / 2);
    // g1 + g2.
    const auto g1_g2 = g1 + g2;

    const auto xdot = px + y;
    const auto ydot = py - x;
    const auto zdot = pz;
    const auto pxdot = py - g1 * x_m_mu - g2 * x_m_mu_p1;
    const auto pydot = -px - g1_g2 * y;
    const auto pzdot = -g1_g2 * z;

    return {prime(x) = xdot, prime(y) = ydot, prime(z) = zdot, prime(px) = pxdot, prime(py) = pydot, prime(pz) = pzdot};
}

expression cr3bp_jacobi_impl(const expression &mu)
{
    cr3bp_check_mu(mu);

    // Init the state variables,
    auto [px, py, pz, x, y, z] = make_vars("px", "py", "pz", "x", "y", "z");

    // x - mu.
    const auto x_m_mu = x - mu;
    // x - mu + 1.
    const auto x_m_mu_p1 = x_m_mu + 1.;
    // y**2 + z**2.
    const auto y_p_z_2 = pow(y, 2_dbl) + pow(z, 2_dbl);
    // rp1**2.
    const auto rp1_2 = pow(x_m_mu, 2_dbl) + y_p_z_2;
    // rp2**2.
    const auto rp2_2 = pow(x_m_mu_p1, 2_dbl) + y_p_z_2;
    // (1 - mu) / rp1.
    const auto g1 = (1. - mu) / sqrt(rp1_2);
    // mu / rp2.
    const auto g2 = mu / sqrt(rp2_2);

    const auto kin = 0.5 * (pow(px, 2_dbl) + pow(py, 2_dbl) + pow(pz, 2_dbl));

    return kin + y * px - x * py - g1 - g2;
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
