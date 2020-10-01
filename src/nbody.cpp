// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka::detail
{

std::vector<std::pair<expression, expression>> make_nbody_sys_parametric_masses(std::uint32_t n, expression Gconst)
{
    assert(n >= 2u);

    // Create the state variables and the mass variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars, m_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        const auto i_str = li_to_string(i);

        x_vars.emplace_back(variable("x_" + i_str));
        y_vars.emplace_back(variable("y_" + i_str));
        z_vars.emplace_back(variable("z_" + i_str));

        vx_vars.emplace_back(variable("vx_" + i_str));
        vy_vars.emplace_back(variable("vy_" + i_str));
        vz_vars.emplace_back(variable("vz_" + i_str));

        m_vars.emplace_back(variable("m_" + i_str));
    }

    // Create the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    // NOTE: no need to check n, we already successfully created
    // vectors of size n above.
    std::vector<expression> x_acc(n, expression{number{0.}}), y_acc(x_acc), z_acc(x_acc);

    for (std::uint32_t i = 0; i < n; ++i) {
        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Compute the acceleration on the body i,
        // accumulating the result in x/y/z_acc.
        for (std::uint32_t j = i + 1u; j < n; ++j) {
            auto diff_x = x_vars[j] - x_vars[i];
            auto diff_y = y_vars[j] - y_vars[i];
            auto diff_z = z_vars[j] - z_vars[i];

            auto r_m3 = pow(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, expression{number{-3. / 2}});

            // Acceleration exerted by j on i.
            x_acc[i] += Gconst * m_vars[j] * diff_x * r_m3;
            y_acc[i] += Gconst * m_vars[j] * diff_y * r_m3;
            z_acc[i] += Gconst * m_vars[j] * diff_z * r_m3;

            // Acceleration exerted by i on j.
            x_acc[j] -= Gconst * m_vars[i] * diff_x * r_m3;
            y_acc[j] -= Gconst * m_vars[i] * diff_y * r_m3;
            z_acc[j] -= Gconst * m_vars[i] * diff_z * r_m3;
        }

        // Add the expressions of the accelerations to the system.
        retval.push_back(prime(vx_vars[i]) = x_acc[i]);
        retval.push_back(prime(vy_vars[i]) = y_acc[i]);
        retval.push_back(prime(vz_vars[i]) = z_acc[i]);

        // Add the equation for the mass.
        retval.push_back(prime(m_vars[i]) = expression{number{0.}});
    }

    return retval;
}

std::vector<std::pair<expression, expression>> make_nbody_sys_fixed_masses(std::uint32_t n, expression Gconst,
                                                                           std::vector<expression> masses)
{
    assert(n >= 2u);

    if (masses.size() != n) {
        throw std::invalid_argument(
            "Inconsistent sizes detected while creating an N-body system: the vector of masses has a size of "
            + std::to_string(masses.size()) + ", while the number of bodies is " + std::to_string(n));
    }

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        const auto i_str = li_to_string(i);

        x_vars.emplace_back(variable("x_" + i_str));
        y_vars.emplace_back(variable("y_" + i_str));
        z_vars.emplace_back(variable("z_" + i_str));

        vx_vars.emplace_back(variable("vx_" + i_str));
        vy_vars.emplace_back(variable("vy_" + i_str));
        vz_vars.emplace_back(variable("vz_" + i_str));
    }

    // Create the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    // NOTE: no need to check n, we already successfully created
    // vectors of size n above.
    std::vector<expression> x_acc(n, expression{number{0.}}), y_acc(x_acc), z_acc(x_acc);

    for (std::uint32_t i = 0; i < n; ++i) {
        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Compute the acceleration on the body i,
        // accumulating the result in x/y/z_acc.
        for (std::uint32_t j = i + 1u; j < n; ++j) {
            auto diff_x = x_vars[j] - x_vars[i];
            auto diff_y = y_vars[j] - y_vars[i];
            auto diff_z = z_vars[j] - z_vars[i];

            auto r_m3 = pow(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, expression{number{-3. / 2}});

            // Acceleration exerted by j on i.
            x_acc[i] += Gconst * masses[j] * diff_x * r_m3;
            y_acc[i] += Gconst * masses[j] * diff_y * r_m3;
            z_acc[i] += Gconst * masses[j] * diff_z * r_m3;

            // Acceleration exerted by i on j.
            // NOTE: do the negation on the masses, which
            // here are guaranteed to have numerical values.
            x_acc[j] += Gconst * -masses[i] * diff_x * r_m3;
            y_acc[j] += Gconst * -masses[i] * diff_y * r_m3;
            z_acc[j] += Gconst * -masses[i] * diff_z * r_m3;
        }

        // Add the expressions of the accelerations to the system.
        retval.push_back(prime(vx_vars[i]) = x_acc[i]);
        retval.push_back(prime(vy_vars[i]) = y_acc[i]);
        retval.push_back(prime(vz_vars[i]) = z_acc[i]);
    }

    return retval;
}

} // namespace heyoka::detail
