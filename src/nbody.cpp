// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail

{

std::vector<std::pair<expression, expression>> make_nbody_sys_fixed_masses(std::uint32_t n, number Gconst,
                                                                           std::vector<number> masses)
{
    assert(n >= 2u);

    if (masses.size() != n) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent sizes detected while creating an N-body system: the vector of masses has a size of "
            "{}, while the number of bodies is {}"_format(masses.size(), n));
    }

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(variable("x_{}"_format(i)));
        y_vars.emplace_back(variable("y_{}"_format(i)));
        z_vars.emplace_back(variable("z_{}"_format(i)));

        vx_vars.emplace_back(variable("vx_{}"_format(i)));
        vy_vars.emplace_back(variable("vy_{}"_format(i)));
        vz_vars.emplace_back(variable("vz_{}"_format(i)));
    }

    // Create the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    // The i-th element of x/y/z_acc contains the list of
    // accelerations on body i due to all the other bodies.
    std::vector<std::vector<expression>> x_acc;
    x_acc.resize(boost::numeric_cast<decltype(x_acc.size())>(n));
    auto y_acc = x_acc;
    auto z_acc = x_acc;

    // Detect if we are creating a restricted problem. In a restricted
    // problem, the first n particles have mass, the remaining ones do not.
    std::uint32_t n_fc_massive = 0, n_fc_massless = 0;

    // Determine the number of massive particles at the beginning
    // of the masses vector.
    auto it = masses.begin();
    while (it != masses.end() && !is_zero(*it)) {
        ++n_fc_massive;
        ++it;
    }

    // Determine the number of massless particles following
    // the first group of massive particles at the beginning
    // of the masses vector.
    while (it != masses.end() && is_zero(*it)) {
        ++n_fc_massless;
        ++it;
    }
    assert(n_fc_massive + n_fc_massless <= n);

    if (n_fc_massless != 0u && n_fc_massive + n_fc_massless == n) {
        // We have some massless particles, and the vector of masses
        // is divided into two parts: massive particles followed by
        // massless particles. Thus, we are in the restricted case.

        // Compute the accelerations exerted by the massive particles
        // on all particles.
        for (std::uint32_t i = 0; i < n_fc_massive; ++i) {
            // r' = v.
            retval.push_back(prime(x_vars[i]) = vx_vars[i]);
            retval.push_back(prime(y_vars[i]) = vy_vars[i]);
            retval.push_back(prime(z_vars[i]) = vz_vars[i]);

            // Compute the total acceleration on body i,
            // accumulating the result in x/y/z_acc. Part of the
            // calculation will be re-used to compute
            // the contribution from i to the total acceleration
            // on body j.
            for (std::uint32_t j = i + 1u; j < n; ++j) {
                auto diff_x = x_vars[j] - x_vars[i];
                auto diff_y = y_vars[j] - y_vars[i];
                auto diff_z = z_vars[j] - z_vars[i];

                auto r_m3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});
                if (j < n_fc_massive) {
                    // Body j is massive and it interacts mutually with body i.
                    // NOTE: the idea here is that we want to help the CSE process
                    // when computing the Taylor decomposition. Thus, we try
                    // to maximise the re-use of an expression with the goal
                    // of having it simplified in the CSE.
                    auto fac_j = expression{Gconst * masses[j]} * r_m3;
                    auto c_ij = expression{-masses[i] / masses[j]};

                    // Acceleration exerted by j on i.
                    x_acc[i].push_back(diff_x * fac_j);
                    y_acc[i].push_back(diff_y * fac_j);
                    z_acc[i].push_back(diff_z * fac_j);

                    // Acceleration exerted by i on j.
                    x_acc[j].push_back(x_acc[i].back() * c_ij);
                    y_acc[j].push_back(y_acc[i].back() * c_ij);
                    z_acc[j].push_back(z_acc[i].back() * c_ij);
                } else {
                    // Body j is massless, add the acceleration
                    // on it due to the massive body i.
                    auto fac = expression{-Gconst * masses[i]} * r_m3;

                    x_acc[j].push_back(diff_x * fac);
                    y_acc[j].push_back(diff_y * fac);
                    z_acc[j].push_back(diff_z * fac);
                }
            }

            // Add the expressions of the accelerations to the system.
            retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
            retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
            retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
        }

        // All the accelerations on the massless particles
        // have already been accumulated in the loop above.
        // We just need to perform the pairwise sums on x/y/z_acc.
        for (std::uint32_t i = n_fc_massive; i < n; ++i) {
            retval.push_back(prime(x_vars[i]) = vx_vars[i]);
            retval.push_back(prime(y_vars[i]) = vy_vars[i]);
            retval.push_back(prime(z_vars[i]) = vz_vars[i]);

            retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
            retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
            retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
        }
    } else {
        for (std::uint32_t i = 0; i < n; ++i) {
            // r' = v.
            retval.push_back(prime(x_vars[i]) = vx_vars[i]);
            retval.push_back(prime(y_vars[i]) = vy_vars[i]);
            retval.push_back(prime(z_vars[i]) = vz_vars[i]);

            // Compute the total acceleration on body i,
            // accumulating the result in x/y/z_acc. Part of the
            // calculation will be re-used to compute
            // the contribution from i to the total acceleration
            // on body j.
            for (std::uint32_t j = i + 1u; j < n; ++j) {
                auto diff_x = x_vars[j] - x_vars[i];
                auto diff_y = y_vars[j] - y_vars[i];
                auto diff_z = z_vars[j] - z_vars[i];

                auto r_m3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});
                if (is_zero(masses[j])) {
                    // NOTE: special-case for m_j = 0, so that
                    // we avoid a division by zero in the other branch.
                    auto fac = expression{-Gconst * masses[i]} * r_m3;

                    x_acc[j].push_back(diff_x * fac);
                    y_acc[j].push_back(diff_y * fac);
                    z_acc[j].push_back(diff_z * fac);
                } else {
                    // NOTE: the idea here is that we want to help the CSE process
                    // when computing the Taylor decomposition. Thus, we try
                    // to maximise the re-use of an expression with the goal
                    // of having it simplified in the CSE.
                    auto fac_j = expression{Gconst * masses[j]} * r_m3;
                    auto c_ij = expression{-masses[i] / masses[j]};

                    // Acceleration exerted by j on i.
                    x_acc[i].push_back(diff_x * fac_j);
                    y_acc[i].push_back(diff_y * fac_j);
                    z_acc[i].push_back(diff_z * fac_j);

                    // Acceleration exerted by i on j.
                    x_acc[j].push_back(x_acc[i].back() * c_ij);
                    y_acc[j].push_back(y_acc[i].back() * c_ij);
                    z_acc[j].push_back(z_acc[i].back() * c_ij);
                }
            }

            // Add the expressions of the accelerations to the system.
            retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
            retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
            retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
        }
    }

    return retval;
}

std::vector<std::pair<expression, expression>> make_nbody_sys_par_masses(std::uint32_t n, number Gconst,
                                                                         std::uint32_t n_massive)
{
    assert(n >= 2u);

    if (n_massive > n) {
        using namespace fmt::literals;

        throw std::invalid_argument("The number of massive bodies, {}, cannot be larger than the "
                                    "total number of bodies, {}"_format(n_massive, n));
    }

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(variable("x_{}"_format(i)));
        y_vars.emplace_back(variable("y_{}"_format(i)));
        z_vars.emplace_back(variable("z_{}"_format(i)));

        vx_vars.emplace_back(variable("vx_{}"_format(i)));
        vy_vars.emplace_back(variable("vy_{}"_format(i)));
        vz_vars.emplace_back(variable("vz_{}"_format(i)));
    }

    // Create the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    // The i-th element of x/y/z_acc contains the list of
    // accelerations on body i due to all the other bodies.
    std::vector<std::vector<expression>> x_acc;
    x_acc.resize(boost::numeric_cast<decltype(x_acc.size())>(n));
    auto y_acc = x_acc;
    auto z_acc = x_acc;

    // Compute the accelerations exerted by the massive particles
    // on all particles.
    for (std::uint32_t i = 0; i < n_massive; ++i) {
        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Compute the total acceleration on body i,
        // accumulating the result in x/y/z_acc. Part of the
        // calculation will be re-used to compute
        // the contribution from i to the total acceleration
        // on body j.
        for (std::uint32_t j = i + 1u; j < n; ++j) {
            auto diff_x = x_vars[j] - x_vars[i];
            auto diff_y = y_vars[j] - y_vars[i];
            auto diff_z = z_vars[j] - z_vars[i];

            auto r_m3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});
            if (j < n_massive) {
                // Body j is massive and it interacts mutually with body i.
                // NOTE: the idea here is that we want to help the CSE process
                // when computing the Taylor decomposition. Thus, we try
                // to maximise the re-use of an expression with the goal
                // of having it simplified in the CSE.
                auto fac_j = expression{Gconst} * par[j] * r_m3;
                auto c_ij = -par[i] / par[j];

                // Acceleration exerted by j on i.
                x_acc[i].push_back(diff_x * fac_j);
                y_acc[i].push_back(diff_y * fac_j);
                z_acc[i].push_back(diff_z * fac_j);

                // Acceleration exerted by i on j.
                x_acc[j].push_back(x_acc[i].back() * c_ij);
                y_acc[j].push_back(y_acc[i].back() * c_ij);
                z_acc[j].push_back(z_acc[i].back() * c_ij);
            } else {
                // Body j is massless, add the acceleration
                // on it due to the massive body i.
                auto fac = expression{-Gconst} * par[i] * r_m3;

                x_acc[j].push_back(diff_x * fac);
                y_acc[j].push_back(diff_y * fac);
                z_acc[j].push_back(diff_z * fac);
            }
        }

        // Add the expressions of the accelerations to the system.
        retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
        retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
        retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
    }

    // All the accelerations on the massless particles
    // have already been accumulated in the loop above.
    // We just need to perform the pairwise sums on x/y/z_acc.
    for (std::uint32_t i = n_massive; i < n; ++i) {
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
        retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
        retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
    }

    return retval;
}

} // namespace detail

} // namespace heyoka
