// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <limits>
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

        throw std::invalid_argument(fmt::format(
            "Inconsistent sizes detected while creating an N-body system: the vector of masses has a size of "
            "{}, while the number of bodies is {}",
            masses.size(), n));
    }

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(variable(fmt::format("x_{}", i)));
        y_vars.emplace_back(variable(fmt::format("y_{}", i)));
        z_vars.emplace_back(variable(fmt::format("z_{}", i)));

        vx_vars.emplace_back(variable(fmt::format("vx_{}", i)));
        vy_vars.emplace_back(variable(fmt::format("vy_{}", i)));
        vz_vars.emplace_back(variable(fmt::format("vz_{}", i)));
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

    {
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
    }

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
        // We just need to perform the sums on x/y/z_acc.
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

std::vector<std::pair<expression, expression>> make_np1body_sys_fixed_masses(std::uint32_t n, number Gconst,
                                                                             std::vector<number> masses)
{
    // LCOV_EXCL_START
    assert(n >= 1u);

    if (n == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("The number of bodies specified for the creation of an (N+1)-body problem is too "
                                  "large, resulting in an overflow condition");
    }
    // LCOV_EXCL_STOP

    if (masses.size() != n + 1u) {
        throw std::invalid_argument(fmt::format(
            "Inconsistent sizes detected while creating an (N+1)-body system: the vector of masses has a size of "
            "{}, while the number of bodies is {} (the number of masses must be one more than the number of bodies)",
            masses.size(), n));
    }

    // Create the state variables.
    // NOTE: the body indices will be in the [1, n] range.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(variable(fmt::format("x_{}", i + 1u)));
        y_vars.emplace_back(variable(fmt::format("y_{}", i + 1u)));
        z_vars.emplace_back(variable(fmt::format("z_{}", i + 1u)));

        vx_vars.emplace_back(variable(fmt::format("vx_{}", i + 1u)));
        vy_vars.emplace_back(variable(fmt::format("vy_{}", i + 1u)));
        vz_vars.emplace_back(variable(fmt::format("vz_{}", i + 1u)));
    }

    // Create vectors containing r_i/(r_i)**3 for each body.
    std::vector<expression> x_r3, y_r3, z_r3;
    for (std::uint32_t i = 0; i < n; ++i) {
        auto rm3 = pow(sum_sq({x_vars[i], y_vars[i], z_vars[i]}), expression{-3. / 2});

        x_r3.push_back(x_vars[i] * rm3);
        y_r3.push_back(y_vars[i] * rm3);
        z_r3.push_back(z_vars[i] * rm3);
    }

    // Create the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    std::vector<expression> x_acc, y_acc, z_acc;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_acc.clear();
        y_acc.clear();
        z_acc.clear();

        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Add the acceleration due to the zero-th body.
        x_acc.push_back(expression(-Gconst * (masses[0] + masses[i + 1u])) * x_r3[i]);
        y_acc.push_back(expression(-Gconst * (masses[0] + masses[i + 1u])) * y_r3[i]);
        z_acc.push_back(expression(-Gconst * (masses[0] + masses[i + 1u])) * z_r3[i]);

        // Add the accelerations due to the other bodies and the non-intertial
        // reference frame.
        for (std::uint32_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }

            const auto j_gt_i = j > i;

            auto diff_x = j_gt_i ? x_vars[j] - x_vars[i] : x_vars[i] - x_vars[j];
            auto diff_y = j_gt_i ? y_vars[j] - y_vars[i] : y_vars[i] - y_vars[j];
            auto diff_z = j_gt_i ? z_vars[j] - z_vars[i] : z_vars[i] - z_vars[j];

            auto diff_rm3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});

            auto tmp_acc_x = j_gt_i ? x_r3[j] - diff_x * diff_rm3 : x_r3[j] + diff_x * diff_rm3;
            auto tmp_acc_y = j_gt_i ? y_r3[j] - diff_y * diff_rm3 : y_r3[j] + diff_y * diff_rm3;
            auto tmp_acc_z = j_gt_i ? z_r3[j] - diff_z * diff_rm3 : z_r3[j] + diff_z * diff_rm3;

            auto cur_mu = expression(-Gconst * masses[j + 1u]);

            x_acc.push_back(cur_mu * tmp_acc_x);
            y_acc.push_back(cur_mu * tmp_acc_y);
            z_acc.push_back(cur_mu * tmp_acc_z);
        }

        // Add the expressions of the accelerations to the system.
        retval.push_back(prime(vx_vars[i]) = sum(x_acc));
        retval.push_back(prime(vy_vars[i]) = sum(y_acc));
        retval.push_back(prime(vz_vars[i]) = sum(z_acc));
    }

    return retval;
}

std::vector<std::pair<expression, expression>> make_nbody_sys_par_masses(std::uint32_t n, number Gconst,
                                                                         std::uint32_t n_massive)
{
    assert(n >= 2u);

    if (n_massive > n) {
        using namespace fmt::literals;

        throw std::invalid_argument(fmt::format("The number of massive bodies, {}, cannot be larger than the "
                                                "total number of bodies, {}",
                                                n_massive, n));
    }

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(variable(fmt::format("x_{}", i)));
        y_vars.emplace_back(variable(fmt::format("y_{}", i)));
        z_vars.emplace_back(variable(fmt::format("z_{}", i)));

        vx_vars.emplace_back(variable(fmt::format("vx_{}", i)));
        vy_vars.emplace_back(variable(fmt::format("vy_{}", i)));
        vz_vars.emplace_back(variable(fmt::format("vz_{}", i)));
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
    // We just need to perform the sums on x/y/z_acc.
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
