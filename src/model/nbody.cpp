// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

std::vector<std::pair<expression, expression>> nbody_impl(std::uint32_t n, const expression &Gconst,
                                                          const std::vector<expression> &masses_vec)
{
    assert(n >= masses_vec.size());

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
        y_vars.emplace_back(fmt::format("y_{}", i));
        z_vars.emplace_back(fmt::format("z_{}", i));

        vx_vars.emplace_back(fmt::format("vx_{}", i));
        vy_vars.emplace_back(fmt::format("vy_{}", i));
        vz_vars.emplace_back(fmt::format("vz_{}", i));
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

    // Store the number of massive particles.
    const auto n_massive = masses_vec.size();

    // Create the dynamics for the massive particles.
    for (std::uint32_t i = 0; i < n_massive; ++i) {
        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Compute all the gravitational interactions involving
        // body i: that is, both the accelerations on i
        // due to the other massive bodies, and the accelerations
        // due to i on *all* the bodies.
        for (std::uint32_t j = i + 1u; j < n; ++j) {
            const auto diff_x = x_vars[j] - x_vars[i];
            const auto diff_y = y_vars[j] - y_vars[i];
            const auto diff_z = z_vars[j] - z_vars[i];

            const auto r_m3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});

            // If j is massive and masses_vec[j] is a non-zero number,
            // we compute the accelerations using a grouping
            // of the operations optimised for the common case
            // in which masses and G are numbers.
            const auto j_massive = j < n_massive;
            const auto opt_grouping = j_massive && std::holds_alternative<number>(masses_vec[j].value())
                                      && !is_zero(std::get<number>(masses_vec[j].value()));

            if (opt_grouping) {
                // NOTE: Gconst * masses_vec[j] and -masses_vec[i] / masses_vec[j]
                // will be constant folded into the numerical result, if masses
                // and Gconst are all numbers. This allows to reduce the number
                // of operations wrt the other branch.
                const auto fac_j = Gconst * masses_vec[j] * r_m3;
                const auto c_ij = -masses_vec[i] / masses_vec[j];

                // Acceleration exerted by j on i.
                x_acc[i].push_back(diff_x * fac_j);
                y_acc[i].push_back(diff_y * fac_j);
                z_acc[i].push_back(diff_z * fac_j);

                // Acceleration exerted by i on j.
                x_acc[j].push_back(x_acc[i].back() * c_ij);
                y_acc[j].push_back(y_acc[i].back() * c_ij);
                z_acc[j].push_back(z_acc[i].back() * c_ij);
            } else {
                const auto G_r_m3 = Gconst * r_m3;

                // Acceleration due to i on j.
                const auto fac_i = -masses_vec[i] * G_r_m3;
                x_acc[j].push_back(diff_x * fac_i);
                y_acc[j].push_back(diff_y * fac_i);
                z_acc[j].push_back(diff_z * fac_i);

                if (j_massive) {
                    // Body j is massive and it interacts mutually with body i
                    // (which is also massive).
                    const auto fac_j = masses_vec[j] * G_r_m3;

                    // Acceleration due to j on i.
                    x_acc[i].push_back(diff_x * fac_j);
                    y_acc[i].push_back(diff_y * fac_j);
                    z_acc[i].push_back(diff_z * fac_j);
                }
            }
        }

        // Add the expressions of the accelerations on body i to the system.
        retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
        retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
        retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
    }

    // Now the dynamics of the massless particles.
    // All the accelerations on the massless particles
    // have already been accumulated in the loop above.
    // We just need to perform the sums on x/y/z_acc.
    for (auto i = n_massive; i < n; ++i) {
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        retval.push_back(prime(vx_vars[i]) = sum(x_acc[i]));
        retval.push_back(prime(vy_vars[i]) = sum(y_acc[i]));
        retval.push_back(prime(vz_vars[i]) = sum(z_acc[i]));
    }

    return retval;
}

expression nbody_energy_impl(std::uint32_t n, const expression &Gconst, const std::vector<expression> &masses_vec)
{
    assert(n >= masses_vec.size());

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
        y_vars.emplace_back(fmt::format("y_{}", i));
        z_vars.emplace_back(fmt::format("z_{}", i));

        vx_vars.emplace_back(fmt::format("vx_{}", i));
        vy_vars.emplace_back(fmt::format("vy_{}", i));
        vz_vars.emplace_back(fmt::format("vz_{}", i));
    }

    // Store the number of massive particles.
    const auto n_massive = masses_vec.size();

    // The kinetic terms.
    std::vector<expression> kin;
    for (std::uint32_t i = 0; i < n_massive; ++i) {
        kin.push_back(masses_vec[i] * sum_sq({vx_vars[i], vy_vars[i], vz_vars[i]}));
    }

    // The potential terms.
    std::vector<expression> pot;
    for (std::uint32_t i = 0; i < n_massive; ++i) {
        for (std::uint32_t j = i + 1u; j < n_massive; ++j) {
            const auto diff_x = x_vars[j] - x_vars[i];
            const auto diff_y = y_vars[j] - y_vars[i];
            const auto diff_z = z_vars[j] - z_vars[i];

            pot.push_back(masses_vec[i] * masses_vec[j] * pow(sum_sq({diff_x, diff_y, diff_z}), -.5_dbl));
        }
    }

    return .5_dbl * sum(std::move(kin)) - Gconst * sum(std::move(pot));
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
