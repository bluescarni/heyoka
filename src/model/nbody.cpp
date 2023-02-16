// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/model/nbody.hpp>

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
                // Body j is massive and it interacts mutually with body i
                // (which is also massive).
                // NOTE: the idea here is that we want to help the CSE process
                // when computing the Taylor decomposition. Thus, we try
                // to maximise the re-use of an expression with the goal
                // of having it simplified in the CSE.
                auto fac_j = Gconst * masses_vec[j] * r_m3;
                auto c_ij = -masses_vec[i] / masses_vec[j];

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
                auto fac = -Gconst * masses_vec[i] * r_m3;

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

    return retval;
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
