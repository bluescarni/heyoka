// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
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
    assert(n >= 2u);
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

expression nbody_energy_impl([[maybe_unused]] std::uint32_t n, const expression &Gconst,
                             const std::vector<expression> &masses_vec)
{
    assert(n >= 2u);
    assert(n >= masses_vec.size());

    // Store the number of massive particles.
    const auto n_massive = masses_vec.size();

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n_massive; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
        y_vars.emplace_back(fmt::format("y_{}", i));
        z_vars.emplace_back(fmt::format("z_{}", i));

        vx_vars.emplace_back(fmt::format("vx_{}", i));
        vy_vars.emplace_back(fmt::format("vy_{}", i));
        vz_vars.emplace_back(fmt::format("vz_{}", i));
    }

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

            // NOTE: this is not the most optimal grouping of terms,
            // since if both masses and G are constants, it would be
            // better to do the multiplication by G directly here.
            // However, the difference is just a single multiplication.
            pot.push_back(masses_vec[i] * masses_vec[j] / sqrt(sum_sq({diff_x, diff_y, diff_z})));
        }
    }

    return .5_dbl * sum(std::move(kin)) - Gconst * sum(std::move(pot));
}

std::vector<std::pair<expression, expression>> np1body_impl(std::uint32_t n, const expression &Gconst,
                                                            const std::vector<expression> &masses_vec)
{
    assert(n >= 2u);
    assert(n >= masses_vec.size());

    // Create the state variables.
    // NOTE: the zeroth body is **not** included in the state.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n - 1u; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i + 1u));
        y_vars.emplace_back(fmt::format("y_{}", i + 1u));
        z_vars.emplace_back(fmt::format("z_{}", i + 1u));

        vx_vars.emplace_back(fmt::format("vx_{}", i + 1u));
        vy_vars.emplace_back(fmt::format("vy_{}", i + 1u));
        vz_vars.emplace_back(fmt::format("vz_{}", i + 1u));
    }

    // Create vectors containing r_i/(r_i)**3 for each body
    // (except the zeroth).
    std::vector<expression> x_r3, y_r3, z_r3;
    for (std::uint32_t i = 0; i < n - 1u; ++i) {
        const auto rm3 = pow(sum_sq({x_vars[i], y_vars[i], z_vars[i]}), expression{-3. / 2});

        x_r3.push_back(x_vars[i] * rm3);
        y_r3.push_back(y_vars[i] * rm3);
        z_r3.push_back(z_vars[i] * rm3);
    }

    // Init the return value.
    std::vector<std::pair<expression, expression>> retval;

    // Accumulators for the accelerations on the bodies.
    std::vector<expression> x_acc, y_acc, z_acc;

    // Store the number of massive particles (including the zeroth particle).
    const auto n_massive = masses_vec.size();

    for (std::uint32_t i = 0; i < n - 1u; ++i) {
        x_acc.clear();
        y_acc.clear();
        z_acc.clear();

        // r' = v.
        retval.push_back(prime(x_vars[i]) = vx_vars[i]);
        retval.push_back(prime(y_vars[i]) = vy_vars[i]);
        retval.push_back(prime(z_vars[i]) = vz_vars[i]);

        // Add the acceleration due to the zero-th body.
        const auto m_0 = (n_massive > 0u) ? masses_vec[0] : 0_dbl;
        const auto m_i = (i + 1u < n_massive) ? masses_vec[i + 1u] : 0_dbl;
        const auto mu_0i = -Gconst * (m_0 + m_i);
        x_acc.push_back(mu_0i * x_r3[i]);
        y_acc.push_back(mu_0i * y_r3[i]);
        z_acc.push_back(mu_0i * z_r3[i]);

        // Add the accelerations due to the other bodies and the non-intertial
        // reference frame.
        for (std::uint32_t j = 0; n_massive > 0u && j < n_massive - 1u; ++j) {
            if (j == i) {
                continue;
            }

            const auto j_gt_i = j > i;

            const auto diff_x = j_gt_i ? x_vars[j] - x_vars[i] : x_vars[i] - x_vars[j];
            const auto diff_y = j_gt_i ? y_vars[j] - y_vars[i] : y_vars[i] - y_vars[j];
            const auto diff_z = j_gt_i ? z_vars[j] - z_vars[i] : z_vars[i] - z_vars[j];

            const auto diff_rm3 = pow(sum_sq({diff_x, diff_y, diff_z}), expression{-3. / 2});

            const auto tmp_acc_x = j_gt_i ? x_r3[j] - diff_x * diff_rm3 : x_r3[j] + diff_x * diff_rm3;
            const auto tmp_acc_y = j_gt_i ? y_r3[j] - diff_y * diff_rm3 : y_r3[j] + diff_y * diff_rm3;
            const auto tmp_acc_z = j_gt_i ? z_r3[j] - diff_z * diff_rm3 : z_r3[j] + diff_z * diff_rm3;

            const auto cur_mu = -Gconst * masses_vec[j + 1u];

            x_acc.push_back(cur_mu * tmp_acc_x);
            y_acc.push_back(cur_mu * tmp_acc_y);
            z_acc.push_back(cur_mu * tmp_acc_z);
        }

        // Add the expressions of the accelerations to the system.
        retval.push_back(prime(vx_vars[i]) = sum(std::move(x_acc)));
        retval.push_back(prime(vy_vars[i]) = sum(std::move(y_acc)));
        retval.push_back(prime(vz_vars[i]) = sum(std::move(z_acc)));
    }

    return retval;
}

expression np1body_energy_impl([[maybe_unused]] std::uint32_t n, const expression &Gconst,
                               const std::vector<expression> &masses_vec)
{
    assert(n >= 2u);
    assert(n >= masses_vec.size());

    // NOTE: if masses_vec is empty, then take a shortcut avoiding
    // divisions by zero and out-of-bounds conditions.
    if (masses_vec.empty()) {
        return 0_dbl;
    }

    // Store the number of massive particles (including the zeroth particle).
    const auto n_massive = masses_vec.size();

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (std::uint32_t i = 0; i < n_massive - 1u; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i + 1u));
        y_vars.emplace_back(fmt::format("y_{}", i + 1u));
        z_vars.emplace_back(fmt::format("z_{}", i + 1u));

        vx_vars.emplace_back(fmt::format("vx_{}", i + 1u));
        vy_vars.emplace_back(fmt::format("vy_{}", i + 1u));
        vz_vars.emplace_back(fmt::format("vz_{}", i + 1u));
    }

    // Compute position and velocity of the zeroth particle in the barycentric reference frame.
    std::vector<expression> u0_x_terms, u0_y_terms, u0_z_terms, ud0_x_terms, ud0_y_terms, ud0_z_terms,
        tot_mass_terms{masses_vec[0]};

    for (std::uint32_t i = 0; i < n_massive - 1u; ++i) {
        u0_x_terms.push_back(masses_vec[i + 1u] * x_vars[i]);
        u0_y_terms.push_back(masses_vec[i + 1u] * y_vars[i]);
        u0_z_terms.push_back(masses_vec[i + 1u] * z_vars[i]);

        ud0_x_terms.push_back(masses_vec[i + 1u] * vx_vars[i]);
        ud0_y_terms.push_back(masses_vec[i + 1u] * vy_vars[i]);
        ud0_z_terms.push_back(masses_vec[i + 1u] * vz_vars[i]);

        tot_mass_terms.push_back(masses_vec[i + 1u]);
    }

    const auto tot_mass = sum(std::move(tot_mass_terms));

    const auto u0_x = -sum(std::move(u0_x_terms)) / tot_mass;
    const auto u0_y = -sum(std::move(u0_y_terms)) / tot_mass;
    const auto u0_z = -sum(std::move(u0_z_terms)) / tot_mass;

    const auto ud0_x = -sum(std::move(ud0_x_terms)) / tot_mass;
    const auto ud0_y = -sum(std::move(ud0_y_terms)) / tot_mass;
    const auto ud0_z = -sum(std::move(ud0_z_terms)) / tot_mass;

    // The kinetic terms.
    std::vector<expression> kin{masses_vec[0] * sum_sq({ud0_x, ud0_y, ud0_z})};
    for (std::uint32_t i = 0; i < n_massive - 1u; ++i) {
        kin.push_back(masses_vec[i + 1u] * sum_sq({vx_vars[i] + ud0_x, vy_vars[i] + ud0_y, vz_vars[i] + ud0_z}));
    }

    // The potential terms.
    std::vector<expression> pot;
    // Add the potential between the zeroth body and the rest.
    for (std::uint32_t i = 0; i < n_massive - 1u; ++i) {
        pot.push_back(masses_vec[0] * masses_vec[i + 1u] / sqrt(sum_sq({x_vars[i], y_vars[i], z_vars[i]})));
    }
    // Add the mutual potentials.
    for (std::uint32_t i = 0; i < n_massive - 1u; ++i) {
        for (std::uint32_t j = i + 1u; j < n_massive - 1u; ++j) {
            const auto diff_x = x_vars[j] - x_vars[i];
            const auto diff_y = y_vars[j] - y_vars[i];
            const auto diff_z = z_vars[j] - z_vars[i];

            pot.push_back(masses_vec[i + 1u] * masses_vec[j + 1u] / sqrt(sum_sq({diff_x, diff_y, diff_z})));
        }
    }

    return .5_dbl * sum(std::move(kin)) - Gconst * sum(std::move(pot));
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
