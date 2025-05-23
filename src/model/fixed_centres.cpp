// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/fixed_centres.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

void fixed_centres_check_masses_pos(const std::vector<expression> &masses, const std::vector<expression> &positions)
{
    const auto pos_size = positions.size();

    if (pos_size % 3u != 0u) {
        throw std::invalid_argument(fmt::format(
            "In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is {}",
            pos_size));
    }

    const auto masses_size = masses.size();

    if (pos_size / 3u != masses_size) {
        throw std::invalid_argument(fmt::format(
            "In a fixed centres system the number of masses ({}) differs from the number of position vectors ({})",
            masses_size, pos_size / 3u));
    }
}

} // namespace

std::vector<std::pair<expression, expression>>
fixed_centres_impl(const expression &G, const std::vector<expression> &masses, const std::vector<expression> &positions)
{
    // Check masses/positions consistency.
    fixed_centres_check_masses_pos(masses, positions);

    // Compute the number of masses.
    const auto n_masses = masses.size();

    // Init the state variables,
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Accumulate the accelerations.
    std::vector<expression> acc_x, acc_y, acc_z;
    acc_x.reserve(n_masses);
    acc_y.reserve(n_masses);
    acc_z.reserve(n_masses);

    for (decltype(masses.size()) i = 0; i < n_masses; ++i) {
        const auto diff_x = positions[3u * i] - x;
        const auto diff_y = positions[(3u * i) + 1u] - y;
        const auto diff_z = positions[(3u * i) + 2u] - z;

        const auto dist2 = sum({pow(diff_x, 2_dbl), pow(diff_y, 2_dbl), pow(diff_z, 2_dbl)});
        const auto Mrm3 = masses[i] * pow(dist2, -1.5);

        acc_x.push_back(diff_x * Mrm3);
        acc_y.push_back(diff_y * Mrm3);
        acc_z.push_back(diff_z * Mrm3);
    }

    // Create the equations of motion.
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(6u);

    ret.push_back(prime(x) = vx);
    ret.push_back(prime(y) = vy);
    ret.push_back(prime(z) = vz);
    ret.push_back(prime(vx) = G * sum(acc_x));
    ret.push_back(prime(vy) = G * sum(acc_y));
    ret.push_back(prime(vz) = G * sum(acc_z));

    return ret;
}

expression fixed_centres_potential_impl(const expression &G, const std::vector<expression> &masses,
                                        const std::vector<expression> &positions)
{
    // Check masses/positions consistency.
    fixed_centres_check_masses_pos(masses, positions);

    // Compute the number of masses.
    const auto n_masses = masses.size();

    // Init the position variables,
    auto [x, y, z] = make_vars("x", "y", "z");

    // Accumulate the potential.
    std::vector<expression> pot;
    pot.reserve(n_masses);

    for (decltype(masses.size()) i = 0; i < n_masses; ++i) {
        const auto diff_x = positions[3u * i] - x;
        const auto diff_y = positions[(3u * i) + 1u] - y;
        const auto diff_z = positions[(3u * i) + 2u] - z;

        auto dist = sqrt(sum({pow(diff_x, 2_dbl), pow(diff_y, 2_dbl), pow(diff_z, 2_dbl)}));

        pot.push_back(masses[i] / dist);
    }

    return -G * sum(pot);
}

expression fixed_centres_energy_impl(const expression &G, const std::vector<expression> &masses,
                                     const std::vector<expression> &positions)
{
    // Init the velocity variables,
    auto [vx, vy, vz] = make_vars("vx", "vy", "vz");

    // Kinetic energy.
    auto kin = 0.5_dbl * sum({pow(vx, 2_dbl), pow(vy, 2_dbl), pow(vz, 2_dbl)});

    return kin + fixed_centres_potential_impl(G, masses, positions);
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
