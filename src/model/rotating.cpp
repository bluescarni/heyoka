// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math/sum.hpp>
#include <heyoka/model/rotating.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

void rotating_check_omega(const std::vector<expression> &omega)
{
    if (!omega.empty() && omega.size() != 3u) {
        throw std::invalid_argument(fmt::format("In a rotating reference frame model the angular velocity must be a "
                                                "3-dimensional vector, but instead it is a {}-dimensional vector",
                                                omega.size()));
    }
}

} // namespace

std::vector<std::pair<expression, expression>> rotating_impl(const std::vector<expression> &omega)
{
    // Check the angular velocity vector.
    rotating_check_omega(omega);

    // Init the state variables,
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Accumulate the accelerations.
    std::vector<expression> acc_x, acc_y, acc_z;
    if (!omega.empty()) {
        // The components of the angular velocity.
        const auto &pe = omega[0];
        const auto &qe = omega[1];
        const auto &re = omega[2];

        // -(w x w x r) -> centripetal.
        // NOTE: pre-compute a few common subexpressions.
        const auto qe_x = qe * x;
        const auto re_x = re * x;
        const auto qe_y = qe * y;
        const auto re_z = re * z;
        acc_x.push_back(qe * qe_x);
        acc_x.push_back(re * re_x);
        acc_x.push_back(-(pe * qe_y));
        acc_x.push_back(-(pe * re_z));

        acc_y.push_back(pe * pe * y);
        acc_y.push_back(re * re * y);
        acc_y.push_back(-(pe * qe_x));
        acc_y.push_back(-(qe * re_z));

        acc_z.push_back(pe * pe * z);
        acc_z.push_back(qe * qe * z);
        acc_z.push_back(-(pe * re_x));
        acc_z.push_back(-(re * qe_y));

        // -(2 w x v) -> coriolis.
        acc_x.push_back(-2_dbl * (qe * vz - re * vy));
        acc_y.push_back(-2_dbl * (re * vx - pe * vz));
        acc_z.push_back(-2_dbl * (pe * vy - qe * vx));
    }

    // Create the equations of motion.
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(6u);

    ret.push_back(prime(x) = vx);
    ret.push_back(prime(y) = vy);
    ret.push_back(prime(z) = vz);
    ret.push_back(prime(vx) = sum(acc_x));
    ret.push_back(prime(vy) = sum(acc_y));
    ret.push_back(prime(vz) = sum(acc_z));

    return ret;
}

expression rotating_potential_impl(const std::vector<expression> &omega)
{
    // Check the angular velocity vector.
    rotating_check_omega(omega);

    // Init the position variables.
    auto [x, y, z] = make_vars("x", "y", "z");

    if (omega.empty()) {
        return 0_dbl;
    } else {
        // The components of the angular velocity.
        const auto &pe = omega[0];
        const auto &qe = omega[1];
        const auto &re = omega[2];

        const auto tmp = fix_nn(sum({pe * x, qe * y, re * z}));

        return 0.5_dbl
               * fix_nn(tmp * tmp
                        - fix_nn(sum({pow(pe, 2_dbl), pow(qe, 2_dbl), pow(re, 2_dbl)}))
                              * fix_nn(sum({pow(x, 2_dbl), pow(y, 2_dbl), pow(z, 2_dbl)})));
    }
}

expression rotating_energy_impl(const std::vector<expression> &omega)
{
    // Init the velocity variables.
    auto [vx, vy, vz] = make_vars("vx", "vy", "vz");

    return 0.5_dbl * fix_nn(sum({pow(vx, 2_dbl), pow(vy, 2_dbl), pow(vz, 2_dbl)})) + rotating_potential_impl(omega);
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
