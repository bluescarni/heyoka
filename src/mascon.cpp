// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/mascon.hpp>
#include <heyoka/math.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

std::vector<std::pair<expression, expression>>
make_mascon_system_impl(expression Gconst, std::vector<std::vector<expression>> mascon_points,
                        std::vector<expression> mascon_masses, expression pe, expression qe, expression re)
{
    // 3 - Create the return value.
    std::vector<std::pair<expression, expression>> retval;
    // 4 - Main code
    auto dim = std::size(mascon_masses);
    auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");
    // Assemble the contributions to the x/y/z accelerations from each mass.
    std::vector<expression> x_acc, y_acc, z_acc;
    // Assembling the r.h.s.
    // FIRST: the acceleration due to the mascon points
    for (decltype(dim) i = 0; i < dim; ++i) {
        auto x_masc = mascon_points[i][0];
        auto y_masc = mascon_points[i][1];
        auto z_masc = mascon_points[i][2];
        auto m_masc = mascon_masses[i];
        auto xdiff = (x - x_masc);
        auto ydiff = (y - y_masc);
        auto zdiff = (z - z_masc);
        auto r2 = sum({square(xdiff), square(ydiff), square(zdiff)});
        auto common_factor = -Gconst * m_masc * pow(r2, expression{-3. / 2.});
        x_acc.push_back(common_factor * xdiff);
        y_acc.push_back(common_factor * ydiff);
        z_acc.push_back(common_factor * zdiff);
    }
    // SECOND: centripetal and Coriolis
    // w x w x r
    auto centripetal_x = -qe * qe * x - re * re * x + qe * y * pe + re * z * pe;
    auto centripetal_y = -pe * pe * y - re * re * y + pe * x * qe + re * z * qe;
    auto centripetal_z = -pe * pe * z - qe * qe * z + pe * x * re + qe * y * re;
    // 2 w x v
    auto coriolis_x = expression{2.} * (qe * vz - re * vy);
    auto coriolis_y = expression{2.} * (re * vx - pe * vz);
    auto coriolis_z = expression{2.} * (pe * vy - qe * vx);

    // Assembling the return vector containing l.h.s. and r.h.s. (note the fundamental use of sum() for
    // efficiency and to allow compact mode to do his job)
    retval.push_back(prime(x) = vx);
    retval.push_back(prime(y) = vy);
    retval.push_back(prime(z) = vz);
    retval.push_back(prime(vx) = sum(x_acc) - centripetal_x - coriolis_x);
    retval.push_back(prime(vy) = sum(y_acc) - centripetal_y - coriolis_y);
    retval.push_back(prime(vz) = sum(z_acc) - centripetal_z - coriolis_z);

    return retval;
}

expression energy_mascon_system_impl(expression Gconst, std::vector<expression> x,
                                     std::vector<std::vector<expression>> mascon_points,
                                     std::vector<expression> mascon_masses, expression pe, expression qe, expression re)
{
    auto kinetic = expression{(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) / expression{2.}};
    auto potential_g = expression{0.};
    for (decltype(mascon_masses.size()) i = 0u; i < mascon_masses.size(); ++i) {
        auto distance = expression{sqrt((x[0] - mascon_points[i][0]) * (x[0] - mascon_points[i][0])
                                        + (x[1] - mascon_points[i][1]) * (x[1] - mascon_points[i][1])
                                        + (x[2] - mascon_points[i][2]) * (x[2] - mascon_points[i][2]))};
        potential_g -= Gconst * mascon_masses[i] / distance;
    }
    auto potential_c
        = -expression{number{0.5}} * (pe * pe + qe * qe + re * re) * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
          + expression{number{0.5}} * (x[0] * pe + x[1] * qe + x[2] * re) * (x[0] * pe + x[1] * qe + x[2] * re);
    return kinetic + potential_g + potential_c;
}

} // namespace detail

HEYOKA_END_NAMESPACE
