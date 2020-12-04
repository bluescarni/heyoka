// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MASCON_HPP
#define HEYOKA_MASCON_HPP

#include <heyoka/detail/igor.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/square.hpp>
#include <vector>

namespace heyoka
{

// mascon_points -> [N,3] array containing the positions of the masses (units L)
// mascon_masses -> [N] array containing the position of the masses (units M)
// pd, qd, rd -> angular velocity of the asteroid in the frame used for the mascon model (units rad/T)
//
// GConst kwarg -> Cavendish constant (units L^3/T^2/M)
// Note, units must be consistent. Choosing L and M is done via the mascon model, T is derived by the value of G. The
// angular velocity must be consequent (equivalently one can choose the units for w and induce them on the value of G).
template <typename... KwArgs>
inline std::vector<std::pair<expression, expression>> make_mascon_system(KwArgs &&... kw_args)
{
    // 1 - Check input consistency (TODO)
    // 2 - We parse the unnamed arguments
    igor::parser p{kw_args...};
    if constexpr (p.has_unnamed_arguments()) {
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments in the construction of the mascon system contain "
                      "unnamed arguments.");
    } else {
        // G constant (defaults to 1).
        auto Gconst = [&p]() {
            if constexpr (p.has(kw::Gconst)) {
                return expression{number{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))}};
            } else {
                return expression{number{1.}};
            }
        }();

        // mascon_points (no default)
        std::vector<std::vector<number>> mascon_points;
        if constexpr (p.has(kw::mascon_points)) {
            for (const auto &point : p(kw::mascon_points)) {
                mascon_points.emplace_back(std::vector<number>{number{point[0]}, number{point[1]}, number{point[2]}});
            }
        } else {
            throw std::invalid_argument("mascon_points is missing from the kwarg list!");
        };

        // mascon_masses (no default)
        std::vector<number> mascon_masses;
        if constexpr (p.has(kw::mascon_masses)) {
            for (const auto &mass : p(kw::mascon_masses)) {
                mascon_masses.emplace_back(mass);
            }
        } else {
            throw std::invalid_argument("mascon_masses is missing from the kwarg list!");
        };

        // omega (no default)
        number pn(0.), qn(0.), rn(0.);
        if constexpr (p.has(kw::omega)) {
            pn = number{p(kw::omega)[0]};
            qn = number{p(kw::omega)[1]};
            rn = number{p(kw::omega)[2]};
        } else {
            throw std::invalid_argument("omega is missing from the kwarg list!");
        };

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
            auto x_masc = expression{mascon_points[i][0]};
            auto y_masc = expression{mascon_points[i][1]};
            auto z_masc = expression{mascon_points[i][2]};
            auto m_masc = expression{mascon_masses[i]};
            auto xdiff = (x - x_masc);
            auto ydiff = (y - y_masc);
            auto zdiff = (z - z_masc);
            auto r2 = square(xdiff) + square(ydiff) + square(zdiff);
            auto common_factor = -Gconst * m_masc * pow(r2, expression{number{-3. / 2.}});
            x_acc.push_back(common_factor * xdiff);
            y_acc.push_back(common_factor * ydiff);
            z_acc.push_back(common_factor * zdiff);
        }
        // SECOND: centripetal and Coriolis
        auto p = expression{pn};
        auto q = expression{qn};
        auto r = expression{rn};
        // w x w x r
        auto centripetal_x = -q * q * x - r * r * x + q * y * p + r * z * p;
        auto centripetal_y = -p * p * y - r * r * y + p * x * q + r * z * q;
        auto centripetal_z = -p * p * z - q * q * z + p * x * r + q * y * r;
        // 2 w x v
        auto coriolis_x = expression{number{2.}} * (q * vz - r * vy);
        auto coriolis_y = expression{number{2.}} * (r * vx - p * vz);
        auto coriolis_z = expression{number{2.}} * (p * vy - q * vx);

        // Assembling the return vector containing l.h.s. and r.h.s. (note the fundamental use of pairwise_sum for
        // efficiency and to allow compact mode to do his job)
        retval.push_back(prime(x) = vx);
        retval.push_back(prime(y) = vy);
        retval.push_back(prime(z) = vz);
        retval.push_back(prime(vx) = pairwise_sum(x_acc) - centripetal_x - coriolis_x);
        retval.push_back(prime(vy) = pairwise_sum(y_acc) - centripetal_y - coriolis_y);
        retval.push_back(prime(vz) = pairwise_sum(z_acc) - centripetal_z - coriolis_z);

        return retval;
    }
}

// x -> system state
// mascon_points -> [N,3] array containing the positions of the masses (units L)
// mascon_masses -> [N] array containing the position of the masses (units M)
// pd, qd, rd -> angular velocity of the asteroid in the frame used for the mascon model (units rad/T)
//
// GConst kwarg -> Cavendish constant (units L^3/T^2/M)
// Note, units must be consistent. Choosing L and M is done via the mascon model, T is derived by the value of G. The
// angular velocity must be consequent (equivalently one can choose the units for w and induce them on the value of G).
template <typename... KwArgs>
expression energy_mascon_system(KwArgs &&... kw_args)
// const std::vector<double> x, const P &mascon_points, const M &mascon_masses, double p, double q,
// double r, double G)
{
    igor::parser p{kw_args...};
    if constexpr (p.has_unnamed_arguments()) {
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments in the construction of the mascon system contain "
                      "unnamed arguments.");
    } else {
        // G constant (defaults to 1).
        auto Gconst = [&p]() {
            if constexpr (p.has(kw::Gconst)) {
                return expression{number{std::forward<decltype(p(kw::Gconst))>(p(kw::Gconst))}};
            } else {
                return expression{number{1.}};
            }
        }();
        // state of the integrator 6D (no default)
        std::vector<expression> x;
        if constexpr (p.has(kw::state)) {
            for (const auto &component : p(kw::state)) {
                x.emplace_back(number{component});
            }
        } else {
            throw std::invalid_argument("state is missing from the kwarg list!");
        };
        // mascon_points (no default)
        std::vector<std::vector<expression>> mascon_points;
        if constexpr (p.has(kw::mascon_points)) {
            for (const auto &point : p(kw::mascon_points)) {
                mascon_points.emplace_back(
                    std::vector<expression>{expression{number{point[0]}}, expression{number{point[1]}}, expression{number{point[2]}}});
            }
        } else {
            throw std::invalid_argument("mascon_points is missing from the kwarg list!");
        };

        // mascon_masses (no default)
        std::vector<expression> mascon_masses;
        if constexpr (p.has(kw::mascon_masses)) {
            for (const auto &mass : p(kw::mascon_masses)) {
                mascon_masses.emplace_back(number{mass});
            }
        } else {
            throw std::invalid_argument("mascon_masses is missing from the kwarg list!");
        };

        // omega (no default)
        expression pe(0.), qe(0.), re(0.);
        if constexpr (p.has(kw::omega)) {
            pe = expression{number{p(kw::omega)[0]}};
            qe = expression{number{p(kw::omega)[1]}};
            re = expression{number{p(kw::omega)[2]}};
        } else {
            throw std::invalid_argument("omega is missing from the kwarg list!");
        };

        expression kinetic = expression{(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) / expression{number{2.}}};
        expression potential_g = expression{number{0.}};
        for (decltype(mascon_masses.size()) i = 0u; i < mascon_masses.size(); ++i) {
            expression distance = expression{sqrt((x[0] - mascon_points[i][0]) * (x[0] - mascon_points[i][0])
                                                  + (x[1] - mascon_points[i][1]) * (x[1] - mascon_points[i][1])
                                                  + (x[2] - mascon_points[i][2]) * (x[2] - mascon_points[i][2]))};
            potential_g -= Gconst * mascon_masses[i] / distance;
        }
        auto potential_c
            = -expression{number{0.5}} * (pe * pe + qe * qe + re * re) * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
              + expression{number{0.5}} * (x[0] * pe + x[1] * qe + x[2] * re) * (x[0] * pe + x[1] * qe + x[2] * re);
        return kinetic + potential_g + potential_c;
    }
}

} // namespace heyoka

#endif
