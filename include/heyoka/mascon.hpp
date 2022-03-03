// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math.hpp>
#include <vector>

namespace heyoka
{

namespace detail
{

HEYOKA_DLL_PUBLIC std::vector<std::pair<expression, expression>>
    make_mascon_system_impl(expression, std::vector<std::vector<expression>>, std::vector<expression>, expression,
                            expression, expression);

HEYOKA_DLL_PUBLIC expression energy_mascon_system_impl(expression, std::vector<expression>,
                                                       std::vector<std::vector<expression>>, std::vector<expression>,
                                                       expression, expression, expression);

} // namespace detail

// mascon_points -> [N,3] array containing the positions of the masses (units L)
// mascon_masses -> [N] array containing the position of the masses (units M)
// pd, qd, rd -> angular velocity of the asteroid in the frame used for the mascon model (units rad/T)
//
// GConst kwarg -> Cavendish constant (units L^3/T^2/M)
// Note, units must be consistent. Choosing L and M is done via the mascon model, T is derived by the value of G. The
// angular velocity must be consequent (equivalently one can choose the units for w and induce them on the value of G).
template <typename... KwArgs>
inline std::vector<std::pair<expression, expression>> make_mascon_system(KwArgs &&...kw_args)
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
        std::vector<std::vector<expression>> mascon_points;
        if constexpr (p.has(kw::points)) {
            for (const auto &point : p(kw::points)) {
                if (std::size(point) == 3) {
                    mascon_points.emplace_back(
                        std::vector<expression>{expression{point[0]}, expression{point[1]}, expression{point[2]}});
                } else {
                    throw std::invalid_argument("All mascon points must have a dimension of exactly 3. A dimension of "
                                                + std::to_string(std::size(point))
                                                + " was detected when constructing the mascon system.");
                }
            }
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "mascon_points is missing from the kwarg list!");
        };

        // mascon_masses (no default)
        std::vector<expression> mascon_masses;
        if constexpr (p.has(kw::masses)) {
            for (const auto &mass : p(kw::masses)) {
                mascon_masses.emplace_back(mass);
            }
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "mascon_masses is missing from the kwarg list!");
        };

        // omega (no default)
        expression pe(0.), qe(0.), re(0.);
        if constexpr (p.has(kw::omega)) {
            pe = expression{p(kw::omega)[0]};
            qe = expression{p(kw::omega)[1]};
            re = expression{p(kw::omega)[2]};
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "omega is missing from the kwarg list!");
        };

        return detail::make_mascon_system_impl(std::move(Gconst), std::move(mascon_points), std::move(mascon_masses),
                                               std::move(pe), std::move(qe), std::move(re));
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
expression energy_mascon_system(KwArgs &&...kw_args)
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
                return expression{1.};
            }
        }();
        // state of the integrator 6D (no default)
        std::vector<expression> x;
        if constexpr (p.has(kw::state)) {
            for (const auto &component : p(kw::state)) {
                x.emplace_back(component);
            }
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "state is missing from the kwarg list!");
        };
        // mascon_points (no default)
        std::vector<std::vector<expression>> mascon_points;
        if constexpr (p.has(kw::points)) {
            for (const auto &point : p(kw::points)) {
                if (std::size(point) == 3) {
                    mascon_points.emplace_back(
                        std::vector<expression>{expression{point[0]}, expression{point[1]}, expression{point[2]}});
                } else {
                    throw std::invalid_argument("All mascon points must have a dimension of exactly 3. A dimension of "
                                                + std::to_string(std::size(point))
                                                + " was detected when computing the energy of the mascon system.");
                }
            }
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "mascon_points is missing from the kwarg list!");
        };

        // mascon_masses (no default)
        std::vector<expression> mascon_masses;
        if constexpr (p.has(kw::masses)) {
            for (const auto &mass : p(kw::masses)) {
                mascon_masses.emplace_back(mass);
            }
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "mascon_masses is missing from the kwarg list!");
        };

        // omega (no default)
        expression pe(0.), qe(0.), re(0.);
        if constexpr (p.has(kw::omega)) {
            pe = expression{p(kw::omega)[0]};
            qe = expression{p(kw::omega)[1]};
            re = expression{p(kw::omega)[2]};
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "omega is missing from the kwarg list!");
        };
        return detail::energy_mascon_system_impl(Gconst, x, mascon_points, mascon_masses, pe, qe, re);
    }
}

} // namespace heyoka

#endif
