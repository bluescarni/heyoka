// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/mascon.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const heyoka::expression &x)
{
    return x * x;
}

// NOTE: original code implementing the mascon model.
// We will be using it to check the new implementation.

HEYOKA_BEGIN_NAMESPACE

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(points);
IGOR_MAKE_NAMED_ARGUMENT(state);

} // namespace kw

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
        auto r2 = sum({square_wrapper(xdiff), square_wrapper(ydiff), square_wrapper(zdiff)});
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
    // 1 - Check input consistency
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
                return expression{number{p(kw::Gconst)}};
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
                return expression{number{p(kw::Gconst)}};
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

HEYOKA_END_NAMESPACE

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic cmp")
{
    std::mt19937 rng;
    std::uniform_real_distribution<double> rdist(-1e-2, 1e-2);

    const auto n_masses = 1u;
    std::vector<double> pos, masses;
    std::vector<std::vector<double>> points(n_masses);
    for (auto i = 0u; i < n_masses; ++i) {
        masses.push_back(rdist(rng));

        for (auto j = 0u; j < 3u; ++j) {
            const auto tmp = rdist(rng);

            pos.push_back(tmp);
            points[i].push_back(tmp);
        }
    }

    const std::vector omega = {.1, .11, .12};

    // Randomly generate a fixed centres problem and check against the old implementation.
    {
        auto dyn = model::mascon(kw::masses = masses, kw::positions = pos, kw::Gconst = 1.01, kw::omega = omega);
        auto dyn_old
            = make_mascon_system(kw::masses = masses, kw::points = points, kw::Gconst = 1.01, kw::omega = omega);

        const std::vector init_state = {1., 0., 0., 0., 1., 0.};

        auto ta = taylor_adaptive{dyn, init_state, kw::compact_mode = true};

        REQUIRE(ta.get_decomposition().size() == 45u);

        auto ta_old = taylor_adaptive{dyn_old, init_state, kw::compact_mode = true};

        ta.propagate_until(10.);
        ta_old.propagate_until(10.);

        for (auto i = 0u; i < 6u; ++i) {
            REQUIRE(ta.get_state()[i] == approximately(ta_old.get_state()[i], 1000.));
        }
    }

    // Do an energy conservation check as well.
    {
        auto dyn = model::mascon(kw::masses = masses, kw::positions = pos, kw::Gconst = 1.01, kw::omega = omega);
        const std::vector init_state = {1., 0., 0., 0., 1., 0.};

        auto ta = taylor_adaptive{dyn, init_state, kw::compact_mode = true};

        llvm_state s;
        const auto dc1 = add_cfunc<double>(
            s, "en",
            {model::mascon_energy(kw::masses = masses, kw::positions = pos, kw::Gconst = 1.01, kw::omega = omega)},
            kw::vars = {"x"_var, "y"_var, "z"_var, "vx"_var, "vy"_var, "vz"_var});

        const auto dc2 = add_cfunc<double>(
            s, "en2",
            {0.5 * ("vx"_var * "vx"_var + "vy"_var * "vy"_var + "vz"_var * "vz"_var)
             + model::mascon_potential(kw::masses = masses, kw::positions = pos, kw::Gconst = 1.01, kw::omega = omega)},
            kw::vars = {"x"_var, "y"_var, "z"_var, "vx"_var, "vy"_var, "vz"_var});

        REQUIRE(dc1.size() == 27u);
        REQUIRE(dc2.size() == 30u);

        s.compile();

        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en"));
        auto *cf2
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en2"));

        double E0 = 0;
        cf(&E0, ta.get_state().data(), nullptr, nullptr);

        ta.propagate_until(10.);

        double E = 0;
        cf(&E, ta.get_state().data(), nullptr, nullptr);

        REQUIRE(E == approximately(E0));

        cf2(&E, ta.get_state().data(), nullptr, nullptr);

        REQUIRE(E == approximately(E0));
    }
}
