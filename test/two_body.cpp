// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

#include <iostream>

using namespace heyoka;

// Small helper to compute the angular momentum
// wrt the origin given a state vector.
template <typename T>
T compute_am(const std::vector<T> &st)
{
    auto vx0 = st[0];
    auto vx1 = st[1];
    auto vy0 = st[2];
    auto vy1 = st[3];
    auto vz0 = st[4];
    auto vz1 = st[5];

    auto x0 = st[6];
    auto x1 = st[7];
    auto y0 = st[8];
    auto y1 = st[9];
    auto z0 = st[10];
    auto z1 = st[11];

    double am0[] = {y0 * vz0 - z0 * vy0, z0 * vx0 - x0 * vz0, x0 * vy0 - y0 * vx0};
    double am1[] = {y1 * vz1 - z1 * vy1, z1 * vx1 - x1 * vz1, x1 * vy1 - y1 * vx1};
    double am[] = {am0[0] + am1[0], am0[1] + am1[1], am0[2] + am1[2]};

    return std::sqrt(am[0] * am[0] + am[1] * am[1] + am[2] * am[2]);
}

// Two-body problem energy from the state vector.
template <typename T>
T tbp_energy(const std::vector<T> &st)
{
    auto Dx = st[6] - st[7];
    auto Dy = st[8] - st[9];
    auto Dz = st[10] - st[11];
    auto dist = std::sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
    auto U = -1 / dist;

    auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
    auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];

    return 1 / 2. * (v2_0 + v2_1) + U;
}

TEST_CASE("two body")
{
    std::cout.precision(14);

    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    std::vector<double> init_state{0.593, -0.593, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0};

    taylor_adaptive_dbl tad{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3, vx0,
                             vx1, vy0, vy1, vz0, vz1},
                            std::move(init_state),
                            0,
                            1E-16,
                            1E-16};

    const auto &st = tad.get_state();

    for (auto i = 0; i < 200; ++i) {
        const auto [oc, h, ord] = tad.step();
        REQUIRE(oc == taylor_adaptive_dbl::outcome::success);

        std::cout << "h=" << h << ", order=" << ord << ", pos=";
        for (auto j = 6; j < 12; ++j) {
            std::cout << st[j] << ", ";
        }
        std::cout << "energy=" << tbp_energy(st) << ", am=" << compute_am(st) << '\n';
    }
}

// Energy of two uniform overlapping spheres.
template <typename T>
T tus_energy(T rs, const std::vector<T> &st)
{
    auto Dx = st[6] - st[7];
    auto Dy = st[8] - st[9];
    auto Dz = st[10] - st[11];
    auto dist = std::sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
    auto U = 1 / (160 * std::pow(rs, 6))
             * (std::pow(dist, 5) - 30 * rs * rs * std::pow(dist, 3) + 80 * std::pow(rs, 3) * dist * dist);

    auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
    auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];

    // NOTE: -6/(5*rs) is a corrective term to ensure
    // the potential matches the 2bp potential at the
    // transition threshold.
    return 1 / 2. * (v2_0 + v2_1) + U - 6. / (5 * rs);
}

TEST_CASE("two uniform spheres")
{
    std::cout.precision(14);

    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01 = pow(x01 * x01 + y01 * y01 + z01 * z01, 1_dbl / 2_dbl);
    const auto rs_val = std::sqrt(2);
    auto rs = expression{number{rs_val}};
    auto rs2 = rs * rs;
    auto rs3 = rs2 * rs;
    auto rs6 = rs3 * rs3;
    auto fac = (r01 * r01 * r01 - 18_dbl * rs2 * r01 + 32_dbl * rs3) / (32_dbl * rs6);

    std::vector<double> init_state{0.593, -0.593, 0, 0, 0, 0, -1, 1, -1, 1, 0, 0};

    taylor_adaptive_dbl tad{
        {x01 * fac, -x01 * fac, y01 * fac, -y01 * fac, z01 * fac, -z01 * fac, vx0, vx1, vy0, vy1, vz0, vz1},
        std::move(init_state),
        0,
        1E-16,
        1E-16};

    const auto &st = tad.get_state();

    for (auto i = 0; i < 200; ++i) {
        const auto [oc, h, ord] = tad.step();
        REQUIRE(oc == taylor_adaptive_dbl::outcome::success);

        std::cout << "h=" << h << ", order=" << ord << ", pos=";
        for (auto j = 6; j < 12; ++j) {
            std::cout << st[j] << ", ";
        }
        std::cout << "energy=" << tus_energy(rs_val, st) << ", am=" << compute_am(st) << '\n';
    }
}

TEST_CASE("mixed tb/spheres")
{
    std::cout.precision(14);

    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01 = pow(x01 * x01 + y01 * y01 + z01 * z01, 1_dbl / 2_dbl);
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);
    const auto rs_val = std::sqrt(2);
    auto rs = expression{number{rs_val}};
    auto rs2 = rs * rs;
    auto rs3 = rs2 * rs;
    auto rs6 = rs3 * rs3;
    auto fac = (r01 * r01 * r01 - 18_dbl * rs2 * r01 + 32_dbl * rs3) / (32_dbl * rs6);

    std::vector<double> init_state{0.595, -0.595, 0, 0, 0, 0, -1.0001, 1.0001, -1, 1, 0, 0};

    // 2BP integrator.
    taylor_adaptive_dbl t_2bp{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3,
                               vx0, vx1, vy0, vy1, vz0, vz1},
                              init_state,
                              0,
                              1E-16,
                              1E-16};

    // 2US integrator.
    taylor_adaptive_dbl t_2us{
        {x01 * fac, -x01 * fac, y01 * fac, -y01 * fac, z01 * fac, -z01 * fac, vx0, vx1, vy0, vy1, vz0, vz1},
        std::move(init_state),
        0,
        1E-16,
        1E-16};

    // Helper to compute the distance**2 between
    // the sphere's centres given a state vector.
    auto compute_dist2 = [](const auto &st) {
        auto Dx = st[6] - st[7];
        auto Dy = st[8] - st[9];
        auto Dz = st[10] - st[11];
        return Dx * Dx + Dy * Dy + Dz * Dz;
    };

    // Helper to get the dynamical regime given an input state vector:
    // this returns true for the Keplerian regime (non-overlapping spheres),
    // false for the 2US regime (overlapping spheres).
    auto get_regime = [rs_val, compute_dist2](const auto &st) { return compute_dist2(st) > 4 * rs_val * rs_val; };

    // The simulation is set up to start in the Keplerian regime.
    auto cur_regime = get_regime(t_2bp.get_state());
    REQUIRE(cur_regime);

    // Pointers to the current and other integrators.
    taylor_adaptive_dbl *cur_t = &t_2bp, *other_t = &t_2us;

    while (true) {
        // Compute the max velocity in the system.
        const auto &st = cur_t->get_state();
        auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
        auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];
        auto max_v = std::max(std::sqrt(v2_0), std::sqrt(v2_1));

        // Do a timestep imposing that that max_v * delta_t < 1/2*rs.
        auto [oc, h, order] = cur_t->step(rs_val / (2 * max_v));
        REQUIRE(oc == taylor_adaptive_dbl::outcome::success);

        if (get_regime(cur_t->get_state()) != cur_regime) {
            std::cout << "Regime CHANGE\n";

            auto cur_dist = std::sqrt(compute_dist2(cur_t->get_state()));

            while (std::abs(cur_dist - 2 * rs_val) > 1E-12 && h > 1E-12) {
                std::cout << "Flappo: " << std::abs(cur_dist - 2 * rs_val) << '\n';

                if (cur_regime) {
                    // Keplerian regime -> sphere regime.
                    if (cur_dist < 2 * rs_val) {
                        // Spheres are overlapping, integrate
                        // backward.
                        cur_t->propagate_for(-h / 2);
                    } else {
                        // Spheres are apart, integrate forward.
                        cur_t->propagate_for(h / 2);
                    }
                } else {
                    // Sphere regime -> Kepler regime.
                    if (cur_dist < 2 * rs_val) {
                        // Spheres are overlapping, integrate
                        // forward.
                        cur_t->propagate_for(h / 2);
                    } else {
                        // Spheres are apart, integrate backward.
                        cur_t->propagate_for(-h / 2);
                    }
                }
                // Update h and cur_dist.
                h /= 2;
                cur_dist = std::sqrt(compute_dist2(cur_t->get_state()));
            }

            // Copy the current state to the other integrator.
            other_t->set_time(cur_t->get_time());
            other_t->set_state(cur_t->get_state());

            // Swap around, update regime.
            std::swap(other_t, cur_t);
            cur_regime = !cur_regime;
        } else {
            std::cout << "State: ";
            for (auto j = 6; j < 12; ++j) {
                std::cout << cur_t->get_state()[j] << ", ";
            }
            std::cout << '\n';
            std::cout << "Current energy: "
                      << (cur_regime ? tbp_energy(cur_t->get_state()) : tus_energy(rs_val, cur_t->get_state())) << '\n';
            std::cout << "Current am: " << compute_am(cur_t->get_state()) << '\n';
        }
    }
}
