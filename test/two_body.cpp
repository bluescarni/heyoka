// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Small helper to compute the angular momentum
// wrt the origin given a state vector.
template <typename T>
T compute_am(const std::vector<T> &st)
{
    using std::sqrt;

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

    T am0[] = {y0 * vz0 - z0 * vy0, z0 * vx0 - x0 * vz0, x0 * vy0 - y0 * vx0};
    T am1[] = {y1 * vz1 - z1 * vy1, z1 * vx1 - x1 * vz1, x1 * vy1 - y1 * vx1};
    T am[] = {am0[0] + am1[0], am0[1] + am1[1], am0[2] + am1[2]};

    return sqrt(am[0] * am[0] + am[1] * am[1] + am[2] * am[2]);
}

// Two-body problem energy from the state vector.
template <typename T>
T tbp_energy(const std::vector<T> &st)
{
    using std::sqrt;

    auto Dx = st[6] - st[7];
    auto Dy = st[8] - st[9];
    auto Dz = st[10] - st[11];
    auto dist = sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
    auto U = -1 / dist;

    auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
    auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];

    return T(1) / T(2) * (v2_0 + v2_1) + U;
}

TEST_CASE("two body")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;
        using std::cos;

        using fp_t = decltype(fp_x);

        auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
            = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

        auto x01 = x1 - x0;
        auto y01 = y1 - y0;
        auto z01 = z1 - z0;
        auto r01_m3
            = pow(x01 * x01 + y01 * y01 + z01 * z01, expression{number{fp_t{-3}}} / expression{number{fp_t{2}}});

        const auto kep = std::array<fp_t, 6>{fp_t{1.5}, fp_t{.2}, fp_t{.3}, fp_t{.4}, fp_t{.5}, fp_t{.6}};
        const auto [c_x, c_v] = kep_to_cart(kep, fp_t{1} / 4);

        std::vector<fp_t> init_state{c_v[0], -c_v[0], c_v[1], -c_v[1], c_v[2], -c_v[2],
                                     c_x[0], -c_x[0], c_x[1], -c_x[1], c_x[2], -c_x[2]};

        taylor_adaptive<fp_t> tad{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                   -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                  std::move(init_state),
                                  kw::opt_level = opt_level,
                                  kw::high_accuracy = high_accuracy,
                                  kw::compact_mode = compact_mode};

        const auto &st = tad.get_state();

        const auto en = tbp_energy(st);
        const auto am = compute_am(st);

        for (auto i = 0; i < 200; ++i) {
            const auto [oc, h] = tad.step();
            REQUIRE(oc == taylor_outcome::success);
            REQUIRE(tbp_energy(st) == approximately(en, fp_t{1E2}));
            REQUIRE(compute_am(st) == approximately(am, fp_t{1E2}));

            const auto kep1 = cart_to_kep(std::array<fp_t, 3>{st[6], st[8], st[10]},
                                          std::array<fp_t, 3>{st[0], st[2], st[4]}, fp_t{1} / 4);
            const auto kep2 = cart_to_kep(std::array<fp_t, 3>{st[7], st[9], st[11]},
                                          std::array<fp_t, 3>{st[1], st[3], st[5]}, fp_t{1} / 4);

            REQUIRE(kep1[0] == approximately(fp_t{1.5}, fp_t{1E2}));
            REQUIRE(kep2[0] == approximately(fp_t{1.5}, fp_t{1E2}));
            REQUIRE(kep1[1] == approximately(fp_t{.2}, fp_t{1E3}));
            REQUIRE(kep2[1] == approximately(fp_t{.2}, fp_t{1E3}));
            REQUIRE(kep1[2] == approximately(fp_t{.3}, fp_t{1E2}));
            REQUIRE(kep2[2] == approximately(fp_t{.3}, fp_t{1E2}));
            REQUIRE(abs(cos(kep1[3])) == approximately(abs(cos(fp_t{.4})), fp_t{1E3}));
            REQUIRE(abs(cos(kep2[3])) == approximately(abs(cos(fp_t{.4})), fp_t{1E3}));
            REQUIRE(kep1[4] == approximately(fp_t{.5}, fp_t{1E2}));
            REQUIRE(kep2[4] == approximately(fp_t{.5}, fp_t{1E2}));
        }
    };

    for (auto cm : {true, false}) {
        for (auto ha : {true, false}) {
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 0, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 1, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 2, ha, cm); });
            tuple_for_each(fp_types, [&tester, ha, cm](auto x) { tester(x, 3, ha, cm); });
        }
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

    std::vector<double> init_state{0.593, -0.593, 0, 0, 0, 0, -1.000001, 1.000001, -1, 1, 0, 0};

    taylor_adaptive<double> tad{
        {x01 * fac, -x01 * fac, y01 * fac, -y01 * fac, z01 * fac, -z01 * fac, vx0, vx1, vy0, vy1, vz0, vz1},
        std::move(init_state)};

    const auto &st = tad.get_state();

    const auto en = tbp_energy(st);
    const auto am = compute_am(st);

    for (auto i = 0; i < 200; ++i) {
        const auto [oc, h] = tad.step();
        REQUIRE(oc == taylor_outcome::success);
        REQUIRE(std::abs((en - tus_energy(rs_val, st)) / en) <= 1E-11);
        REQUIRE(std::abs((am - compute_am(st)) / am) <= 1E-11);
    }
}

TEST_CASE("mixed tb/spheres")
{
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

    std::vector<double> init_state{0.593, -0.593, 0, 0, 0, 0, -1.000001, 1.000001, -1, 1, 0, 0};

    // 2BP integrator.
    taylor_adaptive<double> t_2bp{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                   -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                  init_state};

    // 2US integrator.
    taylor_adaptive<double> t_2us{
        {x01 * fac, -x01 * fac, y01 * fac, -y01 * fac, z01 * fac, -z01 * fac, vx0, vx1, vy0, vy1, vz0, vz1},
        std::move(init_state)};

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
    taylor_adaptive<double> *cur_t = &t_2bp, *other_t = &t_2us;

    const auto en = tbp_energy(cur_t->get_state());
    const auto am = compute_am(cur_t->get_state());

    for (auto i = 0; i < 200; ++i) {
        // Compute the max velocity in the system.
        const auto &st = cur_t->get_state();
        auto v2_0 = st[0] * st[0] + st[2] * st[2] + st[4] * st[4];
        auto v2_1 = st[1] * st[1] + st[3] * st[3] + st[5] * st[5];
        auto max_v = std::max(std::sqrt(v2_0), std::sqrt(v2_1));

        // Do a timestep imposing that that max_v * delta_t < 1/2*rs.
        auto [oc, h] = cur_t->step(rs_val / (2 * max_v));
        REQUIRE((oc == taylor_outcome::success || oc == taylor_outcome::time_limit));

        if (get_regime(cur_t->get_state()) != cur_regime) {
            auto cur_dist = std::sqrt(compute_dist2(cur_t->get_state()));

            while (std::abs(cur_dist - 2 * rs_val) > 1E-12 && h > 1E-12) {
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
            std::copy(cur_t->get_state().begin(), cur_t->get_state().end(), other_t->get_state_data());

            // Swap around, update regime.
            std::swap(other_t, cur_t);
            cur_regime = !cur_regime;
        } else {
            const auto cur_energy
                = cur_regime ? tbp_energy(cur_t->get_state()) : tus_energy(rs_val, cur_t->get_state());
            REQUIRE(std::abs((en - cur_energy) / en) <= 1E-8);
            REQUIRE(std::abs((am - compute_am(cur_t->get_state())) / am) <= 1E-8);
        }
    }
}
