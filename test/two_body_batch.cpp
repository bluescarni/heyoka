// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("two body batch")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool ha, bool cm) {
        using std::cos;
        using std::abs;

        using fp_t = decltype(fp_x);

        // NOTE: don't test larger batch sizes for types other than double.
        const std::uint32_t batch_size = std::is_same_v<fp_t, double> ? 4 : 1;

        auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
            = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

        auto x01 = x1 - x0;
        auto y01 = y1 - y0;
        auto z01 = z1 - z0;
        auto r01_m3
            = pow(x01 * x01 + y01 * y01 + z01 * z01, expression{number{fp_t{-3}}} / expression{number{fp_t{2}}});

        // The system of equations.
        auto sys = {x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3,
                    vx0,          vx1,           vy0,          vy1,           vz0,          vz1};

        // Generate a bunch of random initial conditions in orbital elements.
        std::vector<std::array<fp_t, 6>> v_kep;
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            std::uniform_real_distribution<float> a_dist(0.1f, 10.f), e_dist(0.1f, 0.5f), i_dist(0.1f, 3.13f),
                ang_dist(0.1f, 6.28f);
            v_kep.push_back(std::array<fp_t, 6>{fp_t{a_dist(rng)}, fp_t{e_dist(rng)}, fp_t{i_dist(rng)},
                                                fp_t{ang_dist(rng)}, fp_t{ang_dist(rng)}, fp_t{ang_dist(rng)}});
        }

        // Generate the initial state/time vector for the batch integrator.
        std::vector<fp_t> init_states(batch_size * 12u);
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const auto [x, v] = kep_to_cart(v_kep[i], fp_t(1) / 4);

            init_states[0u * batch_size + i] = v[0];
            init_states[1u * batch_size + i] = -v[0];
            init_states[2u * batch_size + i] = v[1];
            init_states[3u * batch_size + i] = -v[1];
            init_states[4u * batch_size + i] = v[2];
            init_states[5u * batch_size + i] = -v[2];
            init_states[6u * batch_size + i] = x[0];
            init_states[7u * batch_size + i] = -x[0];
            init_states[8u * batch_size + i] = x[1];
            init_states[9u * batch_size + i] = -x[1];
            init_states[10u * batch_size + i] = x[2];
            init_states[11u * batch_size + i] = -x[2];
        }

        // Create a corresponding scalar integrator.
        // Init with the first state in init_states
        // (does not matter, will be changed in the loop below).
        std::vector<fp_t> scalar_init;
        for (std::uint32_t j = 0; j < 12u; ++j) {
            scalar_init.push_back(init_states[j * batch_size]);
        }
        taylor_adaptive<fp_t> ta{sys, std::move(scalar_init), kw::opt_level = opt_level, kw::high_accuracy = ha,
                                 kw::compact_mode = cm};
        const auto &st = ta.get_state();
        auto scal_st(ta.get_state());

        // Init the batch integrator.
        taylor_adaptive_batch<fp_t> tab{sys,
                                        std::move(init_states),
                                        batch_size,
                                        kw::opt_level = opt_level,
                                        kw::high_accuracy = ha,
                                        kw::compact_mode = cm};

        const auto &bst = tab.get_states();
        auto bst_copy(bst);
        auto times_copy(tab.get_times());

        for (auto _ = 0; _ < 200; ++_) {
            // Copy the batch state/times before propagation.
            bst_copy = bst;
            times_copy = tab.get_times();

            const auto &res = tab.step();

            for (std::uint32_t i = 0; i < batch_size; ++i) {
                // Evolve separately the scalar integrators.

                // Set state and time.
                for (std::uint32_t j = 0; j < 12u; ++j) {
                    scal_st[j] = bst_copy[j * batch_size + i];
                }
                std::copy(scal_st.begin(), scal_st.end(), ta.get_state_data());
                ta.set_time(times_copy[i]);

                auto s_res = ta.step();

                // NOTE: this tolerance can be lowered once
                // we have a way of disabling fast math flags.
                // We will probably have to add some extra leeway
                // in the conservation of the orbital elements though.
                const auto tol_mul = fp_t{1E4};

                // Check the result of the integration.
                REQUIRE(std::get<0>(s_res) == std::get<0>(res[i]));
                REQUIRE(std::get<1>(s_res) == approximately(std::get<1>(res[i]), tol_mul));

                // Check the state vectors.
                for (std::uint32_t j = 0; j < 12u; ++j) {
                    REQUIRE(st[j] == approximately(bst[j * batch_size + i], tol_mul));
                }

                // Check the conservation of the orbital elements.
                const auto kep1 = cart_to_kep<fp_t>({st[6], st[8], st[10]}, {st[0], st[2], st[4]}, fp_t{1} / 4);
                const auto kep2 = cart_to_kep<fp_t>({st[7], st[9], st[11]}, {st[1], st[3], st[5]}, fp_t{1} / 4);

                // Both bodies have the same semi-major axis.
                REQUIRE(kep1[0] == approximately(v_kep[i][0], tol_mul));
                REQUIRE(kep2[0] == approximately(v_kep[i][0], tol_mul));

                // Same eccentricity.
                REQUIRE(kep1[1] == approximately(v_kep[i][1], tol_mul));
                REQUIRE(kep2[1] == approximately(v_kep[i][1], tol_mul));

                // Same inclination.
                REQUIRE(kep1[2] == approximately(v_kep[i][2], tol_mul));
                REQUIRE(kep2[2] == approximately(v_kep[i][2], tol_mul));

                // omega is phased by pi.
                REQUIRE(abs(cos(kep1[3])) == approximately(abs(cos(v_kep[i][3])), tol_mul));
                REQUIRE(abs(cos(kep2[3])) == approximately(abs(cos(v_kep[i][3])), tol_mul));

                // Same Omega.
                REQUIRE(kep1[4] == approximately(v_kep[i][4], tol_mul));
                REQUIRE(kep2[4] == approximately(v_kep[i][4], tol_mul));
            }
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
