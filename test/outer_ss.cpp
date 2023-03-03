// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/detail/simple_timer.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("outer solar system")
{
    using fp_t = double;

    using std::abs;

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895;

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);

    for (auto cm : {false, true}) {
        for (auto ha : {false, true}) {
            // NOTE: data taken from JPL's horizon.
            taylor_adaptive<fp_t> ta{sys,
                                     {// Sun.
                                      -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6,
                                      +6.69048890636161e-6, -6.33922479583593e-6, -3.13202145590767e-9,
                                      // Jupiter.
                                      +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2,
                                      -5.59797969310664e-3, +5.51815399480116e-3, -2.66711392865591e-6,
                                      // Saturn.
                                      +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1,
                                      -4.17354020307064e-3, +3.99723751748116e-3, +1.67206320571441e-5,
                                      // Uranus.
                                      +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1,
                                      -3.25884806151064e-3, +2.06438412905916e-3, -2.17699042180559e-5,
                                      // Neptune.
                                      -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1,
                                      -2.17471785045538e-4, -3.11361111025884e-3, +3.58344705491441e-5,
                                      // Pluto.
                                      -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0,
                                      -1.76936577252484e-3, -2.06720938381724e-3, +6.58091931493844e-4},
                                     kw::high_accuracy = ha,
                                     kw::compact_mode = cm};

            // Create xtensor views on the the state and mass vectors
            // for ease of indexing.
            auto s_array = xt::adapt(ta.get_state_data(), {6, 6});
            auto m_array = xt::adapt(masses.data(), {6});

            // Cache the total mass.
            const auto tot_mass = xt::sum(m_array)[0];

            // Helpers to compute the position and velocity of the COM.
            auto get_com = [&s_array, &m_array, &tot_mass]() {
                auto com_x = xt::sum(m_array * xt::view(s_array, xt::all(), 0)) / tot_mass;
                auto com_y = xt::sum(m_array * xt::view(s_array, xt::all(), 1)) / tot_mass;
                auto com_z = xt::sum(m_array * xt::view(s_array, xt::all(), 2)) / tot_mass;

                return vNd<fp_t, 3>{com_x[0], com_y[0], com_z[0]};
            };

            auto get_com_v = [&s_array, &m_array, &tot_mass]() {
                auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3)) / tot_mass;
                auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4)) / tot_mass;
                auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5)) / tot_mass;

                return vNd<fp_t, 3>{com_vx[0], com_vy[0], com_vz[0]};
            };

            // Helper to compute the total energy in the system.
            auto get_energy = [&s_array, &m_array, G]() {
                // Kinetic energy.
                fp_t kin = 0;
                for (auto i = 0u; i < 6u; ++i) {
                    const auto v = xt::view(s_array, i, xt::range(3, 6));

                    kin += fp_t{1} / 2 * m_array[i] * xt::linalg::dot(v, v)[0];
                }

                // Potential energy.
                fp_t pot = 0;
                for (auto i = 0u; i < 6u; ++i) {
                    const auto ri = xt::view(s_array, i, xt::range(0, 3));

                    for (auto j = i + 1u; j < 6u; ++j) {
                        const auto rj = xt::view(s_array, j, xt::range(0, 3));

                        pot -= G * m_array[i] * m_array[j] / xt::linalg::norm(ri - rj);
                    }
                }

                return kin + pot;
            };

            // Compute position and velocity of the COM.
            const auto init_com = get_com();
            const auto init_com_v = get_com_v();

            std::cout << "Initial COM         : " << init_com << '\n';
            std::cout << "Initial COM velocity: " << init_com_v << '\n';

            // Offset the existing positions/velocities so that they refer
            // to the COM.
            xt::view(s_array, 0, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 1, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 2, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 3, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 4, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 5, xt::range(0, 3)) -= init_com;

            xt::view(s_array, 0, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 1, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 2, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 3, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 4, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 5, xt::range(3, 6)) -= init_com_v;

            std::cout << "New COM         : " << get_com() << '\n';
            std::cout << "New COM velocity: " << get_com_v() << '\n';

            const auto init_energy = get_energy();
            std::cout << "Initial energy: " << init_energy << '\n';

            {
                detail::simple_timer st{"Integration time"};

                for (auto i = 0ul; ta.get_time() < 1e6 * 365; ++i) {
                    if (i % 100000 == 0) {
                        std::cout << "Energy diff : " << abs((init_energy - get_energy()) / init_energy) << '\n';
                    }

                    // NOTE: uncomment to print the orbital elements very 1000 steps.
                    // if (i % 1000 == 0) {
                    //     std::cout << "jup: "
                    //               << cart_to_kep(xt::view(s_array, 1, xt::range(0, 3)), xt::view(s_array, 1,
                    //               xt::range(3, 6)),
                    //                              sun_mu)
                    //               << '\n';
                    //     std::cout << "sat: "
                    //               << cart_to_kep(xt::view(s_array, 2, xt::range(0, 3)), xt::view(s_array, 2,
                    //               xt::range(3, 6)),
                    //                              sun_mu)
                    //               << '\n';
                    // }

                    const auto step_res = ta.step();
                    REQUIRE(std::get<0>(step_res) == taylor_outcome::success);

                    // if (i % 100000 == 0) {
                    //     std::cout << std::get<1>(step_res) << '\n';
                    // }
                }
            }

            std::cout << "Final time: " << ta.get_time() << '\n';

            std::cout << "COM         : " << get_com() << '\n';
            std::cout << "COM velocity: " << get_com_v() << '\n';
            std::cout << "Energy diff : " << abs((init_energy - get_energy()) / init_energy) << '\n';
        }
    }
}
