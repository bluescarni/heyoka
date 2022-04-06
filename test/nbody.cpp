// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/detail/simple_timer.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("N-body param")
{
    for (const auto &[_, ex] : make_nbody_par_sys(2)) {
        std::cout << ex << '\n';
    }

    for (const auto &[_, ex] : make_nbody_par_sys(20, kw::n_massive = 3)) {
        std::cout << ex << '\n';
    }

    // Exercise fixed masses + massless particles.
    const auto G = 6.674e-11;
    auto masses = std::vector{1.989e30, 1.898e27, 0.};
    auto sys = make_nbody_sys(3, kw::masses = masses, kw::Gconst = G);
}

// Test case for an issue that arised when using
// null masses.
TEST_CASE("zero mass")
{
    auto sys = make_nbody_sys(2, kw::masses = {1., 0.});

    auto tad = taylor_adaptive<double>{std::move(sys), {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}};

    auto [oc, _1, _2, _3, _4] = tad.propagate_until(10000);

    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(std::isfinite(tad.get_time()));
    REQUIRE(
        std::all_of(tad.get_state().begin(), tad.get_state().end(), [](const auto &x) { return std::isfinite(x); }));
}

TEST_CASE("N-body")
{
    using fp_t = double;

    using std::abs;

    auto masses = std::vector{1.989e30, 1.898e27, 5.683e26};

    const auto sun_mu = 1.327e20;
    const auto G = 6.674e-11;

    auto sys = make_nbody_sys(3, kw::masses = masses, kw::Gconst = G);

    for (auto cm : {false, true}) {
        for (auto ha : {false, true}) {
            // Initial states in orbital elements for Jupiter and Saturn.
            // NOTE: a, e, i are realistic, the angles are random.
            const auto j_kep = std::array{778.57e9, 0.0489, 0.02274164, .1, .2, .3};
            const auto [j_x, j_v] = kep_to_cart(j_kep, sun_mu);

            const auto s_kep = std::array{1433.53e9, 0.0565, 0.043371432, .4, .5, .6};
            const auto [s_x, s_v] = kep_to_cart(s_kep, sun_mu);

            taylor_adaptive<fp_t> ta{sys,
                                     {// Sun in the origin, zero speed.
                                      0, 0, 0, 0, 0, 0,
                                      // Jupiter.
                                      j_x[0], j_x[1], j_x[2], j_v[0], j_v[1], j_v[2],
                                      // Saturn.
                                      s_x[0], s_x[1], s_x[2], s_v[0], s_v[1], s_v[2]},
                                     kw::high_accuracy = ha,
                                     kw::compact_mode = cm,
                                     kw::tol = 1e-18};

            // Create xtensor views on the the state and mass vectors
            // for ease of indexing.
            auto s_array = xt::adapt(ta.get_state_data(), {3, 6});
            auto m_array = xt::adapt(masses.data(), {3});

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
                for (auto i = 0u; i < 3u; ++i) {
                    const auto v = xt::view(s_array, i, xt::range(3, 6));

                    kin += fp_t{1} / 2 * m_array[i] * xt::linalg::dot(v, v)[0];
                }

                // Potential energy.
                fp_t pot = 0;
                for (auto i = 0u; i < 3u; ++i) {
                    const auto ri = xt::view(s_array, i, xt::range(0, 3));

                    for (auto j = i + 1u; j < 3u; ++j) {
                        const auto rj = xt::view(s_array, j, xt::range(0, 3));

                        pot -= G * m_array[i] * m_array[j] / xt::linalg::norm(ri - rj);
                    }
                }

                return kin + pot;
            };

            // Compute position and velocity of the COM.
            const auto init_com = get_com();
            const auto init_com_v = get_com_v();

            // Offset the existing positions/velocities so that they refer
            // to the COM.
            xt::view(s_array, 0, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 1, xt::range(0, 3)) -= init_com;
            xt::view(s_array, 2, xt::range(0, 3)) -= init_com;

            xt::view(s_array, 0, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 1, xt::range(3, 6)) -= init_com_v;
            xt::view(s_array, 2, xt::range(3, 6)) -= init_com_v;

            std::cout << "New COM         : " << get_com() << '\n';
            std::cout << "New COM velocity: " << get_com_v() << '\n';

            const auto init_energy = get_energy();
            std::cout << "Initial energy: " << init_energy << '\n';

            {
                detail::simple_timer st{"Integration time"};

                for (auto i = 0ul; ta.get_time() < 1e6 * 86400 * 365; ++i) {
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

TEST_CASE("(N+1)-body")
{
    using Catch::Matchers::Message;

    // Failure modes.
    REQUIRE_THROWS_MATCHES(make_np1body_sys(0), std::invalid_argument,
                           Message("At least 1 body is needed to construct an (N+1)-body system"));
    REQUIRE_THROWS_MATCHES(
        make_np1body_sys(1, kw::masses = {1.}), std::invalid_argument,
        Message(
            "Inconsistent sizes detected while creating an (N+1)-body system: the vector of masses has a size of "
            "1, while the number of bodies is 1 (the number of masses must be one more than the number of bodies)"));

    auto sys = make_np1body_sys(1);

    REQUIRE(sys.size() == 6u);
    REQUIRE(sys[0].first == "x_1"_var);
    REQUIRE(sys[0].second == "vx_1"_var);
    REQUIRE(sys[1].first == "y_1"_var);
    REQUIRE(sys[1].second == "vy_1"_var);
    REQUIRE(sys[2].first == "z_1"_var);
    REQUIRE(sys[2].second == "vz_1"_var);

    REQUIRE(sys[3].first == "vx_1"_var);
    REQUIRE(sys[3].second == -2_dbl * ("x_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));
    REQUIRE(sys[4].first == "vy_1"_var);
    REQUIRE(sys[4].second == -2_dbl * ("y_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));
    REQUIRE(sys[5].first == "vz_1"_var);
    REQUIRE(sys[5].second == -2_dbl * ("z_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));

    sys = make_np1body_sys(1, kw::masses = {2., 1.});

    REQUIRE(sys.size() == 6u);
    REQUIRE(sys[0].first == "x_1"_var);
    REQUIRE(sys[0].second == "vx_1"_var);
    REQUIRE(sys[1].first == "y_1"_var);
    REQUIRE(sys[1].second == "vy_1"_var);
    REQUIRE(sys[2].first == "z_1"_var);
    REQUIRE(sys[2].second == "vz_1"_var);

    REQUIRE(sys[3].first == "vx_1"_var);
    REQUIRE(sys[3].second == -3_dbl * ("x_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));
    REQUIRE(sys[4].first == "vy_1"_var);
    REQUIRE(sys[4].second == -3_dbl * ("y_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));
    REQUIRE(sys[5].first == "vz_1"_var);
    REQUIRE(sys[5].second == -3_dbl * ("z_1"_var * pow(sum_sq({"x_1"_var, "y_1"_var, "z_1"_var}), -3_dbl / 2_dbl)));

    // Integrate the usual outer Solar System setup in both the N-body and (N+1)-body
    // formulations, and compare the results.

    const auto Gconst = 0.01720209895 * 0.01720209895 * 365 * 365;

    const auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto n_ic = std::vector{// Sun.
                                  -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6,
                                  +6.69048890636161e-6 * 365, -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                                  // Jupiter.
                                  +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2,
                                  -5.59797969310664e-3 * 365, +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                                  // Saturn.
                                  +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1,
                                  -4.17354020307064e-3 * 365, +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                                  // Uranus.
                                  +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1,
                                  -3.25884806151064e-3 * 365, +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                                  // Neptune.
                                  -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1,
                                  -2.17471785045538e-4 * 365, -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                                  // Pluto.
                                  -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0,
                                  -1.76936577252484e-3 * 365, -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};

    std::vector<double> np1_ic;
    for (auto i = 0u; i < 5u; ++i) {
        for (auto j = 0u; j < 6u; ++j) {
            np1_ic.push_back(n_ic[(i + 1u) * 6u + j] - n_ic[j]);
        }
    }

    auto nsys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = Gconst);
    auto np1sys = make_np1body_sys(5, kw::masses = masses, kw::Gconst = Gconst);

    taylor_adaptive<double> ta_n{std::move(nsys), n_ic, kw::high_accuracy = true, kw::tol = 1e-18};
    taylor_adaptive<double> ta_np1{std::move(np1sys), np1_ic, kw::high_accuracy = true, kw::tol = 1e-18};

    // Correct the state in ta_n to put the barycentre in the origin.
    auto s_array = xt::adapt(ta_n.get_state_data(), {6, 6});
    auto m_array = xt::adapt(masses.data(), {6});

    // Cache the total mass.
    const auto tot_mass = xt::sum(m_array)[0];

    // Helpers to compute the position and velocity of the COM.
    auto get_com = [&s_array, &m_array, &tot_mass]() {
        auto com_x = xt::sum(m_array * xt::view(s_array, xt::all(), 0)) / tot_mass;
        auto com_y = xt::sum(m_array * xt::view(s_array, xt::all(), 1)) / tot_mass;
        auto com_z = xt::sum(m_array * xt::view(s_array, xt::all(), 2)) / tot_mass;

        return vNd<double, 3>{com_x[0], com_y[0], com_z[0]};
    };

    auto get_com_v = [&s_array, &m_array, &tot_mass]() {
        auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3)) / tot_mass;
        auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4)) / tot_mass;
        auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5)) / tot_mass;

        return vNd<double, 3>{com_vx[0], com_vy[0], com_vz[0]};
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

    // Integrate for 100 years.
    REQUIRE(std::get<0>(ta_n.propagate_for(100.)) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta_np1.propagate_for(100.)) == taylor_outcome::time_limit);

    // Compare.
    for (auto i = 0u; i < 5u; ++i) {
        for (auto j = 0u; j < 6u; ++j) {
            REQUIRE(ta_n.get_state()[(i + 1u) * 6u + j] - ta_n.get_state()[j]
                    == approximately(ta_np1.get_state()[i * 6u + j], 10000.));
        }
    }
}
