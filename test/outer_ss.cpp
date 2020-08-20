// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("outer solar system")
{
    using fp_t = double;

    using std::abs;

    auto masses = std::vector{1.989e30, 1.898e27, 5.683e26, 8.681e25, 1.024e26, 1.31e22};

    const auto G = 6.674e-11;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    // NOTE: data taken from JPL's horizon.
    taylor_adaptive<fp_t> ta{std::move(sys),
                             {// Sun.
                              -8.499879096181407e+08, 1.001735774939453e+09, 1.141999745007278e+07,
                              -1.328415580747554e+01, -7.891174237485232e+00, 3.880125037158690e-01,
                              // Jupiter.
                              3.262371692321718e+11, -6.953287940553470e+11, -4.414504748051047e+09,
                              1.166702199610602e+04, 6.168390278105449e+03, -2.866931853208339e+02,
                              // Saturn.
                              7.310395308416098e+11, -1.304862736935276e+12, -6.414749979299486e+09,
                              7.889926837929547e+03, 4.694479691694787e+03, -3.952237338526181e+02,
                              // Uranus.
                              2.344842648710289e+12, 1.806976824128821e+12, -2.366655583679092e+10,
                              -4.206778646012585e+03, 5.076829479408024e+03, 7.348196963714071e+01,
                              // Neptune.
                              4.395223603228156e+12, -8.440256467792909e+11, -8.391116288415986e+10,
                              9.891808958607942e+02, 5.370766039497987e+03, -1.328684463220895e+02,
                              // Pluto.
                              2.043470428523008e+12, -4.672388820183485e+12, -9.111734786375618e+10,
                              5.123505032114531e+03, 1.028250197589481e+03, -1.595852584061212e+03}};

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
            //               << cart_to_kep(xt::view(s_array, 1, xt::range(0, 3)), xt::view(s_array, 1, xt::range(3,
            //               6)),
            //                              sun_mu)
            //               << '\n';
            //     std::cout << "sat: "
            //               << cart_to_kep(xt::view(s_array, 2, xt::range(0, 3)), xt::view(s_array, 2, xt::range(3,
            //               6)),
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
