// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

template <typename T>
void run_integration()
{
    using std::abs;
    using std::log10;
    using std::pow;
    using namespace heyoka;
    using namespace heyoka_benchmark;

    auto masses = std::vector<T>{1.989e30, 1.898e27, 5.683e26, 8.681e25, 1.024e26, 1.31e22};

    const auto G = T(6.674e-11);

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    // NOTE: data taken from JPL's horizon.
    taylor_adaptive<T> ta{std::move(sys),
                          std::vector<T>{// Sun.
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
                                         5.123505032114531e+03, 1.028250197589481e+03, -1.595852584061212e+03},
                          kw::high_accuracy = true};

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

        return vNd<T, 3>{com_x[0], com_y[0], com_z[0]};
    };

    auto get_com_v = [&s_array, &m_array, &tot_mass]() {
        auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3)) / tot_mass;
        auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4)) / tot_mass;
        auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5)) / tot_mass;

        return vNd<T, 3>{com_vx[0], com_vy[0], com_vz[0]};
    };

    // Helper to compute the total energy in the system.
    auto get_energy = [&s_array, &m_array, G]() {
        // Kinetic energy.
        T kin = 0;
        for (auto i = 0u; i < 6u; ++i) {
            const auto v = xt::view(s_array, i, xt::range(3, 6));

            kin += T{1} / 2 * m_array[i] * xt::linalg::dot(v, v)[0];
        }

        // Potential energy.
        T pot = 0;
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

    // Base-10 logs of the initial and final saving times.
    const auto start_time = T(0), final_time = log10(T(1E8 * 365 * 86400));
    // Number of snapshots to take.
    const auto n_snaps = 10000u;
    // Build the vector of log10 saving times.
    std::vector<T> save_times;
    save_times.push_back(start_time);
    for (auto i = 1u; i < n_snaps; ++i) {
        save_times.push_back(save_times.back() + (final_time - start_time) / (n_snaps - 1u));
    }
    for (auto &t : save_times) {
        t = pow(T(10), t);
    }

    std::ofstream of("outer_ss_long_term.txt");
    of.precision(std::numeric_limits<T>::max_digits10);
    auto it = save_times.begin();
    while (ta.get_time() < pow(T(10), final_time)) {
        if (it != save_times.end() && ta.get_time() >= *it) {
            // We are at or past the current saving time, record
            // the time, energy error and orbital elements.
            of << ta.get_time() << " " << abs((init_energy - get_energy()) / init_energy) << " ";

            // Store the state.
            for (auto val : s_array) {
                of << val << " ";
            }

            // Store the orbital elements wrt the Sun.
            for (auto i = 1u; i < 6u; ++i) {
                auto rel_x = xt::view(s_array, i, xt::range(0, 3)) - xt::view(s_array, 0, xt::range(0, 3));
                auto rel_v = xt::view(s_array, i, xt::range(3, 6)) - xt::view(s_array, 0, xt::range(3, 6));

                auto kep = cart_to_kep(rel_x, rel_v, G * masses[0]);

                for (auto oe : kep) {
                    of << oe << " ";
                }
            }

            of << std::endl;

            // Locate the next saving time (that is, the first saving
            // time which is greater than the current time).
            it = std::upper_bound(it, save_times.end(), ta.get_time());
        }
        auto [res, h] = ta.step();
        if (res != taylor_outcome::success) {
            throw std::runtime_error("Error status detected: " + std::to_string(static_cast<int>(res)));
        }
    }
}

int main(int argc, char *argv[])
{
    run_integration<double>();
}
