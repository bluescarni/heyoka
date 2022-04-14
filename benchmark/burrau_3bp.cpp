// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int, char *[])
{
    using std::abs;
    using std::sqrt;

    std::vector masses = {5., 4., 3.};

    auto sys = make_nbody_sys(3, kw::masses = masses);

    auto ta = taylor_adaptive<double>{sys,
                                      {1., -1., 0., 0., 0., 0., -2., -1., 0., 0., 0., 0., 1., 3., 0., 0., 0., 0.},
                                      kw::tol = 1e-12,
                                      kw::high_accuracy = true};

    auto s_array = xt::adapt(ta.get_state_data(), {3, 6});
    auto m_array = xt::adapt(masses.data(), {3});

    auto start = std::chrono::high_resolution_clock::now();

    auto get_energy = [&s_array, &m_array, G = 1.]() {
        // Kinetic energy.
        double kin(0), c(0);
        for (auto i = 0u; i < 3u; ++i) {
            auto vx = xt::view(s_array, i, 3)[0];
            auto vy = xt::view(s_array, i, 4)[0];
            auto vz = xt::view(s_array, i, 5)[0];

            auto tmp = 1. / 2 * m_array[i] * (vx * vx + vy * vy + vz * vz);
            auto y = tmp - c;
            auto t = kin + y;
            c = (t - kin) - y;
            kin = t;
        }

        // Potential energy.
        double pot(0);
        c = 0;
        for (auto i = 0u; i < 3u; ++i) {
            auto xi = xt::view(s_array, i, 0)[0];
            auto yi = xt::view(s_array, i, 1)[0];
            auto zi = xt::view(s_array, i, 2)[0];

            for (auto j = i + 1u; j < 3u; ++j) {
                auto xj = xt::view(s_array, j, 0)[0];
                auto yj = xt::view(s_array, j, 1)[0];
                auto zj = xt::view(s_array, j, 2)[0];

                auto tmp = -G * m_array[i] * m_array[j]
                           / sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj));
                auto y = tmp - c;
                auto t = pot + y;
                c = (t - pot) - y;
                pot = t;
            }
        }

        return kin + pot;
    };

    const auto init_energy = get_energy();
    std::cout << "Initial energy: " << init_energy << '\n';

    auto res = ta.propagate_for(63.);

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Outcome: " << std::get<0>(res) << '\n';
    std::cout << "Number of steps: " << std::get<3>(res) << '\n';
    std::cout << "Integration time: " << elapsed << "ms\n";
    std::cout << "Final energy error: " << abs((init_energy - get_energy()) / init_energy) << '\n';
}
