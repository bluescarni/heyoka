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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

template <typename T>
void run_benchmark(double tol, bool high_accuracy, unsigned ntrials)
{
    using std::abs;
    using std::sqrt;

    std::mt19937 rng(std::random_device{}());

    const std::vector<T> masses = {5., 4., 3.},
                         ic = {1., -1., 0., 0., 0., 0., -2., -1., 0., 0., 0., 0., 1., 3., 0., 0., 0., 0.};

    auto sys = model::nbody(3, kw::masses = masses);

    auto ta = taylor_adaptive<T>{sys, ic, kw::tol = tol, kw::high_accuracy = high_accuracy};

    auto s_array = xt::adapt(ta.get_state_data(), {3, 6});
    auto m_array = xt::adapt(masses.data(), {3});

    auto get_energy = [&s_array, &m_array, G = static_cast<T>(1)]() {
        // Kinetic energy.
        T kin(0), c(0);
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
        T pot(0);
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

    auto start = std::chrono::high_resolution_clock::now();

    auto res = ta.propagate_for(63);

    auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::high_resolution_clock::now() - start)
                                           .count())
                   / 1e6;

    std::cout << "Outcome: " << std::get<0>(res) << '\n';
    std::cout << "Number of steps: " << std::get<3>(res) << '\n';
    std::cout << "Integration time: " << elapsed << "ms\n";

    T err_acc = 0.;

    for (auto _ = 0u; _ < ntrials; ++_) {
        // Reset time.
        ta.set_time(0);

        // Generate new state.
        std::uniform_real_distribution<double> rdist(1e-13, 1e-12);
        std::uniform_int_distribution<int> idist(0, 1);
        for (auto i = 0u; i < 18u; ++i) {
            const auto pert = rdist(rng);
            s_array[i] = ic[i] + ic[i] * (idist(rng) == 0 ? pert : -pert);
        }

        const auto init_energy = get_energy();

        ta.propagate_for(63);

        err_acc += abs((init_energy - get_energy()) / init_energy);
    }

    std::cout << "Average relative energy error: " << err_acc / ntrials << '\n';
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double tol = 0.;
    unsigned ntrials = 0;
    bool high_accuracy = false;
    std::string fp_type;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-12),
                                                       "tolerance")("high_accuracy", "enable high accuracy mode")(
        "ntrials", po::value<unsigned>(&ntrials)->default_value(100u), "number of trials")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("high_accuracy") != 0u) {
        high_accuracy = true;
    }

    if (fp_type == "double") {
        run_benchmark<double>(tol, high_accuracy, ntrials);
    } else {
        if (fp_type != "long double") {
            throw std::invalid_argument(R"(Only the "double" and "long double" floating-point types are supported)");
        }

        run_benchmark<long double>(tol, high_accuracy, ntrials);
    }
}
