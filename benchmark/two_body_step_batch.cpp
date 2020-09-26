// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_benchmark;

template <typename T>
void run_bench(T tol, bool high_accuracy, std::uint32_t batch_size)
{
    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    // The system of equations.
    auto sys = {x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3,
                vx0,          vx1,           vy0,          vy1,           vz0,          vz1};

    // Generate a bunch of random initial conditions in orbital elements.
    std::vector<std::array<T, 6>> v_kep;
    std::uniform_real_distribution<double> a_dist(0.1, 10.), e_dist(0.1, 0.5), i_dist(0.1, 3.13), ang_dist(0.1, 6.28);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        v_kep.push_back(std::array{T(a_dist(rng)), T(e_dist(rng)), T(i_dist(rng)), T(ang_dist(rng)), T(ang_dist(rng)),
                                   T(ang_dist(rng))});
    }

    // Generate the initial state/time vector for the batch integrator.
    std::vector<T> init_states(batch_size * 12u);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        const auto [x, v] = kep_to_cart(v_kep[i], T(1) / 4);

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

    auto start = std::chrono::high_resolution_clock::now();

    // Init the batch integrator.
    auto tad = (tol == T(0)) ? taylor_adaptive_batch<T>{sys, std::move(init_states), batch_size,
                                                        kw::high_accuracy = high_accuracy}
                             : taylor_adaptive_batch<T>{sys, std::move(init_states), batch_size,
                                                        kw::high_accuracy = high_accuracy, kw::tol = tol};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";

    std::vector<std::tuple<taylor_outcome, T>> res(batch_size);

    start = std::chrono::high_resolution_clock::now();

    // Do 4000 steps.
    for (auto i = 0; i < 4000; ++i) {
        tad.step(res);
    }

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Elapsed time for a single timestep per batch element: " << elapsed / 4000 / batch_size << "ns\n";
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    bool high_accuracy = false;
    double tol;
    std::uint32_t batch_size;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be automatically deduced)")(
        "high_accuracy", "high-accuracy mode")("batch_size", po::value<std::uint32_t>(&batch_size)->default_value(1u),
                                               "batch size");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // Validate the command-line arguments.
    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size cannot be zero");
    }

    if (vm.count("high_accuracy")) {
        high_accuracy = true;
    }

    if (fp_type == "double") {
        run_bench<double>(tol, high_accuracy, batch_size);
    } else if (fp_type == "long double") {
        run_bench<long double>(tol, high_accuracy, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_bench<mppp::real128>(mppp::real128(tol), high_accuracy, batch_size);
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }
}
