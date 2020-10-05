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
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

using namespace heyoka;
using namespace heyoka_benchmark;

template <typename T>
void run_bench(T tol, bool high_accuracy, bool compact_mode)
{
    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    const auto kep = std::array{T(1.5), T(.2), T(.3), T(.4), T(.5), T(.6)};
    const auto [c_x, c_v] = kep_to_cart(kep, T(1) / 4);

    std::vector init_state{c_v[0], -c_v[0], c_v[1], -c_v[1], c_v[2], -c_v[2],
                           c_x[0], -c_x[0], c_x[1], -c_x[1], c_x[2], -c_x[2]};

    auto start = std::chrono::high_resolution_clock::now();

    auto tad = taylor_adaptive<T>{{x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                   -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                  std::move(init_state),
                                  kw::high_accuracy = high_accuracy,
                                  kw::tol = tol,
                                  kw::compact_mode = compact_mode};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";

    start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < 4000; ++i) {
        tad.step();
    }

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Elapsed time for a single timestep: " << elapsed / 4000 << "ns\n";
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    bool high_accuracy = false;
    double tol;
    bool compact_mode = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be automatically deduced)")(
        "high_accuracy", "high-accuracy mode")("compact_mode", "compact mode");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("high_accuracy")) {
        high_accuracy = true;
    }

    if (vm.count("compact_mode")) {
        compact_mode = true;
    }

    if (fp_type == "double") {
        run_bench<double>(tol, high_accuracy, compact_mode);
    } else if (fp_type == "long double") {
        run_bench<long double>(tol, high_accuracy, compact_mode);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_bench<mppp::real128>(mppp::real128(tol), high_accuracy, compact_mode);
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }
}
