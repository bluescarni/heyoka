// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double tol;
    bool disable_event = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-15),
                                                       "tolerance")("disable_event", "disable event detection");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("disable_event")) {
        disable_event = true;
    }

    warmup();

    // Create the symbolic variables.
    auto [vx, vy, x, y] = make_vars("vx", "vy", "x", "y");

    // Initial conditions.
    auto ic = {-0.2525875586263492, -0.2178423952983717, 0., 0.2587703282931232};

    // The list of intersect times.
    std::vector<double> ix_vals;

    // The callback.
    auto cb = [&ix_vals](auto &, double tm, int) { ix_vals.push_back(tm); };

    // The event.
    auto ev = nt_event<double>(x, cb, kw::direction = event_direction::positive);

    auto ta = disable_event ? taylor_adaptive<double>{{prime(vx) = -x - 2. * x * y, prime(vy) = y * y - y - x * x,
                                                       prime(x) = vx, prime(y) = vy},
                                                      ic,
                                                      kw::tol = tol}
                            : taylor_adaptive<double>{{prime(vx) = -x - 2. * x * y, prime(vy) = y * y - y - x * x,
                                                       prime(x) = vx, prime(y) = vy},
                                                      ic,
                                                      kw::tol = tol,
                                                      kw::nt_events = {ev}};

    auto start = std::chrono::high_resolution_clock::now();

    const auto oc = std::get<0>(ta.propagate_until(2000.));

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Runtime: " << elapsed << "μs\n";
    std::cout << "Outcome: " << oc << '\n';

    if (!disable_event) {
        std::cout << "Number of intersections: " << ix_vals.size() << '\n';
        std::cout.precision(16);
        std::cout << "First 3 crossings: " << ix_vals[0] << ", " << ix_vals[1] << ", " << ix_vals[2] << '\n';
    }

    return 0;
}
