// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <boost/program_options.hpp>

#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::uint32_t n_bodies;
    bool compact_mode = false;
    bool function_inlining = true;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("n", po::value<std::uint32_t>(&n_bodies)->default_value(2),
                                                       "number of bodies")("compact_mode", "compact mode")(
        "disable_inlining", "disable function inlining");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("compact_mode")) {
        compact_mode = true;
    }

    if (vm.count("disable_inlining")) {
        function_inlining = false;
    }

    std::vector<double> init_state(6u * n_bodies);
    std::iota(init_state.begin(), init_state.end(), 1.);

    auto start = std::chrono::high_resolution_clock::now();

    taylor_adaptive<double> ta{make_nbody_sys(n_bodies), std::move(init_state), kw::high_accuracy = true,
                               kw::compact_mode = compact_mode, kw::inline_functions = function_inlining};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    auto counter = 0u;
    for (const auto &ex : ta.get_decomposition()) {
        std::cout << "u_" << counter++ << " = " << ex << '\n';
    }

    std::cout << "Construction time: " << elapsed << "ms\n";
}
