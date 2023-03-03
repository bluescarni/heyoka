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
#include <random>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    unsigned seed;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("seed", po::value<unsigned>(&seed)->default_value(42u),
                                                       "random seed");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::mt19937 rng;
    rng.seed(seed);

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>({prime(x) = v, prime(v) = -x}, {0, 1});

    const int ntrials = 100;
    const auto final_time = 10000.;

    double err = 0;

    std::uniform_real_distribution rdist(-1e-9, 1e-9);

    for (auto i = 0; i < ntrials; ++i) {
        const auto v0 = 1. + rdist(rng);

        ta.set_time(0);
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = v0;

        ta.propagate_until(final_time);

        const auto exact = v0 * std::sin(final_time);
        err += std::abs((exact - ta.get_state()[0]) / exact);
    }

    std::cout << "Average relative error: " << err / ntrials << '\n';
}
