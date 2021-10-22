// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <utility>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double tol;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-15),
                                                       "tolerance");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    warmup();

    auto masses = {1.00000597682, 1 / 1047.355, 1 / 3501.6};
    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_sys(3, kw::masses = masses, kw::Gconst = G);

    auto ic = {// Sun.
               -5.137271893918405e-03, -5.288891104344273e-03, 6.180743702483316e-06, 2.3859757364179156e-03,
               -2.3396779489468049e-03, -8.1384891821122709e-07,
               // Jupiter.
               3.404393156051084, 3.6305811472186558, 0.0342464685434024, -2.0433186406983279e+00,
               2.0141003472039567e+00, -9.7316724504621210e-04,
               // Saturn.
               6.606942557811084, 6.381645992310656, -0.1361381213577972, -1.5233982268351876, 1.4589658329821569,
               0.0061033600397708};

    taylor_adaptive<double> ta{std::move(sys), ic, kw::tol = tol};

    auto start = std::chrono::high_resolution_clock::now();
    auto oc = std::get<0>(ta.propagate_until(100000.));
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
    std::cout << "Outcome: " << oc << '\n';

    std::cout << ta << '\n';

    return 0;
}
