// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cstddef>
#include <iostream>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/vsop2013.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;
using namespace heyoka::model;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double thresh;
    std::uint32_t pl_idx, var_idx;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("thresh", po::value<double>(&thresh)->default_value(1e-9),
                                                       "threshold value")(
        "pl_idx", po::value<std::uint32_t>(&pl_idx)->default_value(1),
        "planet index")("var_idx", po::value<std::uint32_t>(&var_idx)->default_value(1), "variable index");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    auto x = make_vars("x");

    auto start = std::chrono::high_resolution_clock::now();

    auto series = vsop2013_elliptic(pl_idx, var_idx, kw::thresh = thresh, kw::time_expr = par[0]);

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Expression creation time: " << elapsed << "ms\n";

    start = std::chrono::high_resolution_clock::now();

    auto ta = taylor_adaptive<double>{{prime(x) = series}, {0.}, kw::compact_mode = true};

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integrator creation time: " << elapsed << "ms\n";

    std::cout << "Number of terms in the Taylor decomposition: " << ta.get_decomposition().size() - 2u << '\n';
}
