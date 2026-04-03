// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <iostream>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/egm2008.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    // Max degree/order for the geopotential.
    std::uint32_t max_degree = 0;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("max_degree",
                                                       po::value<std::uint32_t>(&max_degree)->default_value(10u),
                                                       "maximum degree/order for the geopotential");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.contains("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Positional variables.
    const auto [x, y, z] = make_vars("x", "y", "z");

    // Acceleration vector.
    const auto [acc_x, acc_y, acc_z] = model::egm2008_acc({x, y, z}, max_degree, max_degree);

    // Create the compiled function.
    const auto cf = cfunc<double>({acc_x, acc_y, acc_z}, {x, y, z}, kw::compact_mode = true);

    // Inputs.
    const auto inputs = std::vector{6740440.0, 0.0, 0.0};

    // Outputs.
    std::vector<double> outputs(3);

    // Evaluate.
    for (auto i = 0; i < 1000; ++i) {
        spdlog::stopwatch sw;

        cf(outputs, inputs);

        logger->trace("Total runtime: {}s", sw);
    }
}
