// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    // Max degree/order for the geopotential.
    std::uint32_t max_degree = 0;
    // Integration tolerance.
    auto tol = 0.;
    // Number of iterations to be run in parallel.
    std::size_t npariter = 0u;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-15),
                                                       "tolerance")(
        "max_degree", po::value<std::uint32_t>(&max_degree)->default_value(10u),
        "maximum degree/order for the geopotential")("npariter", po::value<std::size_t>(&npariter)->default_value(1u),
                                                     "number of iterations to be run in parallel");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.contains("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (npariter == 0u) {
        throw std::invalid_argument("The number of parallel iterations cannot be zero");
    }

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // State variables.
    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Acceleration vector.
    const auto [acc_x, acc_y, acc_z] = model::egm2008_acc({x, y, z}, max_degree, max_degree);

    // Initial conditions in LEO.
    const auto ic_leo = std::vector{6740440.0, 0.0, 0.0, 0.0, 6725.973853066024, 3883.2537950295855};

    // Init the integrator.
    auto ta = taylor_adaptive({{x, vx}, {y, vy}, {z, vz}, {vx, acc_x}, {vy, acc_y}, {vz, acc_z}}, ic_leo,
                              kw::compact_mode = true, kw::tol = 1e-15);

    logger->trace("Decomposition size: {}", ta.get_decomposition().size());

    // Propagate.
    for (auto i = 0; i < 1000; ++i) {
        spdlog::stopwatch sw;
        ta.set_time(0.);
        std::ranges::copy(ic_leo, ta.get_state_data());

        if (npariter == 1u) {
            ta.propagate_until(86400.);
        } else {
            const auto gen = std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)>(
                [](auto ta, auto) { return ta; });
            ensemble_propagate_until(ta, 86400., npariter, gen);
        }

        logger->trace("Total runtime: {}s", sw);
    }
}
