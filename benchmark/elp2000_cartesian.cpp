// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <boost/program_options.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/elp2000.hpp>

using namespace heyoka;
using namespace heyoka::model;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double thresh{};

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("thresh", po::value<double>(&thresh)->default_value(1e-7),
                                                       "threshold value");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    spdlog::stopwatch sw;
    auto sol = model::elp2000_cartesian_e2000(kw::thresh = thresh);
    logger->trace("Creating the solution took: {}", sw);

    sw.reset();
    cfunc<double> cf(sol, {}, kw::compact_mode = true);
    logger->trace("Compiling the solution took: {}", sw);

    std::vector<double> out(3u, 0.);

    for (auto date : {2469000.5, 2449000.5, 2429000.5, 2409000.5, 2389000.5}) {
        const double tm = (date - 2451545.0) / (36525);

        sw.reset();
        cf(out, std::vector<double>{}, kw::time = tm);
        logger->trace("Computing the solution took: {}", sw);
        logger->trace("State for date {}: {}", date, out);
    }
}
