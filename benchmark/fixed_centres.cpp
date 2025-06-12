// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    // Total number of masses.
    auto nmasses = 0u;
    // Integration tolerance.
    auto tol = 0.;
    // Number of iterations to be run in parallel.
    std::size_t npariter = 0u;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(1e-15),
                                                       "tolerance")(
        "nmasses", po::value<unsigned>(&nmasses)->default_value(10u), "total number of point masses")(
        "npariter", po::value<std::size_t>(&npariter)->default_value(1u), "number of iterations to be run in parallel");

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

    if (nmasses == 0u) {
        throw std::invalid_argument("The number of point masses cannot be zero");
    }

    // Random number generator and distributions.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0., 1.);

    // The mass of a single point.
    const auto pmass = 1. / nmasses;

    // Generate the positions.
    std::vector<double> pos;
    for (auto i = 0u; i < nmasses; ++i) {
        // https://mathworld.wolfram.com/SpherePointPicking.html
        const auto u = dist(rng);
        const auto v = dist(rng);
        const auto r = dist(rng) / 100.;
        const auto theta = 2 * boost::math::constants::pi<double>() * u;
        const auto phi = std::acos((2 * v) - 1);

        pos.push_back(r * std::cos(theta) * std::sin(phi));
        pos.push_back(r * std::sin(theta) * std::sin(phi));
        pos.push_back(r * std::cos(phi));
    }

    // Generate the vector of masses.
    const std::vector<double> masses(nmasses, pmass);

    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Generate the dynamics.
    const auto dyn = model::fixed_centres(kw::masses = masses, kw::positions = pos);

    // Initial conditions.
    const auto ic = std::vector{1., 0., 0., 0., 1., 0.};

    // Init the integrator.
    auto ta = taylor_adaptive(dyn, ic, kw::compact_mode = true, kw::tol = 1e-15);

    // Total integration time.
    const auto final_time = 2 * boost::math::constants::pi<double>() * 20;

    logger->trace("Decomposition size: {}", ta.get_decomposition().size());

    // Propagate.
    for (auto i = 0; i < 1000; ++i) {
        spdlog::stopwatch sw;
        ta.set_time(0.);
        std::ranges::copy(ic, ta.get_state_data());

        if (npariter == 1u) {
            ta.propagate_until(final_time);
            std::cout << ta << '\n';
        } else {
            const auto gen = std::function<taylor_adaptive<double>(taylor_adaptive<double>, std::size_t)>(
                [](auto ta, auto) { return ta; });
            ensemble_propagate_until(ta, final_time, npariter, gen);
        }

        logger->trace("Total runtime: {}s", sw);
    }
}
