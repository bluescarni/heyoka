// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>

#include <boost/program_options.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <mp++/real128.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    using std::abs;

    namespace po = boost::program_options;
    using mppp::real128;

    double tol = 0.;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("tol", po::value<double>(&tol)->default_value(0.), "tolerance");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
        std::cout << desc << "\n";
        return 0;
    }

    auto compute_energy = [](const auto &sv) {
        using std::cos;

        return (sv[1] * sv[1]) / 2 + 9.8 * (1 - cos(sv[0]));
    };

    // Fetch the logger.
    create_logger();
    auto logger = spdlog::get("heyoka");

    warmup();

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<real128>({prime(x) = v, prime(v) = -9.8 * sin(x)}, {real128(1.), real128(0)},
                                       kw::tol = real128(tol));

    const auto E0 = compute_energy(ta.get_state());

    set_logger_level_trace();

    spdlog::stopwatch sw;

    ta.propagate_until(real128(1));

    logger->trace("Integration time: {}", sw);
    logger->trace("Rel. energy error: {}", abs((E0 - compute_energy(ta.get_state())) / E0));

    std::cout << ta << '\n';
}
