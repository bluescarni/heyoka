// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    // Fetch the logger.
    create_logger();
    auto logger = spdlog::get("heyoka");
    set_logger_level_trace();

    std::uint32_t order{};
    bool compact_mode = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("order", po::value<std::uint32_t>(&order)->default_value(2))(
        "compact_mode", "enable compact mode");

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

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    spdlog::stopwatch sw;

    // The variational ODEs.
    auto vsys = var_ode_sys(orig_sys, {var_args::vars}, order);

    logger->trace("Variational equations construction runtime: {}", sw);

    const auto ic_x = .2, ic_v = .3, ic_tm = .5, ic_par = .4;

    sw.reset();

    auto ta = taylor_adaptive<double>{
        vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm, kw::compact_mode = compact_mode};

    logger->trace("Taylor adaptive construction runtime: {}", sw);
}
