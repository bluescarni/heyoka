// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int, char *[])
{
    // Fetch the logger.
    create_logger();
    set_logger_level_trace();
    auto logger = spdlog::get("heyoka");

    // Maximum degree/order for the geopotential.
    const auto n = 10u, m = 10u;

    // State variables.
    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Acceleration vector.
    const auto [acc_x, acc_y, acc_z] = model::egm2008_acc({x, y, z}, n, m);

    // Initial conditions in LEO.
    const auto ic_leo = {6740440.0, 0.0, 0.0, 0.0, 6725.973853066024, 3883.2537950295855};

    // Init the integrator.
    auto ta = taylor_adaptive({{x, vx}, {y, vy}, {z, vz}, {vx, acc_x}, {vy, acc_y}, {vz, acc_z}}, ic_leo,
                              kw::compact_mode = true, kw::tol = 1e-15);

    logger->trace("Decomposition size: {}", ta.get_decomposition().size());

    spdlog::stopwatch sw;

    // Propagate.
    ta.propagate_until(86400.);

    logger->trace("Total runtime: {}s", sw);
}
