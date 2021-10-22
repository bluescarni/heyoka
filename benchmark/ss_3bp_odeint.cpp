// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iostream>

#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "benchmark_utils.hpp"

using namespace heyoka_benchmark;

using state_type = std::array<double, 18>;

void ss_3bp(const state_type &q, state_type &dq, double)
{
    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto [x_sun, y_sun, z_sun, vx_sun, vy_sun, vz_sun, x_jup, y_jup, z_jup, vx_jup, vy_jup, vz_jup, x_sat, y_sat, z_sat,
          vx_sat, vy_sat, vz_sat]
        = q;
    const auto [m_sun, m_jup, m_sat] = std::array{1.00000597682, 1 / 1047.355, 1 / 3501.6};

    dq[0] = vx_sun;
    dq[1] = vy_sun;
    dq[2] = vz_sun;

    dq[6] = vx_jup;
    dq[7] = vy_jup;
    dq[8] = vz_jup;

    dq[12] = vx_sat;
    dq[13] = vy_sat;
    dq[14] = vz_sat;

    const auto diff_x_sun_jup = x_sun - x_jup;
    const auto diff_y_sun_jup = y_sun - y_jup;
    const auto diff_z_sun_jup = z_sun - z_jup;
    auto r_sun_jup3 = std::sqrt(diff_x_sun_jup * diff_x_sun_jup + diff_y_sun_jup * diff_y_sun_jup
                                + diff_z_sun_jup * diff_z_sun_jup);
    r_sun_jup3 = r_sun_jup3 * r_sun_jup3 * r_sun_jup3;

    const auto diff_x_sun_sat = x_sun - x_sat;
    const auto diff_y_sun_sat = y_sun - y_sat;
    const auto diff_z_sun_sat = z_sun - z_sat;
    auto r_sun_sat3 = std::sqrt(diff_x_sun_sat * diff_x_sun_sat + diff_y_sun_sat * diff_y_sun_sat
                                + diff_z_sun_sat * diff_z_sun_sat);
    r_sun_sat3 = r_sun_sat3 * r_sun_sat3 * r_sun_sat3;

    const auto diff_x_jup_sat = x_jup - x_sat;
    const auto diff_y_jup_sat = y_jup - y_sat;
    const auto diff_z_jup_sat = z_jup - z_sat;
    auto r_jup_sat3 = std::sqrt(diff_x_jup_sat * diff_x_jup_sat + diff_y_jup_sat * diff_y_jup_sat
                                + diff_z_jup_sat * diff_z_jup_sat);
    r_jup_sat3 = r_jup_sat3 * r_jup_sat3 * r_jup_sat3;

    // Acceleration on the Sun.
    dq[3] = -G * m_jup * diff_x_sun_jup / r_sun_jup3 - G * m_sat * diff_x_sun_sat / r_sun_sat3;
    dq[4] = -G * m_jup * diff_y_sun_jup / r_sun_jup3 - G * m_sat * diff_y_sun_sat / r_sun_sat3;
    dq[5] = -G * m_jup * diff_z_sun_jup / r_sun_jup3 - G * m_sat * diff_z_sun_sat / r_sun_sat3;

    // Acceleration on Jupiter.
    dq[9] = G * m_sun * diff_x_sun_jup / r_sun_jup3 - G * m_sat * diff_x_jup_sat / r_jup_sat3;
    dq[10] = G * m_sun * diff_y_sun_jup / r_sun_jup3 - G * m_sat * diff_y_jup_sat / r_jup_sat3;
    dq[11] = G * m_sun * diff_z_sun_jup / r_sun_jup3 - G * m_sat * diff_z_jup_sat / r_jup_sat3;

    // Acceleration on Saturn.
    dq[15] = G * m_sun * diff_x_sun_sat / r_sun_sat3 + G * m_jup * diff_x_jup_sat / r_jup_sat3;
    dq[16] = G * m_sun * diff_y_sun_sat / r_sun_sat3 + G * m_jup * diff_y_jup_sat / r_jup_sat3;
    dq[17] = G * m_sun * diff_z_sun_sat / r_sun_sat3 + G * m_jup * diff_z_jup_sat / r_jup_sat3;
}

namespace odeint = boost::numeric::odeint;

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

    using error_stepper_type = odeint::runge_kutta_fehlberg78<state_type>;

    state_type ic = {// Sun.
                     -5.137271893918405e-03, -5.288891104344273e-03, 6.180743702483316e-06, 2.3859757364179156e-03,
                     -2.3396779489468049e-03, -8.1384891821122709e-07,
                     // Jupiter.
                     3.404393156051084, 3.6305811472186558, 0.0342464685434024, -2.0433186406983279e+00,
                     2.0141003472039567e+00, -9.7316724504621210e-04,
                     // Saturn.
                     6.606942557811084, 6.381645992310656, -0.1361381213577972, -1.5233982268351876, 1.4589658329821569,
                     0.0061033600397708};

    auto start = std::chrono::high_resolution_clock::now();
    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(tol, tol), &ss_3bp, ic, 0.0, 100000., 1e-8);
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";

    std::cout << fmt::format("Final state: {}", ic) << '\n';
}
