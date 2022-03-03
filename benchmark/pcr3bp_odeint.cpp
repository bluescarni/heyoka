// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include "benchmark_utils.hpp"

using namespace heyoka_benchmark;

using state_type = std::array<double, 4>;

void pcr3bp(const state_type &q, state_type &dq, double)
{
    const auto mu = 0.01;
    const auto onemmu = 1 - mu;
    auto x1 = q[0] - mu;
    auto x1sq = x1 * x1;
    auto y = q[1];
    auto ysq = y * y;
    auto r1_1p5 = std::pow(x1sq + ysq, -1.5);
    auto x2 = q[0] + onemmu;
    auto x2sq = x2 * x2;
    auto r2_1p5 = std::pow(x2sq + ysq, -1.5);
    dq[0] = q[2] + q[1];
    dq[1] = q[3] - q[0];
    dq[2] = (-((onemmu * x1) * r1_1p5) - ((mu * x2) * r2_1p5)) + q[3];
    dq[3] = (-((onemmu * y) * r1_1p5) - ((mu * y) * r2_1p5)) - q[2];
}

namespace odeint = boost::numeric::odeint;

int main(int argc, char *argv[])
{
    warmup();

    using error_stepper_type = odeint::runge_kutta_fehlberg78<state_type>;

    state_type ic = {-0.8, 0.0, 0.0, -0.6276410653920694};

    auto start = std::chrono::high_resolution_clock::now();
    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(1e-15, 1e-15), &pcr3bp, ic, 0.0, 2000.,
                               1e-8);
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
}
