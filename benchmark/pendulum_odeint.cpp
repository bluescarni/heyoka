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

#include "benchmark_utils.hpp"

using namespace heyoka_benchmark;

using state_type = std::array<double, 2>;

void pendulum(const state_type &x, state_type &dxdt, double)
{
    dxdt[0] = x[1];
    dxdt[1] = -std::sin(x[0]);
}

namespace odeint = boost::numeric::odeint;

int main(int argc, char *argv[])
{
    warmup();

    using error_stepper_type = odeint::runge_kutta_fehlberg78<state_type>;

    state_type ic = {1.3, 0};

    auto start = std::chrono::high_resolution_clock::now();
    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(2.2e-16, 2.2e-16), &pendulum, ic, 0.0,
                               10000., 1e-8);
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
}
