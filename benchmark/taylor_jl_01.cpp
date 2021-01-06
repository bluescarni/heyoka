// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iostream>

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

// Example taken from the TaylorIntegration Julia package:
// https://nbviewer.jupyter.org/github/PerezHz/TaylorIntegration.jl/blob/master/examples/x-dot-equals-x-squared.ipynb

int main()
{
    warmup();

    std::cout.precision(16);

    auto x = "x"_var;

    const auto x0 = 3.0;

    taylor_adaptive<double> ta{{prime(x) == x * x}, {x0}, kw::high_accuracy = true, kw::tol = 1e-18};

    const auto final_time = 0.3333333329479479;

    auto start = std::chrono::high_resolution_clock::now();

    auto [oc, min_ss, max_ss, nsteps] = ta.propagate_until(final_time);

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integration time: " << elapsed << "μs\n";

    std::cout << "Status: " << static_cast<int>(oc) << '\n';
    std::cout << "Final time: " << ta.get_time() << '\n';
    std::cout << "Final state: " << ta.get_state()[0] << '\n';
    std::cout << "Total number of steps: " << nsteps << '\n';
    std::cout << "Max step size: " << max_ss << '\n';
    std::cout << "Min step size: " << min_ss << '\n';

    using std::abs;
    const auto exact = x0 / (1. - x0 * ta.get_time());
    std::cout << "Relative error: " << abs((exact - ta.get_state()[0]) / exact) << '\n';
}
