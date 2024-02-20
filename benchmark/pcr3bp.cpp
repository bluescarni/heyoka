// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <initializer_list>
#include <iostream>
#include <tuple>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    warmup();

    const auto mu = par[0];

    auto [x, y, px, py] = make_vars("x", "y", "px", "py");

    auto ta = taylor_adaptive<double>(
        {prime(x) = px + y, prime(y) = py - x,
         prime(px) = py - (1. - mu) * (x - mu) * pow((x - mu) * (x - mu) + y * y, -3 / 2.)
                     - mu * (x + (1. - mu)) * pow((x + (1. - mu)) * (x + (1. - mu)) + y * y, -3 / 2.),
         prime(py) = -px - (1. - mu) * y * pow((x - mu) * (x - mu) + y * y, -3 / 2.)
                     - mu * y * pow((x + (1. - mu)) * (x + (1. - mu)) + y * y, -3 / 2.)},
        {-0.8, 0.0, 0.0, -0.6276410653920694}, kw::pars = {0.01}, kw::tol = 1e-15);

    auto start = std::chrono::high_resolution_clock::now();
    auto oc = std::get<0>(ta.propagate_until(2000.));
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
    std::cout << "Outcome: " << oc << '\n';

    std::cout << ta << '\n';
}
