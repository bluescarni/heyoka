// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <initializer_list>
#include <iostream>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

int main(int argc, char *argv[])
{
    warmup();

    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>({prime(x) = v, prime(v) = -sin(x)}, {1.3, 0}, kw::tol = 1e-15);

    auto start = std::chrono::high_resolution_clock::now();
    ta.propagate_until(10000.);
    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
    std::cout << "Runtime: " << elapsed << "μs\n";
}
