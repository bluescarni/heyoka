// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

using namespace heyoka;

int main()
{
    std::mt19937 rng(0);

    const auto batch_size = 400u;

    std::vector<double> values(batch_size * 3u);
    std::uniform_real_distribution<double> rdist(-10, 10);
    for (auto &v : values) {
        v = rdist(rng);
    }
    std::vector<double> out(batch_size);

    auto out_ptr = out.data();
    auto in_ptr = values.data();

    auto [x, y, z] = make_vars("x", "y", "z");

    llvm_state s{""};

    s.add_function_batch<double>("bench", (x * y) + (x * z) + (y * z) + (x + y) + (x + z) + (y + z), batch_size);

    // NOTE: uncomment to dump IR/object code.
    // std::cout << s.dump_ir() << '\n';
    // s.dump_object_code("flops.o");

    s.compile();

    auto f_ptr = s.fetch_function_batch_dbl("bench");

    auto start = std::chrono::high_resolution_clock::now();

    const auto nruns = 100000u;
    for (auto i = 0u; i < nruns; ++i) {
        f_ptr(out_ptr, in_ptr);
    }

    const auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    // NOTE: the number of 7 FLOP comes for examining
    // the assembly produced on an AVX-enabled machine, where the
    // expression above is simplified to 2 additions, 1 mul
    // and 2 FMAs.
    std::cout << "GFLOPS (assuming AVX): " << ((7. * batch_size * nruns) / (1E-6 * elapsed)) / 1E9 << '\n';

    return 0;
}
