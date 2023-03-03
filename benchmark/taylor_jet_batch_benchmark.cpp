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
#include <stdexcept>
#include <string>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    auto batch_size = 1u;

    if (argc > 1) {
        auto bs = std::stoi(argv[1]);
        if (bs <= 0) {
            throw std::invalid_argument("The batch size must be positive, but it is " + std::string(argv[1])
                                        + " instead");
        }
        batch_size = static_cast<unsigned>(bs);
    }

    if (batch_size == 0u) {
        std::cout << "The vector size on the current machine is zero, exiting.\n";

        return 0;
    }

    auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
        = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

    auto x01 = x1 - x0;
    auto y01 = y1 - y0;
    auto z01 = z1 - z0;
    auto r01_m3 = pow(x01 * x01 + y01 * y01 + z01 * z01, -3_dbl / 2_dbl);

    const auto order = 20u;

    llvm_state s;

    auto dc = taylor_add_jet<double>(s, "jet",
                                     {x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3,
                                      -z01 * r01_m3, vx0, vx1, vy0, vy1, vz0, vz1},
                                     order, batch_size, false, false);

    // std::cout << s.get_ir() << '\n';
    // s.dump_object_code("tjb.o");

    s.compile();

    auto jet_ptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

    std::vector<double> jet(12u * (order + 1u) * batch_size);
    for (auto &v : jet) {
        v = 1;
    }
    auto ptr = jet.data();

    // Warm up.
    jet_ptr(ptr, nullptr, nullptr);

    auto start = std::chrono::high_resolution_clock::now();

    // Do 400 evaluations.
    for (auto i = 0; i < 40; ++i) {
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
        jet_ptr(ptr, nullptr, nullptr);
    }

    const auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Order " << order << ", batch size " << batch_size << ": " << elapsed / 400 << "ns\n";

    return 0;
}
