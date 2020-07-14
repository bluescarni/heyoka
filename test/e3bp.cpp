// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

#include <iostream>

using namespace heyoka;

template <typename T>
std::vector<T> cart_to_ham_ec(const std::vector<T> &s)
{
    const auto x = s[0], y = s[1], z = s[2], vx = s[3], vy = s[4], vz = s[5];

    // First to cylindrical.
    const auto rho = std::sqrt(x * x + y * y);
    const auto phi = std::atan2(y, x);
    const auto vrho = (x * vx + y * vy) / rho;
    const auto vphi = (vy * x - vx * y) / (x * x + y * y);

    // Now to ec.
    const auto xi = (std::sqrt(rho * rho + (z + 1) * (z + 1)) + std::sqrt(rho * rho + (z - 1) * (z - 1))) / 2;
    const auto eta = (std::sqrt(rho * rho + (z + 1) * (z + 1)) - std::sqrt(rho * rho + (z - 1) * (z - 1))) / 2;
    const auto vxi = 0.5
                     * ((vrho * rho + vz * (z + 1)) / std::sqrt(rho * rho + (z + 1) * (z + 1))
                        + (vrho * rho + vz * (z - 1)) / std::sqrt(rho * rho + (z - 1) * (z - 1)));
    const auto veta = 0.5
                      * ((vrho * rho + vz * (z + 1)) / std::sqrt(rho * rho + (z + 1) * (z + 1))
                         - (vrho * rho + vz * (z - 1)) / std::sqrt(rho * rho + (z - 1) * (z - 1)));

    // Now to Hamiltonian momenta.
    const auto pxi = vxi * (xi * xi - eta * eta) / (xi * xi - 1);
    const auto peta = veta * (xi * xi - eta * eta) / (1 - eta * eta);
    const auto pphi = vphi * (xi * xi - 1) * (1 - eta * eta);

    return std::vector{xi, eta, phi, pxi, peta, pphi};
}

TEST_CASE("e3bp periodic")
{
    // x0, x1, x2 = xi, eta, phi
    // y0, y1, y2 = p_xi, p_eta, p_phi
    auto [x0, x1, x2, y0, y1, y2] = make_vars("x0", "x1", "x2", "y0", "y1", "y2");

    auto a = 1_dbl, mu1 = 1_dbl, mu2 = 0.05_dbl;

    auto ham = (y0 * y0 * (x0 * x0 - 1_dbl) + y1 * y1 * (1_dbl - x1 * x1)) / (2_dbl * a * a * (x0 * x0 - y0 * y0))
               + (y2 * y2) / (2_dbl * a * a * (x0 * x0 - 1_dbl) * (1_dbl - x1 * x1)) - mu1 / (a * (x0 - x1))
               - mu2 / (a * (x0 + x1));

    llvm_state ham_llvm{"ham tracing"};
    ham_llvm.add_dbl("ham", ham);
    ham_llvm.compile();
    auto h_trace = ham_llvm.fetch_dbl<5>("ham");

    std::vector<double> init_state
        = cart_to_ham_ec(std::vector{1.20793759666736, -0.493320558636725, 1.19760678594565, -0.498435147674914,
                                     0.548228167205306, 0.49662691628363});

    taylor_decompose({"x0"_p = diff(ham, "y0"), "x1"_p = diff(ham, "y1"), "x2"_p = diff(ham, "y2"),
                      "y0"_p = -diff(ham, "x0"), "y1"_p = -diff(ham, "x1"), "y2"_p = -diff(ham, "x2")});

    taylor_adaptive_dbl tad{
        {diff(ham, "y0"), diff(ham, "y1"), diff(ham, "y2"), -diff(ham, "x0"), -diff(ham, "x1"), x2 - x2},
        std::vector{init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5]},
        0,
        1E-16,
        1E-16};

    const auto &st = tad.get_state();

    std::cout.precision(14);
    std::cout << h_trace(init_state[0], init_state[1], init_state[3], init_state[4], init_state[5]) << '\n';
    std::cout << h_trace(st[0], st[1], st[3], st[4], st[5]) << '\n';

    tad.step();

    std::cout << h_trace(st[0], st[1], st[3], st[4], st[5]) << '\n';
}
