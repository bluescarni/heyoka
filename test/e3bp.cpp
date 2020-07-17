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

TEST_CASE("e3bp")
{
    auto [xi, eta, phi, pxi, peta, pphi] = make_vars("xi", "eta", "phi", "pxi", "peta", "pphi");

    auto a = 1_dbl, mu1 = 1_dbl, mu2 = 0.05_dbl;

    auto ham
        = (pxi * pxi * (xi * xi - 1_dbl) + peta * peta * (1_dbl - eta * eta)) / (2_dbl * a * a * (xi * xi - eta * eta))
          + (pphi * pphi) / (2_dbl * a * a * (xi * xi - 1_dbl) * (1_dbl - eta * eta)) - mu1 / (a * (xi - eta))
          - mu2 / (a * (xi + eta));

    llvm_state ham_llvm{"ham tracing"};
    ham_llvm.add_expression_dbl("ham", ham);
    ham_llvm.compile();
    auto h_trace = ham_llvm.fetch_dbl<5>("ham");

    // NOTE: initial conditions for the periodic orbit from the paper.
    std::vector<double> init_state
        = cart_to_ham_ec(std::vector{1.20793759666736, -0.493320558636725, 1.19760678594565, -0.498435147674914,
                                     0.548228167205306, 0.49662691628363});

    taylor_adaptive_dbl tad{{prime(xi) = diff(ham, pxi), prime(eta) = diff(ham, peta), prime(phi) = diff(ham, pphi),
                             prime(pxi) = -diff(ham, xi), prime(peta) = -diff(ham, eta), prime(pphi) = -diff(ham, phi)},
                            init_state,
                            0,
                            1E-16,
                            1E-16};

    const auto &st = tad.get_state();

    // NOTE: need to pass the state variables in alphabetical order:
    // eta, peta, pphi, pxi, xi.
    const auto orig_en = h_trace(st[1], st[4], st[5], st[3], st[0]);

    for (auto i = 0; i < 200; ++i) {
        tad.step();

        REQUIRE(std::abs((orig_en - h_trace(st[1], st[4], st[5], st[3], st[0])) / orig_en) < 1E-12);
    }

    for (auto i = 0; i < 200; ++i) {
        tad.step_backward();

        REQUIRE(std::abs((orig_en - h_trace(st[1], st[4], st[5], st[3], st[0])) / orig_en) < 1E-12);
    }

    tad.propagate_until(0);

    for (auto i = 0u; i < 6u; ++i) {
        REQUIRE(std::abs((init_state[i] - st[i]) / init_state[i]) < 1E-11);
    }
}
