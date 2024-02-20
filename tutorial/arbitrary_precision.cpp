// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Setup the precision.
    const mpfr_prec_t prec = 237;

    // Create the integrator object
    // in octuple precision.
    auto ta = taylor_adaptive<mppp::real>{// Definition of the ODE system:
                                          // x' = v
                                          // v' = -9.8 * sin(x)
                                          {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          // Initial conditions
                                          // for x and v.
                                          {mppp::real{-1., prec}, mppp::real{0., prec}}};

    // Print the integrator object to screen.
    std::cout << ta << '\n';

    // Create a small helper to compute the energy constant
    // from the state vector.
    auto compute_energy = [](const auto &sv) {
        using std::cos;

        return (sv[1] * sv[1]) / 2 + 9.8 * (1 - cos(sv[0]));
    };

    // Compute and store the intial energy.
    const auto orig_E = compute_energy(ta.get_state());

    // Integrate for a few timesteps.
    for (auto i = 0; i < 20; ++i) {
        using std::abs;

        ta.step();

        std::cout << "Relative energy error: " << abs((orig_E - compute_energy(ta.get_state())) / orig_E) << '\n';
    }
}
