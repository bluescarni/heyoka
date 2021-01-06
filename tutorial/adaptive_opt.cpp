// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iostream>
#include <vector>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Create the integrator object
    // in double precision, specifying
    // a non-default tolerance.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = -9.8 * sin(x)
                                      {prime(x) == v, prime(v) == -9.8 * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0.05, 0.025},
                                      // Set the tolerance to 1e-9.
                                      kw::tol = 1e-9};

    // Print the integrator object to screen.
    std::cout << ta << '\n';

    // Integrate forth to t = 10 and then back to t = 0.
    ta.propagate_until(10.);
    ta.propagate_until(0.);

    std::cout << ta << '\n';

    namespace chrono = std::chrono;

    // Create an nbody system with 6 particles.
    auto sys = make_nbody_sys(6);

    // Create an initial state vector (6 values per body).
    auto sv = std::vector<double>(36);

    // Take the current time.
    auto start = chrono::steady_clock::now();

    // Construct an integrator in default mode.
    auto ta_default = taylor_adaptive<double>{sys, sv};

    std::cout << "Default mode timing: "
              << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "ms\n";

    // Reset the start time.
    start = chrono::steady_clock::now();

    // Construct an integrator in compact mode.
    auto ta_compact = taylor_adaptive<double>{sys, sv, kw::compact_mode = true};

    std::cout << "Compact mode timing: "
              << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "ms\n";
}
