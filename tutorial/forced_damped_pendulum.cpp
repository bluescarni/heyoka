// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;
namespace hy = heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = cos(t) - 0.1*v - sin(x)
                                      {prime(x) = v, prime(v) = cos(hy::time) - .1 * v - sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0., 1.85},
                                      // Explicitly specify the
                                      // initial value for the time
                                      // variable.
                                      kw::time = 0.};

    // Integrate for 50 time units, and print
    // the value of x every 2 time units.
    for (auto i = 0; i < 25; ++i) {
        ta.propagate_for(2.);
        std::cout << "x = " << ta.get_state()[0] << '\n';
    }
}
