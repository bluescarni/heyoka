// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
                                      {0., 1.97},
                                      // Explicitly specify the
                                      // initial value for the time
                                      // variable.
                                      kw::time = 0.};

    // Integrate for 10 time units.
    ta.propagate_for(10.);

    // Print the state vector.
    std::cout << "x(10) = " << ta.get_state()[0] << '\n';
    std::cout << "v(10) = " << ta.get_state()[1] << '\n';
}
