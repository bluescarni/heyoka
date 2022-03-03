// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = -g/l * sin(x)
                                      {prime(x) = v, prime(v) = -par[0] / par[1] * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0.05, 0.},
                                      // Values of the runtime parameters
                                      // g and l. If not provided,
                                      // all parameters will be inited
                                      // to zero.
                                      kw::pars = {9.8, 1.}};

    // Let's print to screen the integrator object.
    std::cout << ta << '\n';

    // Integrate for ~1 period.
    ta.propagate_until(2.0074035758801299);
    std::cout << ta << '\n';

    // Reset the time coordinate.
    ta.set_time(0.);

    // Change the value of g.
    ta.get_pars_data()[0] = 3.72;

    // Integrate for ~1 period on Mars.
    ta.propagate_until(3.2581889116828258);
    std::cout << ta << '\n';
}
