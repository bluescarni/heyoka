// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}, kw::tol = 1e-15};

    // Integrate for 100 time units.
    ta.propagate_for(1000.);

    // Integrate back to zero.
    ta.propagate_until(0.);

    const auto dx = ta.get_state()[0] - 0.05;
    const auto dv = ta.get_state()[1] - 0.025;

    std::cout << "Error: " << std::sqrt(dx * dx + dv * dv) << '\n';
}
