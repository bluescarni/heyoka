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
                                      // v' = -9.8 * sin(x)
                                      {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0., 0.}};

    // Create 10 sets of different initial conditions,
    // one for each element of the ensemble.
    std::vector<std::vector<double>> ensemble_ics;
    for (auto i = 0; i < 10; ++i) {
        ensemble_ics.push_back({0.05 + i / 100., 0.025 + i / 100.});
    }

    // The generator.
    auto gen = [&ensemble_ics](taylor_adaptive<double> ta_copy, std::size_t i) {
        ta_copy.get_state_data()[0] = ensemble_ics[i][0];
        ta_copy.get_state_data()[1] = ensemble_ics[i][1];

        return ta_copy;
    };

    // Run the ensemble propagation up to t = 20.
    auto ret = ensemble_propagate_until<double>(ta, 20., 10, gen);

    // Print to screen the integrator that was used
    // for the last iteration of the ensemble.
    std::cout << std::get<0>(ret[9]) << '\n';

    // Print the values returned by the propagate_until()
    // invocation.
    std::cout << "Integration outcome: " << std::get<1>(ret[9]) << '\n';
    std::cout << "Min/max timesteps  : " << std::get<2>(ret[9]) << "/" << std::get<3>(ret[9]) << '\n';
    std::cout << "N of timesteps     : " << std::get<4>(ret[9]) << '\n';
    std::cout << "Continuous output  : " << static_cast<bool>(std::get<5>(ret[9])) << '\n';
}
