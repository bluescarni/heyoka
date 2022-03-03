// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <sstream>

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
                                      {0.05, 0.025}};

    // Integrate for a few timesteps.
    for (auto i = 0; i < 5; ++i) {
        ta.step();
    }

    // Print out the time and state at the
    // end of the integration.
    std::cout << "ta time (original)     : " << ta.get_time() << '\n';
    std::cout << "ta state (original)    : [" << ta.get_state()[0] << ", " << ta.get_state()[1] << "]\n\n";

    // Serialise ta into a string stream.
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << ta;
    }

    // Reset ta to the initial state.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    std::cout << "ta time (after reset)  : " << ta.get_time() << '\n';
    std::cout << "ta state (after reset) : [" << ta.get_state()[0] << ", " << ta.get_state()[1] << "]\n\n";

    // Restore the serialised representation of ta.
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> ta;
    }

    // Print out the time and state after
    // deserialisation.
    std::cout << "ta time (from archive) : " << ta.get_time() << '\n';
    std::cout << "ta state (from archive): [" << ta.get_state()[0] << ", " << ta.get_state()[1] << "]\n\n";
}
