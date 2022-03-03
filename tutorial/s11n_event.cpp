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

// The callback function object.
struct callback {
    // Leave the callback body empty.
    void operator()(taylor_adaptive<double> &, double, int) const {}

private:
    // Make the callback serialisable.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

// Register the callback in the serialisation system.
HEYOKA_S11N_CALLABLE_EXPORT(callback, void, taylor_adaptive<double> &, double, int)

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
                                      {0.05, 0.025},
                                      // Add the non-terminal event v = 0, using
                                      // the callback defined above.
                                      kw::nt_events = {nt_event<double>(v, callback{})}};

    std::cout << "Number of events (original)    : " << ta.get_nt_events().size() << '\n';

    // Serialise ta into a string stream.
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << ta;
    }

    // Reset ta to an integrator without events.
    ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

    std::cout << "Number of events (after reset) : " << ta.get_nt_events().size() << '\n';

    // Restore the serialised representation of ta.
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> ta;
    }

    std::cout << "Number of events (from archive): " << ta.get_nt_events().size() << '\n';
}
