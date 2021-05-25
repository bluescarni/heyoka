// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

// NOTE: this benchmark is meant to be run under valgrind
// (or other similar tool) with several values of final_time in order to check
// that the number of memory allocations tends to a fixed value
// as final_time increases. This is meant to check the caching
// techniques in the event detection machinery.
int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    double final_time;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "final_time", po::value<double>(&final_time)->default_value(100.), "simulation end time");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (!std::isfinite(final_time) || final_time <= 0) {
        throw std::invalid_argument("The final time must be finite and positive, but it is "
                                    + std::to_string(final_time) + " instead");
    }

    auto [x, v] = make_vars("x", "v");

    using ev_t = taylor_adaptive<double>::nt_event_t;

    auto ta_ev = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                         {-0.25, 0.},
                                         kw::nt_events = {ev_t(v, [](taylor_adaptive<double> &, double, int) {})}};

    for (auto i = 0; i < 10; ++i) {
        ta_ev.propagate_for(final_time);
    }

    std::cout << "Final time: " << ta_ev.get_time() << '\n';
}
