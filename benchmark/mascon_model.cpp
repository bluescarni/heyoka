// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include <heyoka/math_functions.hpp>
#include <heyoka/taylor.hpp>

// This benchmark builds an equation where the r.h.s. is the sum of N terms all depending on the state
// It can stress test the Taylor integrator for very long expressions \dot r = \sum_i f_i(r) where i goes to 10000
//
// As an example we write the equation for N fixed masses producing a gravitational field affecting on mass

using namespace heyoka;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    bool inline_functions = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("inline_functions", "enable function inlining");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("inline_functions")) {
        inline_functions = true;
    }

    // assembling the r.h.s.
    for (double N = 500u; N < 1000; N += 10) {
        auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");
        expression dx(0_dbl), dy(0_dbl), dz(0_dbl);
        for (double i = -N; i < N; ++i) {
            auto xpos = expression{number(i)};
            auto r2 = (x - xpos) * (x - xpos) + y * y + z * z;
            dx += (x - xpos) * pow(r2, expression{number{-3. / 4.}});
            dy += y * pow(r2, expression{number{-3. / 4.}});
            dz += z * pow(r2, expression{number{-3. / 4.}});
        }
        auto start = high_resolution_clock::now();
        taylor_adaptive<double> taylor{
            {prime(x) = vx, prime(y) = vy, prime(z) = vz, prime(vx) = dx, prime(vy) = dy, prime(vz) = dz},
            {0.123, 0.123, 0.123, 0., 0., 0.},
            kw::compact_mode = true,
            kw::inline_functions = inline_functions};
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << N << ": " << duration.count() / 1e6 << "s" << std::endl;

        // taylor.propagate_for(3000);
    }

    return 0;
}
