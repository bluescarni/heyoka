// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/taylor.hpp>

#include <heyoka/logging.hpp>

using namespace heyoka;

std::vector<std::pair<heyoka::expression, heyoka::expression>> construct_sgp4_ode()
{
    // Fetch sgp4's formulae.
    auto sgp4_func = heyoka::model::sgp4();

    // The variable representing tsince in the sgp4 formulae.
    const auto tsince = heyoka::expression("tsince");

    // In sgp4_func, replace the TLE data with params, and tsince
    // with tsince + par[7].
    sgp4_func = heyoka::subs(sgp4_func, {{"n0", heyoka::par[0]},
                                         {"e0", heyoka::par[1]},
                                         {"i0", heyoka::par[2]},
                                         {"node0", heyoka::par[3]},
                                         {"omega0", heyoka::par[4]},
                                         {"m0", heyoka::par[5]},
                                         {"bstar", heyoka::par[6]},
                                         {"tsince", tsince + heyoka::par[7]}});

    // Compute the rhs of the sgp4 ODE, substituting tsince with the time placeholder.
    const auto dt = heyoka::diff_tensors(sgp4_func, {tsince});
    auto sgp4_rhs = heyoka::subs(dt.get_jacobian(), {{tsince, heyoka::time}});

    // Create the state variables for the ODE.
    auto [x, y, z, vx, vy, vz, e, r] = heyoka::make_vars("x", "y", "z", "vx", "vy", "vz", "e", "r");

    // Add the differential equation for r.
    // NOTE: do **not** use vx/vy/vz here. Apparently, in the SGP4 algorithm, if one takes the
    // time derivatives of x/y/z one does not get *exactly* the same values as the vx/vy/vz returned
    // by SGP4. In order for the differential equation for r to be correct, we need the the true time
    // derivatives of x/y/z, and we cannot use what SGP4 says are the velocities.
    sgp4_rhs.push_back(heyoka::sum({x * sgp4_rhs[0], y * sgp4_rhs[1], z * sgp4_rhs[2]}) / r);

    // Return the ODE sys.
    using heyoka::prime;
    return {prime(x) = sgp4_rhs[0],  prime(y) = sgp4_rhs[1],  prime(z) = sgp4_rhs[2], prime(vx) = sgp4_rhs[3],
            prime(vy) = sgp4_rhs[4], prime(vz) = sgp4_rhs[5], prime(e) = sgp4_rhs[6], prime(r) = sgp4_rhs[7]};
}

int main(int argc, char *argv[])
{
    set_logger_level_trace();

    namespace po = boost::program_options;

    bool parjit = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")("parjit", "parallel JIT compilation");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("parjit")) {
        parjit = true;
    }

    taylor_adaptive<double> ta{construct_sgp4_ode(), std::vector<double>(8u), kw::high_accuracy = true,
                               kw::compact_mode = true, kw::parjit = parjit};
}
