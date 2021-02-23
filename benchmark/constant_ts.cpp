// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include <boost/program_options.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/llvm_state.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string filename;
    double final_time, perturb, ts_size;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "filename", po::value<std::string>(&filename)->default_value(""),
        "name of the file into which the simulation data will be saved (if empty, no data will be saved)")(
        "final_time", po::value<double>(&final_time)->default_value(1E6),
        "simulation end time (in years)")("perturb", po::value<double>(&perturb)->default_value(1e-10),
                                          "magnitude of the perturbation on the initial state")(
        "ts_size", po::value<double>(&ts_size)->default_value(0.6996150333), "timestep size (in years)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // Validate the command-line arguments.
    if (!std::isfinite(final_time) || final_time <= 0) {
        throw std::invalid_argument("The final time must be finite and positive, but it is "
                                    + std::to_string(final_time) + " instead");
    }

    if (!std::isfinite(perturb) || perturb < 0) {
        throw std::invalid_argument("The perturbation parameter must be finite and non-negative, but it is "
                                    + std::to_string(perturb) + " instead");
    }

    if (!std::isfinite(ts_size) || ts_size <= 0) {
        throw std::invalid_argument("The timestep size must be finite and positive, but it is "
                                    + std::to_string(ts_size) + " instead");
    }

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    auto ic = std::vector{// Sun.
                          -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6, +6.69048890636161e-6 * 365,
                          -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                          // Jupiter.
                          +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2, -5.59797969310664e-3 * 365,
                          +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                          // Saturn.
                          +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1, -4.17354020307064e-3 * 365,
                          +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                          // Uranus.
                          +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1, -3.25884806151064e-3 * 365,
                          +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                          // Neptune.
                          -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1, -2.17471785045538e-4 * 365,
                          -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                          // Pluto.
                          -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0, -1.76936577252484e-3 * 365,
                          -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};

    // Perturb the initial state.
    std::mt19937 rng;
    rng.seed(static_cast<std::mt19937::result_type>(std::random_device()()));
    std::uniform_real_distribution<double> rdist(-1., 1.);
    for (auto &x : ic) {
        x += std::abs(x) * (rdist(rng) * perturb);
    }

    // Add the custom timestep machinery.
    llvm_state s;
    taylor_add_custom_step<double>(s, "cstep", sys, 22u, 1, true, false);
    s.compile();

    // Fetch the functions.
    auto jptr = reinterpret_cast<void (*)(double *, const double *)>(s.jit_lookup("cstep_jet"));
    auto uptr = reinterpret_cast<void (*)(double *, const double *)>(s.jit_lookup("cstep_updater"));

    // Prepare the buffer for the jet of derivatives.
    std::vector<double> jet_buffer(36u * (22u + 1u));

    // Copy over the initial conditions.
    std::copy(ic.begin(), ic.end(), jet_buffer.begin());

    // Create xtensor views on the state and mass vectors
    // for ease of indexing.
    auto s_array = xt::adapt(jet_buffer.data(), {6, 6});
    auto m_array = xt::adapt(masses.data(), {6});

    // Cache the total mass.
    const auto tot_mass = xt::sum(m_array)[0];

    // Get the com.
    auto com_x = xt::sum(m_array * xt::view(s_array, xt::all(), 0))[0] / tot_mass;
    auto com_y = xt::sum(m_array * xt::view(s_array, xt::all(), 1))[0] / tot_mass;
    auto com_z = xt::sum(m_array * xt::view(s_array, xt::all(), 2))[0] / tot_mass;
    const xt::xtensor_fixed<double, xt::xshape<3>> init_com = {com_x, com_y, com_z};

    // Get the com v.
    auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3))[0] / tot_mass;
    auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4))[0] / tot_mass;
    auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5))[0] / tot_mass;
    const xt::xtensor_fixed<double, xt::xshape<3>> init_com_v = {com_vx, com_vy, com_vz};

    // Offset the existing positions/velocities so that they refer
    // to the COM.
    xt::view(s_array, 0, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 1, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 2, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 3, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 4, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 5, xt::range(0, 3)) -= init_com;

    xt::view(s_array, 0, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 1, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 2, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 3, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 4, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 5, xt::range(3, 6)) -= init_com_v;

    // Helper to compute the total energy in the system.
    auto get_energy = [&s_array, &m_array, G]() {
        // Kinetic energy.
        double kin(0), c(0);
        for (auto i = 0u; i < 6u; ++i) {
            auto vx = xt::view(s_array, i, 3)[0];
            auto vy = xt::view(s_array, i, 4)[0];
            auto vz = xt::view(s_array, i, 5)[0];

            auto tmp = 0.5 * m_array[i] * (vx * vx + vy * vy + vz * vz);
            auto y = tmp - c;
            auto t = kin + y;
            c = (t - kin) - y;
            kin = t;
        }

        // Potential energy.
        double pot(0);
        c = 0;
        for (auto i = 0u; i < 6u; ++i) {
            auto xi = xt::view(s_array, i, 0)[0];
            auto yi = xt::view(s_array, i, 1)[0];
            auto zi = xt::view(s_array, i, 2)[0];

            for (auto j = i + 1u; j < 6u; ++j) {
                auto xj = xt::view(s_array, j, 0)[0];
                auto yj = xt::view(s_array, j, 1)[0];
                auto zj = xt::view(s_array, j, 2)[0];

                auto tmp = -G * m_array[i] * m_array[j]
                           / std::sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj));
                auto y = tmp - c;
                auto t = pot + y;
                c = (t - pot) - y;
                pot = t;
            }
        }

        return kin + pot;
    };

    std::optional<std::ofstream> of;
    if (!filename.empty()) {
        of.emplace(filename, std::ios_base::out | std::ios_base::trunc);
        of->precision(std::numeric_limits<double>::max_digits10);
    }

    double save_time = 0;
    const auto init_energy = get_energy();
    for (double time = 0; time < final_time; time += ts_size, save_time += ts_size) {
        // Save every 100 years.
        if (of && save_time >= 100) {
            *of << time << " " << std::abs((init_energy - get_energy()) / init_energy) << " ";

            // Store the state.
            for (auto val : s_array) {
                *of << val << " ";
            }

            *of << std::endl;

            save_time = 0;
        }

        // Compute the jet of derivatives.
        jptr(jet_buffer.data(), nullptr);

        // Propagate the new state.
        uptr(jet_buffer.data(), &ts_size);
    }
}
