// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <fmt/format.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/math/square.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

template <typename T>
void run_integration(const std::string &filename, T t_final, double perturb, bool compact_mode, T tol)
{
    using std::abs;
    using std::log10;
    using std::pow;

    using namespace heyoka;
    using namespace heyoka_benchmark;
    using namespace fmt::literals;

    auto masses
        = std::vector{T(1.00000597682), T(1) / 1047.355, T(1) / 3501.6, T(1) / 22869., T(1) / 19314., T(7.4074074e-09)};

    const auto G = T(0.01720209895) * T(0.01720209895) * 365 * 365;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    auto ic = {// Sun.
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

    auto init_state = std::vector<T>(ic.begin(), ic.end());

    // Perturb the initial state.
    std::mt19937 rng;
    rng.seed(static_cast<std::mt19937::result_type>(std::random_device()()));
    std::uniform_real_distribution<double> rdist(-1., 1.);
    for (auto &x : init_state) {
        x += abs(x) * (rdist(rng) * perturb);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Create the events. We will assign to each body
    // the radius of jupiter to keep thing simple.
    const auto jradius = T(0.000477895);
    std::vector<nt_event<T>> evs;
    auto cb = [](taylor_adaptive<T> &, T) { throw; };
    for (auto i = 0; i < 6; ++i) {
        auto xi = expression("x_{}"_format(i));
        auto yi = expression("y_{}"_format(i));
        auto zi = expression("z_{}"_format(i));

        for (auto j = i + 1; j < 6; ++j) {
            auto xj = expression("x_{}"_format(j));
            auto yj = expression("y_{}"_format(j));
            auto zj = expression("z_{}"_format(j));

            auto diff_x = xj - xi;
            auto diff_y = yj - yi;
            auto diff_z = zj - zi;

            auto ev_eq = square(diff_x) + square(diff_y) + square(diff_z) - 4 * jradius * jradius;

            evs.emplace_back(std::move(ev_eq), cb);
        }
    }

    taylor_adaptive<T> ta{
        std::move(sys), std::move(init_state),         kw::high_accuracy = true, kw::compact_mode = compact_mode,
        kw::tol = tol,  kw::nt_events = std::move(evs)};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";

    // Create xtensor views on the state and mass vectors
    // for ease of indexing.
    auto s_array = xt::adapt(ta.get_state_data(), {6, 6});
    auto m_array = xt::adapt(masses.data(), {6});

    // Cache the total mass.
    const auto tot_mass = xt::sum(m_array)[0];

    // Helpers to compute the position and velocity of the COM.
    auto get_com = [&s_array, &m_array, &tot_mass]() {
        auto com_x = xt::sum(m_array * xt::view(s_array, xt::all(), 0)) / tot_mass;
        auto com_y = xt::sum(m_array * xt::view(s_array, xt::all(), 1)) / tot_mass;
        auto com_z = xt::sum(m_array * xt::view(s_array, xt::all(), 2)) / tot_mass;

        return vNd<T, 3>{com_x[0], com_y[0], com_z[0]};
    };

    auto get_com_v = [&s_array, &m_array, &tot_mass]() {
        auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3)) / tot_mass;
        auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4)) / tot_mass;
        auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5)) / tot_mass;

        return vNd<T, 3>{com_vx[0], com_vy[0], com_vz[0]};
    };

    // Helper to compute the total energy in the system.
    auto get_energy = [&s_array, &m_array, G]() {
        using std::sqrt;

        // Kinetic energy.
        T kin(0), c(0);
        for (auto i = 0u; i < 6u; ++i) {
            auto vx = xt::view(s_array, i, 3)[0];
            auto vy = xt::view(s_array, i, 4)[0];
            auto vz = xt::view(s_array, i, 5)[0];

            auto tmp = T{1} / 2 * m_array[i] * (vx * vx + vy * vy + vz * vz);
            auto y = tmp - c;
            auto t = kin + y;
            c = (t - kin) - y;
            kin = t;
        }

        // Potential energy.
        T pot(0);
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
                           / sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj));
                auto y = tmp - c;
                auto t = pot + y;
                c = (t - pot) - y;
                pot = t;
            }
        }

        return kin + pot;
    };

    // Compute position and velocity of the COM.
    const auto init_com = get_com();
    const auto init_com_v = get_com_v();

    std::cout << "Initial COM         : " << init_com << '\n';
    std::cout << "Initial COM velocity: " << init_com_v << '\n';

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

    std::cout << "New COM         : " << get_com() << '\n';
    std::cout << "New COM velocity: " << get_com_v() << '\n';

    const auto init_energy = get_energy();
    std::cout << "Initial energy: " << init_energy << '\n';

    // Base-10 logs of the initial and final saving times.
    const auto start_time = T(0), final_time = log10(T(t_final));
    // Number of snapshots to take.
    const auto n_snaps = 50000u;
    // Build the vector of log10 saving times.
    std::vector<T> save_times;
    save_times.push_back(start_time);
    for (auto i = 1u; i < n_snaps; ++i) {
        save_times.push_back(save_times.back() + (final_time - start_time) / (n_snaps - 1u));
    }
    for (auto &t : save_times) {
        t = pow(T(10), t);
    }

    std::optional<std::ofstream> of;
    if (!filename.empty()) {
        of.emplace(filename, std::ios_base::out | std::ios_base::trunc);
        of->precision(std::numeric_limits<T>::max_digits10);
    }
    auto it = save_times.begin();

    start = std::chrono::high_resolution_clock::now();

    while (ta.get_time() < pow(T(10), final_time)) {
        if (of && it != save_times.end() && ta.get_time() >= *it) {
            // We are at or past the current saving time, record
            // the time, energy error and orbital elements.
            *of << ta.get_time() << " " << abs((init_energy - get_energy()) / init_energy) << " ";

            // Store the state.
            for (auto val : s_array) {
                *of << val << " ";
            }

#if defined(HEYOKA_HAVE_REAL128)
            // NOTE: don't try to save the orbital elements
            // if real128 is being used, as the xt linalg functions
            // don't work on real128.
            if constexpr (!std::is_same_v<T, mppp::real128>) {
#endif
                // Store the orbital elements wrt the Sun.
                for (auto i = 1u; i < 6u; ++i) {
                    auto rel_x = xt::view(s_array, i, xt::range(0, 3)) - xt::view(s_array, 0, xt::range(0, 3));
                    auto rel_v = xt::view(s_array, i, xt::range(3, 6)) - xt::view(s_array, 0, xt::range(3, 6));

                    auto kep = cart_to_kep(rel_x, rel_v, G * masses[0]);

                    for (auto oe : kep) {
                        *of << oe << " ";
                    }
                }
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif

            *of << std::endl;

            // Locate the next saving time (that is, the first saving
            // time which is greater than the current time).
            it = std::upper_bound(it, save_times.end(), ta.get_time());
        }
        auto [res, h] = ta.step();
        if (res != taylor_outcome::success) {
            throw std::runtime_error("Error status detected: " + std::to_string(static_cast<int>(res)));
        }
    }

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integration time: " << elapsed << "ms\n";
    std::cout << "Final energy error: " << abs((init_energy - get_energy()) / init_energy) << '\n';
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type, filename;
    double final_time, perturb, tol;
    bool compact_mode = false;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "filename", po::value<std::string>(&filename)->default_value(""),
        "name of the file into which the simulation data will be saved (if empty, no data will be saved)")(
        "final_time", po::value<double>(&final_time)->default_value(1E6), "simulation end time (in years)")(
        "perturb", po::value<double>(&perturb)->default_value(0.),
        "magnitude of the perturbation on the initial state")("compact_mode", "compact mode")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be automatically deduced)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("compact_mode")) {
        compact_mode = true;
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

    if (fp_type == "double") {
        run_integration<double>(filename, final_time, perturb, compact_mode, tol);
    } else if (fp_type == "long double") {
        run_integration<long double>(filename, final_time, perturb, compact_mode, tol);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_integration<mppp::real128>(filename, mppp::real128{final_time}, perturb, compact_mode, mppp::real128{tol});
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }

    return 0;
}
