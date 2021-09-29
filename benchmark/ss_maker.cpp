// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "benchmark_utils.hpp"

using namespace heyoka;
using namespace heyoka_benchmark;

template <typename T>
void run_bench(std::uint32_t nplanets, T tol, bool high_accuracy, bool compact_mode, bool fast_math, double final_time)
{
    warmup();

    // Init the masses vector with the solar mass.
    std::vector masses{T(1)};

    // Add the planets' masses.
    for (std::uint32_t i = 0; i < nplanets; ++i) {
        masses.push_back((T(1) / 333000) / ((i + 1u) * (i + 1u)));
    }

    // G constant, in terms of solar masses, AUs and years.
    const auto G = T(0.01720209895) * T(0.01720209895) * 365 * 365;

    // Create the nbody system.
    auto sys = make_nbody_sys(nplanets + 1u, kw::masses = masses, kw::Gconst = G);

    // The initial state (zeroed out, change it later).
    std::vector<T> init_state((nplanets + 1u) * 6u);

    auto start = std::chrono::high_resolution_clock::now();

    // Create the integrator.
    taylor_adaptive<T> ta{
        std::move(sys), std::move(init_state),    kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
        kw::tol = tol,  kw::fast_math = fast_math};

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "ms\n";
    std::cout << "Decomposition size: " << ta.get_decomposition().size() << '\n';

    // Create xtensor views for ease of indexing.
    auto s_array = xt::adapt(ta.get_state_data(), {nplanets + 1u, 6u});
    auto m_array = xt::adapt(masses.data(), {nplanets + 1u});

    // Set the initial positions at regular intervals on the x axis
    // on circular orbits. The Sun is already in the origin with zero
    // velocity.
    for (std::uint32_t i = 0; i < nplanets; ++i) {
        using std::sqrt;

        s_array(i + 1u, 0) = i + 1u;
        s_array(i + 1u, 4) = sqrt(G / (i + 1u));
    }

    // Move the COM.
    const auto com_x = xt::sum(xt::view(s_array, xt::all(), 0) * m_array) / xt::sum(m_array);
    const auto com_vx = xt::sum(xt::view(s_array, xt::all(), 4) * m_array) / xt::sum(m_array);
    std::cout << "Original com_x: " << com_x << '\n';
    std::cout << "Original com_vx: " << com_vx << '\n';
    xt::view(s_array, xt::all(), 0) -= com_x;
    xt::view(s_array, xt::all(), 4) -= com_vx;
    std::cout << "New com_x: " << xt::sum(xt::view(s_array, xt::all(), 0) * m_array) / xt::sum(m_array) << '\n';
    std::cout << "New com_vx: " << xt::sum(xt::view(s_array, xt::all(), 4) * m_array) / xt::sum(m_array) << '\n';

    std::ofstream ofs("out.txt", std::ios_base::out | std::ios_base::trunc);
    ofs.precision(std::numeric_limits<T>::max_digits10);

    start = std::chrono::high_resolution_clock::now();

    // Run the integration.
    T next_save_point(0);
    while (true) {
        if (auto [oc, _] = ta.step(T(final_time) - ta.get_time()); oc == taylor_outcome::time_limit) {
            break;
        }
        if (ta.get_time() > next_save_point) {
            ofs << ta.get_time() << " ";

            for (auto val : s_array) {
                ofs << val << " ";
            }

#if defined(HEYOKA_HAVE_REAL128)
            // NOTE: don't try to save the orbital elements
            // if real128 is being used, as the xt linalg functions
            // don't work on real128.
            if constexpr (!std::is_same_v<T, mppp::real128>) {
#endif
                // Store the orbital elements wrt the Sun.
                for (std::uint32_t i = 1; i < nplanets + 1u; ++i) {
                    auto rel_x = xt::view(s_array, i, xt::range(0, 3)) - xt::view(s_array, 0, xt::range(0, 3));
                    auto rel_v = xt::view(s_array, i, xt::range(3, 6)) - xt::view(s_array, 0, xt::range(3, 6));

                    auto kep = cart_to_kep(rel_x, rel_v, G * masses[0]);

                    for (auto oe : kep) {
                        ofs << oe << " ";
                    }
                }
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif

            ofs << std::endl;

            // Save every 100 years.
            next_save_point += 100;
        }
    }

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Integration time: " << elapsed << "ms\n";

    std::cout << s_array << '\n';
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    std::string fp_type;
    double tol;
    bool compact_mode = false;
    bool high_accuracy = false;
    bool fast_math = false;
    std::uint32_t nplanets;
    double final_time;

    po::options_description desc("Options");

    desc.add_options()("help", "produce help message")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("double"), "floating-point type")(
        "tol", po::value<double>(&tol)->default_value(0.), "tolerance (if 0, it will be the type's epsilon)")(
        "high_accuracy", "enable high-accuracy mode")("compact_mode", "enable compact mode")(
        "fast_math", "enable fast math flags")("nplanets", po::value<std::uint32_t>(&nplanets)->default_value(1),
                                               "number of planets (>=1)")(
        "final_time", po::value<double>(&final_time)->default_value(1e5), "total integration time (years)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (nplanets == 0u) {
        throw std::invalid_argument("The number of planets cannot be zero");
    }

    if (!std::isfinite(final_time) || final_time <= 0) {
        throw std::invalid_argument("Invalid final time specified: the final time must be finite and positive");
    }

    if (vm.count("high_accuracy")) {
        high_accuracy = true;
    }

    if (vm.count("compact_mode")) {
        compact_mode = true;
    }

    if (vm.count("fast_math")) {
        fast_math = true;
    }

    if (fp_type == "double") {
        run_bench<double>(nplanets, tol, high_accuracy, compact_mode, fast_math, final_time);
    } else if (fp_type == "long double") {
        run_bench<long double>(nplanets, tol, high_accuracy, compact_mode, fast_math, final_time);
#if defined(HEYOKA_HAVE_REAL128)
    } else if (fp_type == "real128") {
        run_bench<mppp::real128>(nplanets, mppp::real128(tol), high_accuracy, compact_mode, fast_math, final_time);
#endif
    } else {
        throw std::invalid_argument("Invalid floating-point type: '" + fp_type + "'");
    }
}
