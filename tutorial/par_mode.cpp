// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/heyoka.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

using namespace heyoka;

// Generic function for running the benchmark with floating-point type T.
template <typename T>
double run_benchmark(T final_time, bool parallel_mode)
{
    // The number of protoplanets.
    const unsigned nplanets = 400;

    // G constant, in terms of solar masses, AUs and years.
    const auto G = T(0.01720209895) * T(0.01720209895) * 365 * 365;

    // Init the masses vector with the solar mass.
    std::vector masses{T(1)};

    // Add the planets' masses.
    for (auto i = 0u; i < nplanets; ++i) {
        masses.push_back((T(1) / 333000) / ((i + 1u) * (i + 1u)));
    }

    // Create the nbody system.
    auto sys = make_nbody_sys(nplanets + 1u, kw::masses = masses, kw::Gconst = G);

    // The initial state (zeroed out, we will change it later).
    std::vector<T> init_state((nplanets + 1u) * 6u);

    // Create the integrator.
    // NOTE: compact_mode is *required* when using parallel mode.
    taylor_adaptive<T> ta{std::move(sys), std::move(init_state), kw::compact_mode = true,
                          kw::parallel_mode = parallel_mode};

    // Create xtensor views for ease of indexing.
    auto s_array = xt::adapt(ta.get_state_data(), {nplanets + 1u, 6u});
    auto m_array = xt::adapt(masses.data(), {nplanets + 1u});

    // Set the initial positions at regular intervals on the x axis
    // on circular orbits. The Sun is already in the origin with zero
    // velocity.
    for (auto i = 0u; i < nplanets; ++i) {
        using std::sqrt;

        s_array(i + 1u, 0) = i + 1u;
        s_array(i + 1u, 4) = sqrt(G / (i + 1u));
    }

    // Take the current time.
    auto start = std::chrono::high_resolution_clock::now();

    // Integrate.
    ta.propagate_for(final_time);

    // Return the elapsed time.
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());
}

int main()
{
    // Limit to 8 threads of execution.
    oneapi::tbb::global_control gc(oneapi::tbb::global_control::parameter::max_allowed_parallelism, 8);

    // Run the serial vs parallel comparison in double precision.
    auto serial_time_dbl = run_benchmark<double>(10, false);
    std::cout << "Serial time (double): " << serial_time_dbl << "ms\n";

    auto parallel_time_dbl = run_benchmark<double>(10, true);
    std::cout << "Parallel time (double): " << parallel_time_dbl << "ms\n";

#if defined(HEYOKA_HAVE_REAL128)

    // Run the serial vs parallel comparison in quadruple precision.
    auto serial_time_f128 = run_benchmark<mppp::real128>(1, false);
    std::cout << "Serial time (real128): " << serial_time_f128 << "ms\n";

    auto parallel_time_f128 = run_benchmark<mppp::real128>(1, true);
    std::cout << "Parallel time (real128): " << parallel_time_f128 << "ms\n";

#endif
}
