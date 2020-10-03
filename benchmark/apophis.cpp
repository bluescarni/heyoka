// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;
using namespace std::chrono;

template <typename Fp_type>
void to_screen(const taylor_adaptive<Fp_type> &taylor)
{
    std::cout << "t:" << taylor.get_time() << "s"
              << " (" << taylor.get_time() / 365.25 / 60 / 60 / 24 << " years)" << std::endl;
    auto st = taylor.get_state();
    std::cout << "r_s = [" << st[0] << ", " << st[1] << ", " << st[2] << "]" << std::endl;
    std::cout << "v_s = [" << st[3] << ", " << st[4] << ", " << st[5] << "]" << std::endl;
    std::cout << "r_e = [" << st[6] << ", " << st[7] << ", " << st[8] << "]" << std::endl;
    std::cout << "v_e = [" << st[9] << ", " << st[10] << ", " << st[11] << "]" << std::endl;
    std::cout << "r_a = [" << st[12] << ", " << st[13] << ", " << st[14] << "]" << std::endl;
    std::cout << "v_a = [" << st[15] << ", " << st[16] << ", " << st[17] << "]" << std::endl;
}

// In this benchmark we integrate the simplified dynamics of the asteroid Apophis over 100 years, including
// its 2029 Earth close encounter. The dynamics is that of a full three body problem in cartesian coordinates (Cowell's
// method) the initial position of the planets are derived by starting at the Apophis-Earth close encounter conditions
// as computed from the JPL Horizon web service integrating back in quedruple precision for 50 years. From these initial
// conditions, a ground truth trajectory and time grid is established integrating forward with the Taylor adaptive
// method for 100 years. The same integration is then made using single precision.

// The benchmark output is quite verbose, but the final numbers that count are the relative precision on r and v and the
// CPU time.

int main()
{
#if defined(HEYOKA_HAVE_REAL128)
    using namespace mppp::literals;

    // Bodies are Sun, Earth and Apophis.
    const auto masses = std::vector{1.989e30, 5.9722e24, 6.1e10};
    // Position at 2029-Apr-13 21:42:00.0000 in the J2000 frame (solar system barycenter)
    // Earth
    const auto r_e = std::vector{-1.370648479191993E+11_rq, -6.075143207036727E+10_rq, 9.502339686807245E+06_rq};
    const auto v_e = std::vector{1.155023262401610E+04_rq, -2.735128886751287E+04_rq, 1.777171492232554_rq};
    // Apophis
    const auto r_a = std::vector{-1.370898394232368E+11_rq, -6.072219715068178E+10_rq, 1.467144922637567E+07_rq};
    const auto v_a = std::vector{1.786753408315841E+04_rq, -2.390511764955029E+04_rq, 1.855718806173850E+03_rq};
    // Cavendish constant
    const auto G = 6.6743e-11;
    // The 3-body system we will study
    auto sys = make_nbody_sys(3u, kw::masses = masses, kw::Gconst = G);
    // The Taylor integrator in quadruple precision
    std::cout << "Constructing the Taylor integrator in quadruple precision ... " << std::flush;
    auto start = high_resolution_clock::now();
    taylor_adaptive<mppp::real128> taylor_q{sys,
                                            {// Sun in the origin, zero speed.
                                             0._rq, 0._rq, 0._rq, 0._rq, 0._rq, 0._rq,
                                             // Earth.
                                             r_e[0], r_e[1], r_e[2], v_e[0], v_e[1], v_e[2],
                                             // Apophis.
                                             r_a[0], r_a[1], r_a[2], v_a[0], v_a[1], v_a[2]}};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1e6 << "s" << std::endl;
    // Backward integration to establish the IC we will use (50 years)
    std::cout << "Numerical integration backward 50 years ... " << std::flush;
    start = high_resolution_clock::now();
    auto res = taylor_q.propagate_until(mppp::real128{-50. * 365.25 * 24. * 60. * 60.});
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1e6 << "s"
              << " -> return code -> " << static_cast<int>(std::get<0>(res)) << std::endl;
    std::cout << "Initial conditions computed:" << std::endl;
    to_screen<mppp::real128>(taylor_q);
    // Forward integration of 100 years
    std::vector<std::vector<mppp::real128>> baseline_s; // baseline states
    std::vector<mppp::real128> baseline_t;              // baseline times
    baseline_s.push_back(taylor_q.get_state());
    baseline_t.push_back(taylor_q.get_time());
    std::cout << "Numerical integration forward 100 years (quadruple) ... " << std::flush;
    start = high_resolution_clock::now();
    while (taylor_q.get_time() < 50. * 365.25 * 24. * 60. * 60.) {
        taylor_q.step();
        baseline_s.push_back(taylor_q.get_state());
        baseline_t.push_back(taylor_q.get_time());
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1e6 << "s -> baseline established" << std::endl;

    // The Taylor integrator in single precision
    std::cout << "\nConstructing the Taylor integrator in single precision ... " << std::flush;
    std::vector<double> ic_d(18);
    std::transform(ic_d.begin(), ic_d.end(), baseline_s[0].begin(), ic_d.begin(),
                   [](double d, mppp::real128 q) { return static_cast<double>(q); });
    start = high_resolution_clock::now();
    taylor_adaptive<double> taylor_d{sys, ic_d, kw::time = -50. * 365.25 * 24. * 60. * 60., kw::high_accuracy = true};
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1e6 << "s" << std::endl;
    to_screen<double>(taylor_d);
    // Forward integration (double) using the same time grid used by the quadruple precision integration
    std::vector<std::vector<double>> states_d; // states
    std::vector<double> times_d;               // baseline times
    std::cout << "Numerical integration forward 100 years (double) ... " << std::flush;
    start = high_resolution_clock::now();
    for (const auto t : baseline_t) {
        taylor_d.propagate_until(static_cast<double>(t));
        states_d.push_back(taylor_d.get_state());
        times_d.push_back(taylor_d.get_time());
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() / 1e6 << "s" << std::endl;

    for (decltype(times_d.size()) i = 0u; i < times_d.size(); ++i) {
        auto r_q = sqrt(baseline_s[i][12] * baseline_s[i][12] + baseline_s[i][13] * baseline_s[i][13]
                        + baseline_s[i][14] * baseline_s[i][14]);
        auto r_d = sqrt(states_d[i][12] * states_d[i][12] + states_d[i][13] * states_d[i][13]
                        + states_d[i][14] * states_d[i][14]);
        auto v_q = sqrt(baseline_s[i][15] * baseline_s[i][15] + baseline_s[i][16] * baseline_s[i][16]
                        + baseline_s[i][17] * baseline_s[i][17]);
        auto v_d = sqrt(states_d[i][15] * states_d[i][15] + states_d[i][16] * states_d[i][16]
                        + states_d[i][17] * states_d[i][17]);
        // Uncomment these lines to print to screen the error on the radius and velocity magnitude
        std::cout << times_d[i] << ", " << abs(r_d - r_q) << ", " << abs(v_q - v_d) << "," << std::endl;
    }
    return 0;
#else

    std::cout << "The mppp::real128 is not available, the benchmark will not be run.\n";

    return 0;
#endif
}
