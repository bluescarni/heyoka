// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <heyoka/detail/igor.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/logging.hpp>
#include <heyoka/mascon.hpp>
#include <heyoka/math.hpp>
#include <heyoka/taylor.hpp>

#include "data/mascon_67p.hpp"
#include "data/mascon_bennu.hpp"
#include "data/mascon_itokawa.hpp"

// This benchmark builds a Taylor integrator for the motion around asteroid Bennu.
// The mascon model for Bennu was generated using a thetraedral mesh built upon
// the polyhedral surface model available. Model units are L = asteroid diameter, M = asteroid mass.
// TODO: check the exact values

using namespace heyoka;
using namespace std::chrono;
namespace odeint = boost::numeric::odeint;

// Pairwise summation of a vector of doubles. Avoiding copies
// https://en.wikipedia.org/wiki/Pairwise_summation
double pairwise_sum(std::vector<double> &sum)
{
    if (sum.size() == std::numeric_limits<decltype(sum.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_sum()");
    }

    if (sum.empty()) {
        return 0.;
    }
    while (sum.size() != 1u) {
        for (decltype(sum.size()) i = 0; i < sum.size(); i += 2u) {
            if (i + 1u == sum.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                sum[i / 2u] = sum[i];
            } else {
                sum[i / 2u] = sum[i] + sum[i + 1u];
            }
        }
        if (sum.size() % 2) {
            sum.resize(sum.size() / 2 + 1);
        } else {
            sum.resize(sum.size() / 2);
        }
    }
    return sum[0];
}

// Here we implement the r.h.s. functor to be used with boost odeint routines.
struct mascon_dynamics {
    // data members
    std::vector<std::vector<double>> m_mascon_points;
    std::vector<double> m_mascon_masses;
    std::vector<double> x_acc, y_acc, z_acc;
    double m_p, m_q, m_r;
    double m_G;
    // constructor
    template <typename P, typename M>
    mascon_dynamics(const P &mascon_points, const M &mascon_masses, double p, double q, double r, double G)
        : m_mascon_masses(std::begin(mascon_masses), std::end(mascon_masses)), m_p(p), m_q(q), m_r(r), m_G(G)
    {
        for (auto i = 0u; i < std::size(m_mascon_masses); ++i) {
            m_mascon_points.push_back(
                std::vector<double>{mascon_points[i][0], mascon_points[i][1], mascon_points[i][2]});
        }
    }
    // call operator
    void operator()(const std::vector<double> &x, std::vector<double> &dxdt, const double /* t */)
    {
        auto dim = std::size(m_mascon_masses);
        x_acc.resize(std::size(m_mascon_masses));
        y_acc.resize(std::size(m_mascon_masses));
        z_acc.resize(std::size(m_mascon_masses));
        // FIRST: Assemble the acceleration due to mascons
        // TODO: switch to pairwise/compensated summation here?
        double x_a = 0, y_a = 0, z_a = 0;
        for (decltype(dim) i = 0; i < dim; ++i) {
            auto r2 = (x[0] - m_mascon_points[i][0]) * (x[0] - m_mascon_points[i][0])
                      + (x[1] - m_mascon_points[i][1]) * (x[1] - m_mascon_points[i][1])
                      + (x[2] - m_mascon_points[i][2]) * (x[2] - m_mascon_points[i][2]);
            // auto mGpow = m_G * std::pow(r2, -3. / 2.) * m_mascon_masses[i];
            auto r = std::sqrt(r2);
            auto mGpow = m_G * m_mascon_masses[i] / (r * r * r);
            x_acc[i] = (m_mascon_points[i][0] - x[0]) * mGpow;
            y_acc[i] = (m_mascon_points[i][1] - x[1]) * mGpow;
            z_acc[i] = (m_mascon_points[i][2] - x[2]) * mGpow;
        }
        x_a = pairwise_sum(x_acc);
        y_a = pairwise_sum(y_acc);
        z_a = pairwise_sum(z_acc);

        // SECOND: centripetal and Coriolis
        // w x w x r
        auto centripetal_x = -m_q * m_q * x[0] - m_r * m_r * x[0] + m_q * x[1] * m_p + m_r * x[2] * m_p;
        auto centripetal_y = -m_p * m_p * x[1] - m_r * m_r * x[1] + m_p * x[0] * m_q + m_r * x[2] * m_q;
        auto centripetal_z = -m_p * m_p * x[2] - m_q * m_q * x[2] + m_p * x[0] * m_r + m_q * x[1] * m_r;
        // 2 w x v
        auto coriolis_x = 2. * (m_q * x[5] - m_r * x[4]);
        auto coriolis_y = 2. * (m_r * x[3] - m_p * x[5]);
        auto coriolis_z = 2. * (m_p * x[4] - m_q * x[3]);

        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];
        dxdt[3] = x_a - centripetal_x - coriolis_x;
        dxdt[4] = y_a - centripetal_y - coriolis_y;
        dxdt[5] = z_a - centripetal_z - coriolis_z;
    }
};

template <typename P, typename M>
taylor_adaptive<double> taylor_factory(const P &mascon_points, const M &mascon_masses, double wz, double r0 = 2.,
                                       double incl = 45., double G = 1.)
{
    // Initial conditions
    auto v0y = std::cos(incl / 360 * 6.28) * std::sqrt(1. / r0) - wz * r0;
    auto v0z = std::sin(incl / 360 * 6.28) * std::sqrt(1. / r0);
    std::vector<double> ic = {r0, 0., 0., 0., v0y, v0z};
    // Constructing the integrator.
    auto eom = make_mascon_system(kw::points = mascon_points, kw::masses = mascon_masses,
                                  kw::omega = std::vector<double>{0., 0., wz}, kw::Gconst = G);
    auto start = high_resolution_clock::now();
    taylor_adaptive<double> taylor{eom, ic, kw::compact_mode = true, kw::tol = 1e-14};
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time to construct the integrator: " << duration.count() / 1e6 << "s" << std::endl;
    std::cout << "Decomposition size: " << taylor.get_decomposition().size() << '\n';
    return taylor;
}

template <typename P, typename M>
double compute_energy(const std::vector<double> x, const P &mascon_points, const M &mascon_masses, double p, double q,
                      double r, double G)
{
    auto energy = energy_mascon_system(kw::state = x, kw::points = mascon_points, kw::masses = mascon_masses,
                                       kw::omega = std::vector<double>{p, q, r}, kw::Gconst = G);
    return eval_dbl(energy, std::unordered_map<std::string, double>());
}

template <typename P, typename M>
void compare_taylor_vs_rkf(const P &mascon_points, const M &mascon_masses, taylor_adaptive<double> &taylor, double wz,
                           double test_time = 10.)
{
    // 1 - Initional conditions
    auto ic = taylor.get_state();
    double E0 = compute_energy(ic, mascon_points, mascon_masses, 0., 0., wz, 1.);
    // Declarations
    decltype(high_resolution_clock::now()) start, stop;
    decltype(duration_cast<microseconds>(stop - start)) duration;
    double energy;

    // TAYLOR ------------------------------------------------------------------------------
    start = high_resolution_clock::now();
    taylor.propagate_until(test_time);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Integration time (Taylor): " << duration.count() / 1e6 << "s" << std::endl;
    energy = compute_energy(taylor.get_state(), mascon_points, mascon_masses, 0., 0., wz, 1.);
    fmt::print("Energy error: {}\n", (energy - E0) / E0);
    // TAYLOR ------------------------------------------------------------------------------

    // RKF7(8) -----------------------------------------------------------------------------
    // The error stepper
    typedef odeint::runge_kutta_fehlberg78<std::vector<double>> error_stepper_type;
    // The adaptive strategy
    typedef odeint::controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    // The dynamics
    mascon_dynamics dynamics(mascon_points, mascon_masses, 0., 0., wz, 1.);
    start = high_resolution_clock::now();
    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(1.0e-14, 1.0e-14), dynamics, ic, 0.0,
                               test_time, 1e-8);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Integration time (RKF7(8)): " << duration.count() / 1e6 << "s" << std::endl;
    energy = compute_energy(ic, mascon_points, mascon_masses, 0., 0., wz, 1.);
    fmt::print("Energy error: {}\n", (energy - E0) / E0);
    // RKF7(8) -----------------------------------------------------------------------------
}

template <typename P, typename M>
void plot_data_taylor(const P &mascon_points, const M &mascon_masses, taylor_adaptive<double> &taylor, double wz,
                      double integration_time, unsigned N = 5000)
{
    double dt = integration_time / N;
    for (decltype(N) i = 0u; i < N + 1; ++i) {
        auto state = taylor.get_state();
        auto energy = compute_energy(state, mascon_points, mascon_masses, 0., 0., wz, 1.);
        fmt::print("[{}, {}, {}, {}, {}, {}, {}],\n", state[0], state[1], state[2], state[3], state[4], state[5],
                   energy);
        taylor.propagate_for(dt);
    }
}

template <typename P, typename M>
void plot_data_rkf78(const P &mascon_points, const M &mascon_masses, taylor_adaptive<double> &taylor, double wz,
                     double integration_time, unsigned N = 5000)
{
    auto ic = taylor.get_state();
    double dt = integration_time / N;
    // The error stepper
    typedef odeint::runge_kutta_fehlberg78<std::vector<double>> error_stepper_type;
    // The dynamics
    mascon_dynamics dynamics(mascon_points, mascon_masses, 0., 0., wz, 1.);
    for (decltype(N) i = 0u; i < N + 1; ++i) {
        auto energy = compute_energy(ic, mascon_points, mascon_masses, 0., 0., wz, 1.);
        fmt::print("[{}, {}, {}, {}, {}, {}, {}],\n", ic[0], ic[1], ic[2], ic[3], ic[4], ic[5], energy);
        odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(1.0e-14, 1.0e-14), dynamics, ic, 0.0, dt,
                                   1e-8);
    }
}

int main(int argc, char *argv[])
{
    set_logger_level_trace();

    auto inclination = 45.;              // degrees
    auto distance = 3.;                  // non dimensional units
    auto integration_time = 86400. * 1.; // seconds (1day of operations)

    // The non dimensional units L, T and M allow to compute the non dimensional period and hence the rotation speed.
    // 67P
    // L = 2380.7169179463426m (computed from the mascon model and since 67P is 3909.769775390625m long from the ESA
    // 3D model) M = (9.982E12 Kg (from wikipedia) G = 6.67430E-11 (wikipedia again) induced time units: T =
    // sqrt(L^3/G/M) = 4500.388359040116s The asteroid angular velocity in our units is thus Wz = 2pi / (12.4 * 60 * 60
    // / T) = 0.633440278094151
    auto T_67p = 4500.388359040116;
    auto wz_67p = 0.633440278094151;
    fmt::print("67P, {} mascons:\n", std::size(mascon_masses_67p));
    auto taylor_67p = taylor_factory(mascon_points_67p, mascon_masses_67p, wz_67p, distance, inclination, 1.);
    // compare_taylor_vs_rkf(mascon_points_67p, mascon_masses_67p, taylor_67p, wz_67p, integration_time / T_67p);
    // plot_data(mascon_points_67p, mascon_masses_67p, taylor_67p, wz_67p, integration_time / T_67p * 365.25, 5000u);

    // Bennu
    // L = 416.45655931190163m (computed from the mascon model and since Bennu  is 562.8699958324432m long from the NASA
    // 3D model) M = 7.329E10 Kg (from wikipedia) G = 6.67430E-11 (wikipedia again) induced time units: T =
    // sqrt(L^3/G/M) = 3842.6367987779804s The asteroid angular velocity in our units is thus Wz = 2pi / (4.29 * 60 * 60
    // / T) = 1.5633255034258877
    auto T_bennu = 3842.6367987779804;
    auto wz_bennu = 1.5633255034258877;
    fmt::print("\nBennu, {} mascons:\n", std::size(mascon_masses_bennu));
    auto taylor_bennu = taylor_factory(mascon_points_bennu, mascon_masses_bennu, wz_bennu, distance, inclination, 1.);
    compare_taylor_vs_rkf(mascon_points_bennu, mascon_masses_bennu, taylor_bennu, wz_bennu, integration_time / T_bennu);
    // plot_data(mascon_points_bennu, mascon_masses_bennu, taylor_bennu, wz_bennu, integration_time / T_bennu * 7,
    // 1000u);

    // Itokawa
    // L = 478.2689458860669m (computed from the mascon model and since Itokawa is 535.3104705810547m long from the NASA
    // 3D model) M = 3.51E10 Kg (from wikipedia) G = 6.67430E-11 (wikipedia again) induced time units: T = sqrt(L^3/G/M)
    // = 6833.636194780773s The asteroid angular velocity in our units is thus Wz = 2pi / (12.132 * 60 * 60 / T)
    // = 0.9830980174940738
    auto T_itokawa = 6833.636194780773;
    auto wz_itokawa = 0.9830980174940738;
    fmt::print("\nItokawa, {} mascons:\n", std::size(mascon_masses_itokawa));
    // auto taylor_itokawa
    //    = taylor_factory(mascon_points_itokawa, mascon_masses_itokawa, wz_itokawa, distance, inclination, 1.);
    // compare_taylor_vs_rkf(mascon_points_itokawa, mascon_masses_itokawa, taylor_itokawa, wz_itokawa,
    //                      integration_time / T_itokawa);
    // plot_data(mascon_points_itokawa, mascon_masses_itokawa, taylor_itokawa, wz_itokawa,
    //         integration_time / T_itokawa * 7, 1000u);
    plot_data_rkf78(mascon_points_bennu, mascon_masses_bennu, taylor_bennu, wz_bennu,
                    integration_time / T_bennu * 365.25, 1000u);

    return 0;
}