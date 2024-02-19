// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>

#include <boost/numeric/odeint.hpp>

using state_type = std::array<double, 2>;

namespace odeint = boost::numeric::odeint;

void pendulum(const state_type &q, state_type &dq, double)
{
    dq[0] = q[1];
    dq[1] = -9.8 * std::sin(q[0]);
}

void pendulum_back(const state_type &q, state_type &dq, double)
{
    dq[0] = -q[1];
    dq[1] = 9.8 * std::sin(q[0]);
}

int main()
{
    using error_stepper_type = odeint::runge_kutta_fehlberg78<state_type>;

    state_type ic = {0.05, 0.025};

    const auto tol = 1e-15;

    const auto dt = 1000.;

    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(tol, tol), &pendulum, ic, 0.0, dt, 1e-8);

    odeint::integrate_adaptive(odeint::make_controlled<error_stepper_type>(tol, tol), &pendulum_back, ic, 0.0, dt,
                               1e-8);

    const auto dx = ic[0] - 0.05;
    const auto dv = ic[1] - 0.025;

    std::cout << "Error: " << std::sqrt(dx * dx + dv * dv) << '\n';
}
