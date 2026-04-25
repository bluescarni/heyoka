// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepDE.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/lagrange_prop.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

// Construct the expression for a Lagrange propagator, given in input the initial state, the gravitational parameter and
// the propagation time.
//
// NOTE: this is the classical Lagrange/F&G propagator and assumes a bound (elliptical) orbit: the specific orbital
// energy eps must be strictly negative, otherwise the semi-major axis computation a = -mu/(2*eps) breaks (parabolic:
// division by zero; hyperbolic: negative a).
std::pair<std::array<expression, 3>, std::array<expression, 3>>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
make_lagrange_prop(const std::array<expression, 3> &pos0, const std::array<expression, 3> &vel0,
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   const expression &mu, const expression &tm)
{
    const auto &[x0, y0, z0] = pos0;
    const auto &[vx0, vy0, vz0] = vel0;

    const auto v02 = sum({pow(vx0, 2.), pow(vy0, 2.), pow(vz0, 2.)});
    const auto r0 = sqrt(sum({pow(x0, 2.), pow(y0, 2.), pow(z0, 2.)}));
    const auto eps = v02 * 0.5 - mu / r0;
    const auto a = -mu / (2.0 * eps);

    const auto sigma0 = sum({x0 * vx0, y0 * vy0, z0 * vz0}) / sqrt(mu);
    const auto s0 = sigma0 / sqrt(a);
    const auto c0 = 1.0 - r0 / a;

    const auto n = sqrt(mu / pow(a, 3.));
    const auto DM = n * tm;

    const auto DE = kepDE(s0, c0, DM);

    const auto cDE = cos(DE);
    const auto sDE = sin(DE);

    const auto r = sum({a, (r0 - a) * cDE, sigma0 * sqrt(a) * sDE});

    const auto F = 1.0 - a / r0 * (1.0 - cDE);
    const auto G = a * sigma0 / sqrt(mu) * (1.0 - cDE) + r0 * sqrt(a / mu) * sDE;
    const auto Ft = -sqrt(mu * a) / (r * r0) * sDE;
    const auto Gt = 1.0 - a / r * (1.0 - cDE);

    std::array pos = {F * pos0[0] + G * vel0[0], F * pos0[1] + G * vel0[1], F * pos0[2] + G * vel0[2]};
    std::array vel = {Ft * pos0[0] + Gt * vel0[0], Ft * pos0[1] + Gt * vel0[1], Ft * pos0[2] + Gt * vel0[2]};

    return std::make_pair(std::move(pos), std::move(vel));
}

} // namespace model

HEYOKA_END_NAMESPACE
