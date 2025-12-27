// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/frame_transformations.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

namespace
{

expression norm(const std::array<expression, 3> &v)
{
    return sqrt(sum({pow(v[0], 2.), pow(v[1], 2.), pow(v[2], 2.)}));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> cross_product(const std::array<expression, 3> &a, const std::array<expression, 3> &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Helper to construct the rotation matrix to the RSW frame and its time derivative. r and v are the Cartesian position
// and velocity defining the RSW frame. Keplerian motion is assumed when constructing the time derivative of the
// rotation matrix.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<std::array<expression, 3>, 3>, 2> to_rsw_rotation_matrix(const std::array<expression, 3> &r,
                                                                               const std::array<expression, 3> &v)
{
    // Construct the R unit vector.
    const auto r2 = sum({pow(r[0], 2.), pow(r[1], 2.), pow(r[2], 2.)});
    const auto r_norm = sqrt(r2);
    const std::array unit_r = {r[0] / r_norm, r[1] / r_norm, r[2] / r_norm};

    // Construct the W unit vector.
    const std::array r_cross_v = cross_product(r, v);
    const auto rcv_norm = norm(r_cross_v);
    const std::array unit_w = {r_cross_v[0] / rcv_norm, r_cross_v[1] / rcv_norm, r_cross_v[2] / rcv_norm};

    // Construct the S unit vector.
    const std::array unit_s = cross_product(unit_w, unit_r);

    // Compute the magnitude of the angular velocity.
    const auto omega = rcv_norm / r2;

    // Construct dot(R).
    const std::array dot_unit_r = {omega * unit_s[0], omega * unit_s[1], omega * unit_s[2]};

    // Construct dot(S).
    const std::array dot_unit_s = {-omega * unit_r[0], -omega * unit_r[1], -omega * unit_r[2]};

    // Assemble and return the matrices.
    return {{{unit_r, unit_s, unit_w}, {dot_unit_r, dot_unit_s, {0_dbl, 0_dbl, 0_dbl}}}};
}

} // namespace

} // namespace detail

// Helper to transform a Cartesian state (position=pos, velocity=vel) into the RSW frame defined by the position r and
// velocity v. Keplerian motion is assumed for the velocity transformation.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 2> state_to_rsw(const std::array<expression, 3> &pos,
                                                      const std::array<expression, 3> &vel,
                                                      const std::array<expression, 3> &r,
                                                      const std::array<expression, 3> &v)
{
    // Fetch the rotation matrix to rsw and its derivative.
    const auto [R, Rp] = detail::to_rsw_rotation_matrix(r, v);

    // Compute the position and velocity relative to r and v.
    const auto [x, y, z] = std::array{pos[0] - r[0], pos[1] - r[1], pos[2] - r[2]};
    const auto [vx, vy, vz] = std::array{vel[0] - v[0], vel[1] - v[1], vel[2] - v[2]};

    // Rotate the position.
    auto xp = sum({R[0][0] * x, R[0][1] * y, R[0][2] * z});
    auto yp = sum({R[1][0] * x, R[1][1] * y, R[1][2] * z});
    auto zp = sum({R[2][0] * x, R[2][1] * y, R[2][2] * z});

    // Rotate the velocity.
    const auto vxp1 = sum({Rp[0][0] * x, Rp[0][1] * y, Rp[0][2] * z});
    const auto vyp1 = sum({Rp[1][0] * x, Rp[1][1] * y, Rp[1][2] * z});
    const auto vzp1 = sum({Rp[2][0] * x, Rp[2][1] * y, Rp[2][2] * z});

    const auto vxp2 = sum({R[0][0] * vx, R[0][1] * vy, R[0][2] * vz});
    const auto vyp2 = sum({R[1][0] * vx, R[1][1] * vy, R[1][2] * vz});
    const auto vzp2 = sum({R[2][0] * vx, R[2][1] * vy, R[2][2] * vz});

    return {
        {{std::move(xp), std::move(yp), std::move(zp)}, {vxp1 + vxp2, vyp1 + vyp2, vzp1 + vzp2}},
    };
}

} // namespace model

HEYOKA_END_NAMESPACE
