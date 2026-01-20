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

// Helper to construct the rotation matrix to the RSW frame. r and v are the Cartesian position and velocity defining
// the RSW frame.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 3> to_rsw_rotation_matrix(const std::array<expression, 3> &r,
                                                                const std::array<expression, 3> &v)
{
    // Construct the R unit vector.
    const auto r_norm = norm(r);
    const std::array unit_r = {r[0] / r_norm, r[1] / r_norm, r[2] / r_norm};

    // Construct the W unit vector.
    const std::array r_cross_v = cross_product(r, v);
    const auto rcv_norm = norm(r_cross_v);
    const std::array unit_w = {r_cross_v[0] / rcv_norm, r_cross_v[1] / rcv_norm, r_cross_v[2] / rcv_norm};

    // Construct the S unit vector.
    const std::array unit_s = cross_product(unit_w, unit_r);

    return {unit_r, unit_s, unit_w};
}

// Helper to compute the Keplerian angular rotation vector in the RSW frame. r and v are the Cartesian position and
// velocity defining the RSW frame.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> rsw_keplerian_omega(const std::array<expression, 3> &r, const std::array<expression, 3> &v)
{
    // NOTE: the Keplerian omega is aligned to the W unit vector, and its magnitude is |r x v|/r**2.
    const std::array r_cross_v = cross_product(r, v);
    const auto rcv_norm = norm(r_cross_v);
    const auto r2 = sum({pow(r[0], 2.), pow(r[1], 2.), pow(r[2], 2.)});

    return {0_dbl, 0_dbl, rcv_norm / r2};
}

} // namespace

} // namespace detail

// Transform a Cartesian state (pos, vel), expressed in the same non-rotating Cartesian reference frame as (r, v), into
// the RSW frame defined by the reference state (r, v). The velocity mapping assumes Keplerian motion.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 2> state_to_rsw(const std::array<expression, 3> &pos,
                                                      const std::array<expression, 3> &vel,
                                                      const std::array<expression, 3> &r,
                                                      const std::array<expression, 3> &v)
{
    // Fetch the rotation matrix to rsw.
    const auto R = detail::to_rsw_rotation_matrix(r, v);

    // Fetch the Keplerian angular rotation vector in the RSW basis.
    const auto omega = detail::rsw_keplerian_omega(r, v);

    // Compute the position and velocity relative to r and v.
    const auto [x, y, z] = std::array{pos[0] - r[0], pos[1] - r[1], pos[2] - r[2]};
    const auto [vx, vy, vz] = std::array{vel[0] - v[0], vel[1] - v[1], vel[2] - v[2]};

    // Compute the rotated relative position.
    auto xp = sum({R[0][0] * x, R[0][1] * y, R[0][2] * z});
    auto yp = sum({R[1][0] * x, R[1][1] * y, R[1][2] * z});
    auto zp = sum({R[2][0] * x, R[2][1] * y, R[2][2] * z});

    // Compute the rotated relative velocity.
    const auto vxp1 = sum({R[0][0] * vx, R[0][1] * vy, R[0][2] * vz});
    const auto vyp1 = sum({R[1][0] * vx, R[1][1] * vy, R[1][2] * vz});
    const auto vzp1 = sum({R[2][0] * vx, R[2][1] * vy, R[2][2] * vz});

    // Compute the cross(omega, x) term.
    //
    // NOTE: this needs both omega and x expressed in the RSW frame.
    const auto [vxp2, vyp2, vzp2] = detail::cross_product(omega, {xp, yp, zp});

    // Assemble and return the result.
    return {{{std::move(xp), std::move(yp), std::move(zp)}, {vxp1 - vxp2, vyp1 - vyp2, vzp1 - vzp2}}};
}

// Transform a Cartesian state (pos, vel), expressed in the same non-rotating Cartesian reference frame as (r, v), into
// the *inertial* variant of the RSW frame defined by the reference state (r, v). Contrary to the usual RSW frame, in
// the inertial RSW frame the axes are fixed in space with no translational or rotational motion.
//
// See the RSW_INERTIAL frame here:
//
// https://sanaregistry.org/r/orbit_relative_reference_frames/
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 2> state_to_rsw_inertial(const std::array<expression, 3> &pos,
                                                               const std::array<expression, 3> &vel,
                                                               const std::array<expression, 3> &r,
                                                               const std::array<expression, 3> &v)
{
    // Fetch the rotation matrix.
    const auto R = detail::to_rsw_rotation_matrix(r, v);

    // Compute the position relative to r.
    const auto [x, y, z] = std::array{pos[0] - r[0], pos[1] - r[1], pos[2] - r[2]};

    // Rotate the position.
    auto xp = sum({R[0][0] * x, R[0][1] * y, R[0][2] * z});
    auto yp = sum({R[1][0] * x, R[1][1] * y, R[1][2] * z});
    auto zp = sum({R[2][0] * x, R[2][1] * y, R[2][2] * z});

    // Rotate the velocity.
    const auto &[vx, vy, vz] = vel;
    auto vxp = sum({R[0][0] * vx, R[0][1] * vy, R[0][2] * vz});
    auto vyp = sum({R[1][0] * vx, R[1][1] * vy, R[1][2] * vz});
    auto vzp = sum({R[2][0] * vx, R[2][1] * vy, R[2][2] * vz});

    return {{{std::move(xp), std::move(yp), std::move(zp)}, {std::move(vxp), std::move(vyp), std::move(vzp)}}};
}

// Transform a Cartesian state (pos, vel) expressed in the RSW frame back to the original non-rotating reference frame.
// The RSW frame is defined by the reference position r and velocity v, which are expressed in the non-rotating frame.
// The velocity mapping assumes Keplerian motion.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 2> state_from_rsw(const std::array<expression, 3> &pos,
                                                        const std::array<expression, 3> &vel,
                                                        const std::array<expression, 3> &r,
                                                        const std::array<expression, 3> &v)
{
    // Fetch the rotation matrix to rsw.
    const auto R = detail::to_rsw_rotation_matrix(r, v);

    // Fetch the Keplerian angular rotation vector in the RSW basis.
    const auto omega = detail::rsw_keplerian_omega(r, v);

    // Compute the position.
    auto xp = sum({R[0][0] * pos[0], R[1][0] * pos[1], R[2][0] * pos[2], r[0]});
    auto yp = sum({R[0][1] * pos[0], R[1][1] * pos[1], R[2][1] * pos[2], r[1]});
    auto zp = sum({R[0][2] * pos[0], R[1][2] * pos[1], R[2][2] * pos[2], r[2]});

    // Compute the velocity.
    const auto omega_pos = detail::cross_product(omega, pos);
    const auto [t0, t1, t2] = std::array{vel[0] + omega_pos[0], vel[1] + omega_pos[1], vel[2] + omega_pos[2]};
    const auto u0 = sum({R[0][0] * t0, R[1][0] * t1, R[2][0] * t2});
    const auto u1 = sum({R[0][1] * t0, R[1][1] * t1, R[2][1] * t2});
    const auto u2 = sum({R[0][2] * t0, R[1][2] * t1, R[2][2] * t2});

    return {{{std::move(xp), std::move(yp), std::move(zp)}, {u0 + v[0], u1 + v[1], u2 + v[2]}}};
}

// Transform a Cartesian state (pos, vel) expressed in the *inertial* RSW frame back to the original non-rotating
// reference frame. The inertial RSW frame is defined by the reference position r and velocity v, which are expressed in
// the non-rotating frame. Contrary to the usual RSW frame, in the inertial RSW frame the axes are fixed in space with
// no translational or rotational motion.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<std::array<expression, 3>, 2> state_from_rsw_inertial(const std::array<expression, 3> &pos,
                                                                 const std::array<expression, 3> &vel,
                                                                 const std::array<expression, 3> &r,
                                                                 const std::array<expression, 3> &v)
{
    // Fetch the rotation matrix.
    const auto R = detail::to_rsw_rotation_matrix(r, v);

    // Compute the position.
    auto xp = sum({R[0][0] * pos[0], R[1][0] * pos[1], R[2][0] * pos[2], r[0]});
    auto yp = sum({R[0][1] * pos[0], R[1][1] * pos[1], R[2][1] * pos[2], r[1]});
    auto zp = sum({R[0][2] * pos[0], R[1][2] * pos[1], R[2][2] * pos[2], r[2]});

    // Compute the velocity.
    auto vxp = sum({R[0][0] * vel[0], R[1][0] * vel[1], R[2][0] * vel[2]});
    auto vyp = sum({R[0][1] * vel[0], R[1][1] * vel[1], R[2][1] * vel[2]});
    auto vzp = sum({R[0][2] * vel[0], R[1][2] * vel[1], R[2][2] * vel[2]});

    return {{{std::move(xp), std::move(yp), std::move(zp)}, {std::move(vxp), std::move(vyp), std::move(vzp)}}};
}

} // namespace model

HEYOKA_END_NAMESPACE
