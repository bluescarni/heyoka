// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/cart2geo.hpp>
#include <heyoka/model/dayfrac.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/model/elp2000.hpp>
#include <heyoka/model/eo_dynamics.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/model/frame_transformations.hpp>
#include <heyoka/model/nrlmsise00_tn.hpp>
#include <heyoka/model/sw.hpp>
#include <heyoka/model/vsop2013.hpp>
#include <heyoka/sw_data.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// A few constants used below.

// Time conversions.
constexpr auto secs_in_day = 86400.0;
constexpr auto days_in_year = 365.25;
constexpr auto secs_in_cy = secs_in_day * days_in_year * 100;
constexpr auto secs_in_mil = secs_in_cy * 10;

// AU in kilometres.
constexpr auto AU_km = 149597870.7;

// Gravitational parameters of the Sun and the Moon, rescaled to km. See:
//
// https://iau-a3.gitlab.io/NSFA/NSFA_cbe.html
constexpr auto sun_mu = 1.32712440041e20 / 1e9;
constexpr auto moon_mu = 4.902800145e12 / 1e9;

// Formulate the gravitational acceleration due to the Sun and the Moon on an Earth-orbiting spacecraft.
//
// xyz is the position vector of the spacecraft in the GCRS (km). vsop2013/elp2000_thresh are the thresholds to use when
// formulating the VSOP2013/ELP2000 theories.
[[nodiscard]] std::array<expression, 3>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
eo_dynamics_make_3rd_body_acc(const std::array<expression, 3> &xyz, const double elp2000_thresh,
                              const double vsop2013_thresh)
{
    // Fetch the Earth's mu in km**3/s**2.
    //
    // NOTE: we use the egm2008 value for this.
    const auto earth_mu = get_egm2008_mu() / 1e9;

    // Compute the Earth/Moon mass ratio. This is used to calculate the Earth's heliocentric position below.
    const auto mu_star = earth_mu / moon_mu;

    // Fetch the GCRS position (in km).
    const auto &[x, y, z] = xyz;

    // Compute the heliocentric state of the Earth-Moon barycentre (planet number 3 in the VSOP2013 theory).
    //
    // NOTE: the VSOP2013 theory requires in input the number of Julian millenia elapsed since JD 2451545.0 TDB. Our
    // time coordinate, on the other hand, is *seconds* since J2000 TT (and recall that J2000 TT == JD 2451545.0 TT). We
    // ignore the very small periodic differences between TDB and TT, and we rescale heyoka::time in order to account
    // for the switch from millenia to seconds.
    const auto EMB_state
        = vsop2013_cartesian_icrf(3, kw::time_expr = heyoka::time / secs_in_mil, kw::thresh = vsop2013_thresh);
    assert(EMB_state.size() == 6u);

    // Fetch the EMB position in km (VSOP2013 returns positions in AU, need to rescale).
    const auto x0EMB = EMB_state[0] * AU_km;
    const auto y0EMB = EMB_state[1] * AU_km;
    const auto z0EMB = EMB_state[2] * AU_km;

    // Compute the geocentric position of the Moon in the FK5@J2000 frame.
    //
    // NOTE: the ELP2000 theory requires in input the number of Julian centuries elapsed since JD 2451545.0 TDB. Our
    // time coordinate, on the other hand, is *seconds* since J2000 TT (and recall that J2000 TT == JD 2451545.0 TT). We
    // ignore the very small periodic differences between TDB and TT, and we rescale heyoka::time in order to account
    // for the switch from centuries to seconds.
    const auto moon_pos_fk5
        = elp2000_cartesian_fk5(kw::time_expr = heyoka::time / secs_in_cy, kw::thresh = elp2000_thresh);
    assert(moon_pos_fk5.size() == 3u);

    // Rotate the Moon's geocentric position to the GCRS frame.
    const auto moon_pos_gcrs = rot_fk5j2000_icrs({moon_pos_fk5[0], moon_pos_fk5[1], moon_pos_fk5[2]});

    // Negate moon_pos_gcrs to yield the Moon-centric position of the Earth in the ICRS frame.
    //
    // NOTE: no need for rescaling here, the ELP2000 theory already gives the positions in km.
    const auto x1E = -moon_pos_gcrs[0];
    const auto y1E = -moon_pos_gcrs[1];
    const auto z1E = -moon_pos_gcrs[2];

    // Compute the Earth's heliocentric position by combining the EMB's heliocentric position with the Moon-centric
    // position of the Earth.
    const auto x0E = x0EMB + x1E / (1.0 + mu_star);
    const auto y0E = y0EMB + y1E / (1.0 + mu_star);
    const auto z0E = z0EMB + z1E / (1.0 + mu_star);

    // Compute the contributions from the Sun.
    const auto x0P = x0E + x;
    const auto y0P = y0E + y;
    const auto z0P = z0E + z;
    const auto r0P_m3 = pow(sum({pow(x0P, 2.), pow(y0P, 2.), pow(z0P, 2.)}), -1.5);
    const auto r0E_m3 = pow(sum({pow(x0E, 2.), pow(y0E, 2.), pow(z0E, 2.)}), -1.5);

    // Compute the contributions from the Moon.
    const auto x1P = x1E + x;
    const auto y1P = y1E + y;
    const auto z1P = z1E + z;
    const auto r1P_m3 = pow(sum({pow(x1P, 2.), pow(y1P, 2.), pow(z1P, 2.)}), -1.5);
    const auto r1E_m3 = pow(sum({pow(x1E, 2.), pow(y1E, 2.), pow(z1E, 2.)}), -1.5);

    // Compute the Earth's acceleration due to the perturbers.
    const auto acc_earth_x = -(sun_mu * x0E * r0E_m3 + moon_mu * x1E * r1E_m3);
    const auto acc_earth_y = -(sun_mu * y0E * r0E_m3 + moon_mu * y1E * r1E_m3);
    const auto acc_earth_z = -(sun_mu * z0E * r0E_m3 + moon_mu * z1E * r1E_m3);

    // Compute the spacecraft's acceleration due to the perturbers.
    const auto acc_direct_x = -(sun_mu * x0P * r0P_m3 + moon_mu * x1P * r1P_m3);
    const auto acc_direct_y = -(sun_mu * y0P * r0P_m3 + moon_mu * y1P * r1P_m3);
    const auto acc_direct_z = -(sun_mu * z0P * r0P_m3 + moon_mu * z1P * r1P_m3);

    // Return the total perturbing acceleration.
    return {
        acc_direct_x - acc_earth_x,
        acc_direct_y - acc_earth_y,
        acc_direct_z - acc_earth_z,
    };
}

// Formulate the x/y/z components of the atmospheric drag acceleration.
//
// state is the state vector of the spacecraft in the GCRS (km, s). iau2006_thresh is the threshold to apply to the
// IAU2006 PN theory. edata/sdata are the EOP/SW datasets to be used in the formulation. Cb is the ballistic coefficient
// of the spacecraft, measured in m**2/kg.
[[nodiscard]] std::array<expression, 3> eo_dynamics_make_drag_acc(const std::array<expression, 6> &state,
                                                                  const double iau2006_thresh, const eop_data &edata,
                                                                  const sw_data &sdata, const expression &Cb)
{
    // Fetch the GCRS state (in km and sec).
    const auto &[x, y, z, vx, vy, vz] = state;

    // Introduce a variable to measure time in *seconds*.
    const auto tm = make_vars("tm");

    // NOTE: the time expressions in the frame rotations must represent the number of Julian centuries elapsed since
    // J2000. Since, as explained above, tm is measured in *seconds*, we need to rescale.
    const auto tm_jcy = tm / secs_in_cy;

    // Step 1: coordinate transformations.
    // -----------------------------------

    // Perform the GCSR->ITRS rotation for the position.
    const auto [x_itrs, y_itrs, z_itrs]
        = rot_icrs_itrs({x, y, z}, kw::thresh = iau2006_thresh, kw::time_expr = tm_jcy, kw::eop_data = edata);

    // Transform into geodetic coordinates.
    //
    // NOTE: the default R_eq value is in metres, we need to provide it in kilometres instead.
    //
    // NOTE: we use a_earth here (from the WGS84 model) rather than the egm2008 'a' value because conceptually it is
    // more correct: a_earth is defined as the semi-major axis of the geodetic reference ellipsoid, while the egm2008 is
    // a dimensional scale factor for the geopotential. They are very close in any case, so likely no practical
    // differences.
    const auto [h, lat, lon] = cart2geo({x_itrs, y_itrs, z_itrs}, kw::R_eq = a_earth / 1e3);

    // Create the variables to represent a fixed point in the ITRS.
    const auto [x0, y0, z0] = make_vars("x0", "y0", "z0");

    // Compute the position of the fixed point in the GCRS.
    const auto [x0_gcrs, y0_gcrs, z0_gcrs]
        = rot_itrs_icrs({x0, y0, z0}, kw::thresh = iau2006_thresh, kw::time_expr = tm_jcy, kw::eop_data = edata);

    // Compute the velocity of the fixed point in the GCRS.
    //
    // NOTE: we are differentiating wrt 'tm', and the positions are assumed in km, thus the velocities are in km/s.
    const auto v0_gcrs = diff_tensors({x0_gcrs, y0_gcrs, z0_gcrs}, {tm}).get_jacobian();
    assert(v0_gcrs.size() == 3u);
    const auto &vx0_gcrs = v0_gcrs[0];
    const auto &vy0_gcrs = v0_gcrs[1];
    const auto &vz0_gcrs = v0_gcrs[2];

    // Replace x0/y0/z0 with x_itrs/y_itrs/z_itrs in order to calculate the velocity of the atmosphere at the position
    // of the spacecraft.
    const auto v_atm = subs({vx0_gcrs, vy0_gcrs, vz0_gcrs}, {{x0, x_itrs}, {y0, y_itrs}, {z0, z_itrs}});
    assert(v_atm.size() == 3u);

    // Compute the velocity of the spacecraft relative to the atmosphere in the GCRS (km/s).
    const auto vrel_x = vx - v_atm[0];
    const auto vrel_y = vy - v_atm[1];
    const auto vrel_z = vz - v_atm[2];

    // Step 2: atmospheric model.
    // --------------------------

    // Time coordinate for use in the atmospheric model.
    //
    // NOTE: the atmospheric model requires the day fraction in input. The day fraction requires the number of *days*
    // elapsed since J2000 in input, hence the need to rescale tm.
    const auto tm_atm = dayfrac(kw::time_expr = tm / secs_in_day);

    // Introduce the space weather indices.
    //
    // NOTE: f107 needs to be offset to the day *before* tm, as required by the nrlmsise00 model.
    const auto f107 = heyoka::model::f107(kw::time_expr = (tm - secs_in_day) / secs_in_cy, kw::sw_data = sdata);
    const auto f107a = f107a_center81(kw::time_expr = tm / secs_in_cy, kw::sw_data = sdata);
    const auto ap = Ap_avg(kw::time_expr = tm / secs_in_cy, kw::sw_data = sdata);

    // Compute the atmospheric density.
    //
    // NOTE: the h geodetic coordinate is already in kilometres, no rescaling needed.
    auto rho = nrlmsise00_tn(kw::geodetic = {h, lat, lon}, kw::f107 = f107, kw::f107a = f107a, kw::ap = ap,
                             kw::time_expr = tm_atm);
    // NOTE: rho comes out in kg/m**3, but we are using km as unit of length. Rescale.
    rho *= 1e9;

    // Step 3: assembling the dynamics.
    // --------------------------------

    // Compute vrel.
    //
    // NOTE: in the computation of vrel = sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2), we run into the usual AD problem of
    // generating an indeterminate form 0/0 if vrel is zero during the computation of Taylor derivatives. In order to
    // avoid this, we adopt the pragmatic approach of adding a small epsilon**2 to the expression under sqrt(). We
    // choose 1e-8 so that its square is close to the FP epsilon, keeping in mind that speeds in LEO (in km/s) are
    // roughly of the order of 1.
    constexpr auto veps = 1e-8;
    const auto vrel = sqrt(sum({pow(vrel_x, 2.), pow(vrel_y, 2.), pow(vrel_z, 2.), expression{veps * veps}}));

    // Compute the components of the drag acceleration.
    //
    // NOTE: Cb needs to be rescaled because it is assumed to be provided in m**2/kg, but the dynamics is formulated in
    // kilometres rather than metres.
    const auto drag_factor = prod({-0.5_dbl, rho, vrel, Cb, 1e-6_dbl});
    const auto acc_drag_x = drag_factor * vrel_x;
    const auto acc_drag_y = drag_factor * vrel_y;
    const auto acc_drag_z = drag_factor * vrel_z;

    // As a final step, replace the time variable tm with heyoka::time.
    const auto acc_drag = subs({acc_drag_x, acc_drag_y, acc_drag_z}, {{tm, heyoka::time}});
    assert(acc_drag.size() == 3u);

    return {acc_drag[0], acc_drag[1], acc_drag[2]};
}

[[nodiscard]] std::vector<std::pair<expression, expression>>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
make_eo_dynamics(const std::uint32_t max_geo_degree, const std::uint32_t max_geo_order, const double iau2006_thresh,
                 const eop_data &eop_data, const sw_data &sw_data, const std::optional<expression> &Cb_opt,
                 const std::optional<double> &elp2000_thresh_opt, const std::optional<double> &vsop2013_thresh_opt)
{
    // Prepare the return value.
    std::vector<std::pair<expression, expression>> dyn;

    // Define the Cartesian state variables.
    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    // Fetch the Earth's mu from the egm2008 model and transform it in km**3/s**2.
    const auto earth_mu = get_egm2008_mu() / 1e9;

    if (max_geo_degree == 0u && max_geo_order == 0u) {
        // Keplerian geopotential - no need to rotate into the ITRS and back, just use the fixed centres model.
        dyn = fixed_centres(kw::Gconst = earth_mu, kw::positions = {0., 0., 0.}, kw::masses = {1.});
    } else {
        // Fetch the Earth's reference radius from the egm2008 model and transform it in km.
        const auto earth_a = get_egm2008_a() / 1e3;

        // NOTE: the time expressions in the frame rotations must represent the number of Julian centuries elapsed since
        // J2000. Since we are measuring time in *seconds*, we need to rescale.
        const auto tm_jcy = time / secs_in_cy;

        // Express the position in the ITRS as a function of the position in the GCRS and time.
        const auto [x_itrs, y_itrs, z_itrs]
            = rot_icrs_itrs({x, y, z}, kw::thresh = iau2006_thresh, kw::time_expr = tm_jcy, kw::eop_data = eop_data);

        // Compute the acceleration due to the geopotential in the ITRS.
        const auto [acc_x_itrs, acc_y_itrs, acc_z_itrs]
            = egm2008_acc({x_itrs, y_itrs, z_itrs}, max_geo_degree, max_geo_order, kw::mu = earth_mu, kw::a = earth_a);

        // Rotate the accelerations back to the GCRS.
        const auto [acc_x_gcrs, acc_y_gcrs, acc_z_gcrs]
            = rot_itrs_icrs({acc_x_itrs, acc_y_itrs, acc_z_itrs}, kw::thresh = iau2006_thresh, kw::time_expr = tm_jcy,
                            kw::eop_data = eop_data);

        // Assign the dynamics.
        dyn = std::vector{prime(x) = vx,          prime(y) = vy,          prime(z) = vz,
                          prime(vx) = acc_x_gcrs, prime(vy) = acc_y_gcrs, prime(vz) = acc_z_gcrs};
    }

    // Add the atmospheric drag, if requested.
    if (Cb_opt) {
        // Compute the components of the drag acceleration in the GCRS.
        const auto [drag_x, drag_y, drag_z]
            = eo_dynamics_make_drag_acc({x, y, z, vx, vy, vz}, iau2006_thresh, eop_data, sw_data, *Cb_opt);

        // Add them.
        dyn[3].second += drag_x;
        dyn[4].second += drag_y;
        dyn[5].second += drag_z;
    }

    // Add the 3rd body perturbations, if requested.
    if (elp2000_thresh_opt && vsop2013_thresh_opt) {
        // Compute the components of the third body perturbations in the GCRS.
        const auto [tb_acc_x, tb_acc_y, tb_acc_z]
            = eo_dynamics_make_3rd_body_acc({x, y, z}, *elp2000_thresh_opt, *vsop2013_thresh_opt);

        // Add them.
        dyn[3].second += tb_acc_x;
        dyn[4].second += tb_acc_y;
        dyn[5].second += tb_acc_z;
    }

    return dyn;
}

} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::vector<std::pair<expression, expression>>
eo_dynamics_impl(const std::uint32_t max_geo_degree, const std::uint32_t max_geo_order, const double iau2006_thresh,
                 const eop_data &eop_data, const sw_data &sw_data, const std::optional<expression> &Cb_opt,
                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                 const std::optional<double> &elp2000_thresh_opt, const std::optional<double> &vsop2013_thresh_opt)
{
    if (static_cast<bool>(elp2000_thresh_opt) != static_cast<bool>(vsop2013_thresh_opt)) [[unlikely]] {
        throw std::invalid_argument("Invalid arguments detected in eo_dynamics(): the 'vsop2013_thresh' and "
                                    "'elp2000_thresh' arguments must both be either present or absent");
    }

    return make_eo_dynamics(max_geo_degree, max_geo_order, iau2006_thresh, eop_data, sw_data, Cb_opt,
                            elp2000_thresh_opt, vsop2013_thresh_opt);
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
