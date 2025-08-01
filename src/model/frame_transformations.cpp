// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <utility>

#include <boost/math/constants/constants.hpp>

#include <heyoka/config.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/eop.hpp>
#include <heyoka/model/frame_transformations.hpp>
#include <heyoka/model/iau2006.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

namespace
{

// FK5@J2000->ICRS rotation matrix, stored in row-major format.
constexpr std::array fk5j2000_icrs_rot = {9.9999999999999278e-01,  1.1102233723050031e-07, 4.4118034269763241e-08,
                                          -1.1102233297408340e-07, 9.9999999999998912e-01, -9.6477927438885170e-08,
                                          -4.4118044980967761e-08, 9.6477922540797404e-08, 9.9999999999999434e-01};

// Implementation of the FK5@J2000<->ICRS rotations. If Layout == right, we get the FK5@J2000->ICRS rotation,
// while with Layout == left we get the ICRS->FK5@J2000 rotation. I.e., changing the layout transposes
// the rotation matrix.
template <typename Layout>
std::array<expression, 3> rot_fk5j2000_icrs_impl(const std::array<expression, 3> &xyz)
{
    const mdspan<const double, extents<std::size_t, 3, 3>, Layout> rot(fk5j2000_icrs_rot.data());

    std::array<expression, 3> retval;
    for (auto i = 0u; i < 3u; ++i) {
        retval[i] = sum({rot(i, 0) * xyz[0], rot(i, 1) * xyz[1], rot(i, 2) * xyz[2]});
    }

    return retval;
}

} // namespace

} // namespace detail

// Function to rotate the input xyz cartesian coordinates from the FK5 frame at the equinox of J2000.0
// to the ICRS frame.
//
// The rotation angles are taken from Table 1 in the standard reference:
//
// https://adsabs.harvard.edu/full/2000A%26A...354..732M
//
// The rotation matrix has been created in astropy with code similar to this:
//
// https://github.com/astropy/astropy/blob/07b8873e0f78fbdb7787960eabf19fe48914218c/astropy/coordinates/builtin_frames/icrs_fk5_transforms.py
std::array<expression, 3> rot_fk5j2000_icrs(const std::array<expression, 3> &xyz)
{
    return detail::rot_fk5j2000_icrs_impl<std::experimental::layout_right>(xyz);
}

// Inverse of fk5j2000_icrs().
std::array<expression, 3> rot_icrs_fk5j2000(const std::array<expression, 3> &xyz)
{
    return detail::rot_fk5j2000_icrs_impl<std::experimental::layout_left>(xyz);
}

namespace detail
{

namespace
{

// Helper to build the rotation matrix from itrs to tirs.
auto build_rot_itrs_tirs(const expression &time_expr, const eop_data &data)
{
    using std::cos;
    using std::sin;

    // Construct x_p, y_p and sp.
    const auto x_p = pm_x(kw::time_expr = time_expr, kw::eop_data = data);
    const auto y_p = pm_y(kw::time_expr = time_expr, kw::eop_data = data);
    const auto sp = -0.000047 * boost::math::constants::pi<double>() / (180 * 3600.);

    // Compute sin/cos of x_p, y_p and sp.
    const auto cxp = cos(x_p);
    const auto sxp = sin(x_p);
    const auto cyp = cos(y_p);
    const auto syp = sin(y_p);
    const auto csp = cos(sp);
    const auto ssp = sin(sp);

    // Create the entries of the rotation matrix.
    const auto R00 = cxp * csp;
    const auto R01 = -cyp * ssp + syp * sxp * csp;
    const auto R02 = -syp * ssp - cyp * sxp * csp;

    const auto R10 = cxp * ssp;
    const auto R11 = cyp * csp + syp * sxp * ssp;
    const auto R12 = syp * csp - cyp * sxp * ssp;

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const auto R20 = sxp;
    const auto R21 = -syp * cxp;
    const auto R22 = cyp * cxp;

    return std::array{R00, R01, R02, R10, R11, R12, R20, R21, R22};
}

std::array<expression, 3> rot_itrs_tirs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        const eop_data &data)
{
    // Build the rotation matrix.
    const auto [R00, R01, R02, R10, R11, R12, R20, R21, R22] = build_rot_itrs_tirs(time_expr, data);

    // Perform the rotation and return the result.
    const auto &[x, y, z] = xyz;
    return {sum({R00 * x, R01 * y, R02 * z}), sum({R10 * x, R11 * y, R12 * z}), sum({R20 * x, R21 * y, R22 * z})};
}

std::array<expression, 3> rot_tirs_cirs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        const eop_data &data)
{
    // NOTE: the transformation is a ROT3 with angle alpha = -era.
    const auto alpha = -era(kw::time_expr = time_expr, kw::eop_data = data);

    const auto calpha = cos(alpha);
    const auto salpha = sin(alpha);

    const auto &[x, y, z] = xyz;
    return {calpha * x + salpha * y, calpha * y - salpha * x, z};
}

// Helper to build the precession-nutation rotation quantities used in the rotation from cirs to icrs.
auto build_cirs_icrs_pn(const expression &time_expr, double thresh, const eop_data &data)
{
    // Construct the IAU2006 theory.
    const auto [X_pn, Y_pn, s_pn] = iau2006(kw::time_expr = time_expr, kw::thresh = thresh);

    // Correct X, Y and s with the eop data.
    const auto DX = dX(kw::time_expr = time_expr, kw::eop_data = data);
    const auto DY = dY(kw::time_expr = time_expr, kw::eop_data = data);

    const auto X = X_pn + DX;
    const auto Y = Y_pn + DY;
    // NOTE: The original s is defined as s_pn = -X_pn*Y_pn/2 + stuff. The new s is defined
    // as -X*Y/2 + stuff, where X = X_pn + DX and Y = Y_pn + DY. Hence, the relationship
    // below between s and s_pn.
    const auto s = s_pn - 0.5 * sum({X_pn * DY, Y_pn * DX, DX * DY});

    // Build the PN rotation matrix.
    const auto X2 = pow(X, 2.);
    const auto Y2 = pow(Y, 2.);
    const auto X2_p_Y2 = X2 + Y2;
    const auto XY = X * Y;
    const auto a = 0.5 + 0.125 * X2_p_Y2;

    // NOLINTBEGIN(performance-unnecessary-copy-initialization)
    const auto R00 = 1. - a * X2;
    const auto R01 = -a * XY;
    const auto R02 = X;

    const auto R10 = R01;
    const auto R11 = 1. - a * Y2;
    const auto R12 = Y;
    // NOLINTEND(performance-unnecessary-copy-initialization)

    const auto R20 = -X;
    const auto R21 = -Y;
    const auto R22 = 1. - a * X2_p_Y2;

    // Return the rotation matrix and s.
    return std::make_pair(std::array{R00, R01, R02, R10, R11, R12, R20, R21, R22}, s);
}

std::array<expression, 3> rot_cirs_icrs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        double thresh, const eop_data &data)
{
    // Build the rotation quantities.
    const auto [R_mat, s] = build_cirs_icrs_pn(time_expr, thresh, data);
    const auto &[R00, R01, R02, R10, R11, R12, R20, R21, R22] = R_mat;

    // First step: ROT3 with angle s.
    const auto cos_s = cos(s);
    const auto sin_s = sin(s);

    const auto &[x, y, z] = xyz;
    const auto x_tmp = cos_s * x + sin_s * y;
    const auto y_tmp = cos_s * y - sin_s * x;
    const auto z_tmp = z;

    // Second step: rotation by R_mat.
    return {sum({R00 * x_tmp, R01 * y_tmp, R02 * z_tmp}), sum({R10 * x_tmp, R11 * y_tmp, R12 * z_tmp}),
            sum({R20 * x_tmp, R21 * y_tmp, R22 * z_tmp})};
}

} // namespace

// NOTE: for the implementation of this rotation, see Vallado 3.7.1 and chapter 5 of the
// IERS conventions:
//
// https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf
//
// See also:
//
// https://hpiers.obspm.fr/iers/bul/bulb/explanatory.html
//
// NOTE: here the input time_expr is expected the measure time in TT Julian centuries since J2000. This time expression
// is passed as-is to the IAU2006 PN theory, even though the PN theory expects the input time in TDB centuries (and not
// TT). The error introduced by this discrepancy should be quite small, hence we ignore it for the time being.
std::array<expression, 3> rot_itrs_icrs_impl(const std::array<expression, 3> &xyz, const expression &time_expr,
                                             double thresh, const eop_data &data)
{
    // Step 1: ITRS -> TIRS (polar motion).
    const auto xyz_tirs = rot_itrs_tirs(xyz, time_expr, data);

    // Step 2: TIRS -> CIRS (accounting for the ERA).
    const auto xyz_cirs = rot_tirs_cirs(xyz_tirs, time_expr, data);

    // Step 3: precession-nutation.
    return rot_cirs_icrs(xyz_cirs, time_expr, thresh, data);
}

namespace
{

std::array<expression, 3> rot_icrs_cirs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        double thresh, const eop_data &data)
{
    // Build the rotation quantities.
    const auto [R_mat, s] = build_cirs_icrs_pn(time_expr, thresh, data);
    const auto &[R00, R01, R02, R10, R11, R12, R20, R21, R22] = R_mat;

    // First step: inverse R_mat rotation.
    // NOTE: although this does not look like a rotation matrix, it actually is - see
    // section 5.4.4 of https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf.
    // Thus, we can invert it as usual via its tranpose.
    const auto &[x, y, z] = xyz;
    const auto x_tmp = sum({R00 * x, R10 * y, R20 * z});
    const auto y_tmp = sum({R01 * x, R11 * y, R21 * z});
    const auto z_tmp = sum({R02 * x, R12 * y, R22 * z});

    // Second step: inverse ROT3 with angle s.
    const auto cos_s = cos(s);
    const auto sin_s = sin(s);

    return {cos_s * x_tmp - sin_s * y_tmp, sin_s * x_tmp + cos_s * y_tmp, z_tmp};
}

std::array<expression, 3> rot_cirs_tirs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        const eop_data &data)
{
    // NOTE: the transformation is an inverse ROT3 with angle alpha = -era.
    const auto alpha = -era(kw::time_expr = time_expr, kw::eop_data = data);

    const auto calpha = cos(alpha);
    const auto salpha = sin(alpha);

    const auto &[x, y, z] = xyz;
    return {calpha * x - salpha * y, salpha * x + calpha * y, z};
}

std::array<expression, 3> rot_tirs_itrs(const std::array<expression, 3> &xyz, const expression &time_expr,
                                        const eop_data &data)
{
    // Build the rotation matrix.
    const auto [R00, R01, R02, R10, R11, R12, R20, R21, R22] = build_rot_itrs_tirs(time_expr, data);

    // Perform the inverse rotation and return the result.
    const auto &[x, y, z] = xyz;
    return {sum({R00 * x, R10 * y, R20 * z}), sum({R01 * x, R11 * y, R21 * z}), sum({R02 * x, R12 * y, R22 * z})};
}

} // namespace

// NOTE: this is the inverse of rot_itrs_icrs().
std::array<expression, 3> rot_icrs_itrs_impl(const std::array<expression, 3> &xyz, const expression &time_expr,
                                             double thresh, const eop_data &data)
{
    // Step 1: from icrs to cirs.
    const auto xyz_cirs = rot_icrs_cirs(xyz, time_expr, thresh, data);

    // Step 2: from cirs to tirs.
    const auto xyz_tirs = rot_cirs_tirs(xyz_cirs, time_expr, data);

    // Step 3: from tirs to itrs.
    return rot_tirs_itrs(xyz_tirs, time_expr, data);
}

} // namespace detail

} // namespace model

HEYOKA_END_NAMESPACE
