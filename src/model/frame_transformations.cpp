// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/frame_transformations.hpp>

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

} // namespace model

HEYOKA_END_NAMESPACE
