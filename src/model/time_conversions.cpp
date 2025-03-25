// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include <heyoka/callable.hpp>
#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/model/time_conversions.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

std::string delta_tt_tai_func::operator()(unsigned) const
{
    // NOTE: regardless of the required precision, we have an exact representation
    // of this constant in decimal format.
    return "32.184";
}

} // namespace detail

// NOLINTNEXTLINE(cert-err58-cpp)
const expression delta_tt_tai(func(constant("delta_tt_tai", detail::delta_tt_tai_func{}, "delta_tt_tai(32.184)")));

// Function to compute the difference between TDB and TT in seconds. The input argument is the number of TDB seconds
// elapsed from the epoch of J2000 (but see note at the end).
//
// TDB is a relativistic time scale depending on the masses and positions of the bodies in the solar system and
// the velocity of the Earth. In principle, for the accurate computation of the TDB-TT difference, one should
// either use accurate ephemeris for the positions of the bodies of the solar system (e.g., VSOP2013) or
// numerically integrate the dynamics of the solar system. Here we are using the simplified approach explained
// in the NASA NAIF documentation:
//
// https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/time.html#The%20Relationship%20between%20TT%20and%20TDB
//
// That is, we are assuming a Keplerian orbit for the Earth, which results in the TDB-TT difference being a periodic
// function. This is accurate to approximately 0.000030 seconds.
//
// As noted in the erfa documentation, although the input time is formally a TDB epoch, the corresponding TT epoch
// can be used with no practical effects on the accuracy of the computation.
//
// NOTE: erfa has a more complex model for this function which is accurate to ~3ns in the 1950-2050 interval.
// It looks like the erfa model may be implementable in the expression system, keep it in mind for future extensions.
expression delta_tdb_tt(const expression &time_expr)
{
    constexpr auto M0 = 6.239996;
    constexpr auto M1 = 1.99096871e-7;
    constexpr auto EB = 1.671e-2;
    constexpr auto K = 1.657e-3;

    const auto M = M0 + M1 * time_expr;
    const auto E = M + EB * sin(M);
    return K * sin(E);
}

} // namespace model

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::model::detail::delta_tt_tai_func, std::string, unsigned)
