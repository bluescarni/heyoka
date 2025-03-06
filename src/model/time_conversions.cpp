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
#include <heyoka/s11n.hpp>

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

namespace
{

// Constants for the implementation of delta_tdb_tt().
// NOTE: we have exact representations in decimal format of these constants.
#define HEYOKA_DEFINE_DELTA_TDB_TT_CONST(cname, cvalue)                                                                \
    class delta_tdb_tt_##cname##_func                                                                                  \
    {                                                                                                                  \
        friend class boost::serialization::access;                                                                     \
        template <typename Archive>                                                                                    \
        void serialize(Archive &, unsigned)                                                                            \
        {                                                                                                              \
        }                                                                                                              \
                                                                                                                       \
    public:                                                                                                            \
        [[nodiscard]] std::string operator()(unsigned) const                                                           \
        {                                                                                                              \
            return #cvalue;                                                                                            \
        }                                                                                                              \
    };                                                                                                                 \
    const expression delta_tdb_tt_##cname(func(constant("delta_tdb_tt_" #cname, detail::delta_tdb_tt_##cname##_func{}, \
                                                        "delta_tdb_tt_" #cname "(" #cvalue ")")));

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_DEFINE_DELTA_TDB_TT_CONST(K, 1.657e-3);
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_DEFINE_DELTA_TDB_TT_CONST(EB, 1.671e-2)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_DEFINE_DELTA_TDB_TT_CONST(M0, 6.239996)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_DEFINE_DELTA_TDB_TT_CONST(M1, 1.99096871e-7)

#undef HEYOKA_DEFINE_DELTA_TDB_TT_CONST

} // namespace

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
    const auto M = detail::delta_tdb_tt_M0 + detail::delta_tdb_tt_M1 * time_expr;
    const auto E = M + detail::delta_tdb_tt_EB * sin(M);
    return detail::delta_tdb_tt_K * sin(E);
}

} // namespace model

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT_IMPLEMENT(heyoka::model::detail::delta_tt_tai_func, std::string, unsigned)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka::model::detail::delta_tdb_tt_K_func, std::string, unsigned)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka::model::detail::delta_tdb_tt_EB_func, std::string, unsigned)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka::model::detail::delta_tdb_tt_M0_func, std::string, unsigned)
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka::model::detail::delta_tdb_tt_M1_func, std::string, unsigned)
