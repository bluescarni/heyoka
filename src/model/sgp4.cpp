// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/task_arena.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/dfun.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/math/logical.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

// The erfa functions for conversion between UTC and TAI Julian dates.
extern "C" {
int eraUtctai(double, double, double *, double *);
int eraTaiutc(double, double, double *, double *);
}

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace
{

// Constants.
//
// NOTE: here we are using the values from the wgs72 model,
// taken from the official C++ source code at:
//
// https://celestrak.org/software/vallado-sw.php
//
// Note that the original celestrak report #3, where sgp4
// was first introduced, uses constants which are very slighly
// different from the wgs72 model constants (specifically,
// the KE constant differs). See the 'gravconsttype' enum
// and the getgravconst() function in the official C++ source code.
//
// According to the sgp4 Python documentation here
//
// https://pypi.org/project/sgp4/
//
// for optimal compatibility with existing tools, TLEs, etc.,
// it is better to keep on using the wgs72 model constants,
// even if using the newer wgs84 model constants could in
// principle lead to slightly more accurate predictions.
//
// If needed, in the future we can implement the ability to select
// the constants to use upon construction of the propagator class.
constexpr auto KMPER = 6378.135;
constexpr auto SIMPHT = 220. / KMPER;
constexpr auto KE = 0.07436691613317342;
constexpr auto TOTHRD = 2 / 3.;
constexpr auto J2 = 1.082616e-3;
constexpr auto CK2 = .5 * J2;
constexpr auto S0 = 20. / KMPER;
constexpr auto S1 = 78. / KMPER;
constexpr auto Q0 = 120. / KMPER;
constexpr auto J3 = -0.253881e-5;
constexpr auto A3OVK2 = -J3 / CK2;
constexpr auto J4 = -0.00000165597;
constexpr auto CK4 = -.375 * J4;

// NOTE: this is the first half of the SGP4 algorithm, which does not depend on
// the propagation time. 'inputs' is the set of 8 expressions representing the input
// quantities for the sgp4 algorithm (if empty, variables with predefined names will be
// used as inputs).
auto sgp4_init(const std::vector<expression> &inputs)
{
    assert(inputs.empty() || inputs.size() == 8u);

    // Several math wrappers used in the original fortran code.
    constexpr auto heyoka_ABS = [](const auto &x) { return select(gte(x, 0.), x, -x); };

    constexpr auto heyoka_MAX = [](const auto &a, const auto &b) { return select(gt(a, b), a, b); };

    constexpr auto heyoka_MIN = [](const auto &a, const auto &b) { return select(lt(a, b), a, b); };

    // Setup the inputs. If they are provided, use them, otherwise use variables with predefined names.
    // NOTE: in this function we do not use tsince, i.e., the last element of inputs.
    const auto [N0, E0, I0, NODE0, OMEGA0, M0, BSTAR] = [&inputs]() {
        if (inputs.empty()) {
            return make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar");
        } else {
            return std::array{inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]};
        }
    }();

    // Recover original mean motion (N0DP) and semimajor axis (A0DP) from input elements.
    const auto A1 = pow(KE / N0, TOTHRD);
    const auto COSI0 = cos(I0);
    const auto THETA2 = pow(COSI0, 2.);
    const auto X3THM1 = 3. * THETA2 - 1.;
    const auto BETA02 = 1. - pow(E0, 2.);
    const auto BETA0 = sqrt(BETA02);
    const auto DELA2 = 1.5 * CK2 * X3THM1 / (BETA0 * BETA02);
    const auto DEL1 = DELA2 / pow(A1, 2.);
    const auto A0 = A1 * (1. - DEL1 * (1. / 3. + DEL1 * (1. + 134. / 81. * DEL1)));
    const auto DEL0 = DELA2 / pow(A0, 2.);
    const auto N0DP = N0 / (1. + DEL0);

    // Initialization for new element set.
    //
    // NOTE: for the computation of A0DP we use the new C++ code. The original
    // Fortran code used this instead:
    //
    // A0DP = A0/(1.-DEL0)
    //
    // Not entirely sure how these two definitions
    // relate - perhaps the original one is an
    // approximated form of the updated one?
    const auto A0DP = pow(KE / N0DP, TOTHRD);
    const auto PERIGE = A0DP * (1. - E0) - 1.;
    const auto S = heyoka_MIN(heyoka_MAX(S0, PERIGE - S1), S1);
    const auto S4 = 1. + S;
    const auto PINVSQ = 1. / pow(A0DP * BETA02, 2.);
    const auto XI = 1. / (A0DP - S4);
    const auto ETA = A0DP * XI * E0;
    const auto ETASQ = pow(ETA, 2.);
    const auto EETA = E0 * ETA;
    const auto PSISQ = heyoka_ABS(1. - ETASQ);
    const auto COEF = pow((Q0 - S) * XI, 4.);
    const auto COEF1 = COEF / (sqrt(PSISQ) * pow(PSISQ, 3.));
    const auto C1 = BSTAR * COEF1 * N0DP
                    * (A0DP * (1. + 1.5 * ETASQ + EETA * (4. + ETASQ))
                       + 0.75 * CK2 * XI / PSISQ * X3THM1 * (8. + 3. * ETASQ * (8. + ETASQ)));

    const auto SINI0 = sin(I0);
    // NOTE: fix for low eccentricity.
    const auto C3 = select(gt(E0, 1.0e-4), COEF * XI * A3OVK2 * N0DP * SINI0 / E0, 0.);
    const auto X1MTH2 = 1. - THETA2;
    const auto C4 = 2. * N0DP * COEF1 * A0DP * BETA02
                    * (ETA * (2. + .5 * ETASQ) + E0 * (.5 + 2. * ETASQ)
                       - 2. * CK2 * XI / (A0DP * PSISQ)
                             * (-3. * X3THM1 * (1. - 2. * EETA + ETASQ * (1.5 - .5 * EETA))
                                + .75 * X1MTH2 * (2. * ETASQ - EETA * (1. + ETASQ)) * cos(2. * OMEGA0)));
    const auto C5 = 2. * COEF1 * A0DP * BETA02 * (1. + 2.75 * (ETASQ + EETA) + EETA * ETASQ);
    const auto THETA4 = pow(THETA2, 2.);
    const auto TEMP1 = 3. * CK2 * PINVSQ * N0DP;
    const auto TEMP2 = TEMP1 * CK2 * PINVSQ;
    const auto TEMP3 = 1.25 * CK4 * pow(PINVSQ, 2.) * N0DP;
    const auto MDOT = N0DP + .5 * TEMP1 * BETA0 * X3THM1 + .0625 * TEMP2 * BETA0 * (13. - 78. * THETA2 + 137. * THETA4);
    const auto OMGDOT = -.5 * TEMP1 * (1. - 5. * THETA2) + 0.0625 * TEMP2 * (7. - 114. * THETA2 + 395. * THETA4)
                        + TEMP3 * (3. - 36. * THETA2 + 49. * THETA4);
    const auto HDOT1 = -TEMP1 * COSI0;
    const auto N0DOT = HDOT1 + (.5 * TEMP2 * (4. - 19. * THETA2) + 2. * TEMP3 * (3. - 7. * THETA2)) * COSI0;
    const auto OMGCOF = BSTAR * C3 * cos(OMEGA0);
    // NOTE: fix for low eccentricity.
    const auto MCOF = select(gt(E0, 1.0e-4), -TOTHRD * COEF * BSTAR / EETA, 0.);
    const auto NODCF = 3.5 * BETA02 * HDOT1 * C1;
    const auto T2COF = 1.5 * C1;
    // NOTE: fix for inclination close to 180 degrees.
    const auto LCOF
        = .125 * A3OVK2 * SINI0 * (3. + 5. * COSI0) / select(gt(heyoka_ABS(1. + COSI0), 1.5e-12), 1. + COSI0, 1.5e-12);
    const auto AYCOF = .25 * A3OVK2 * SINI0;
    const auto DELM0 = pow(1. + ETA * cos(M0), 3.);
    const auto SINM0 = sin(M0);
    const auto X7THM1 = 7. * THETA2 - 1.;

    // For perigee less than 220 kilometers, the equations are
    // truncated to linear variation in sqrt A and quadratic
    // variation in mean anomaly. Also, the C3 term, the
    // delta OMEGA term, and the delta M term are dropped.
    const auto C1SQ = pow(C1, 2.);
    const auto D2 = 4. * A0DP * XI * C1SQ;
    const auto TEMP0 = D2 * XI * C1 / 3.;
    const auto D3 = (17. * A0DP + S4) * TEMP0;
    const auto D4 = .5 * TEMP0 * A0DP * XI * (221. * A0DP + 31. * S4) * C1;
    const auto T3COF = D2 + 2. * C1SQ;
    const auto T4COF = .25 * (3. * D3 + C1 * (12. * D2 + 10. * C1SQ));
    const auto T5COF = .2 * (3. * D4 + 12. * C1 * D3 + 6. * pow(D2, 2.) + 15. * C1SQ * (2. * D2 + C1SQ));

    return std::array{MDOT,   OMGDOT, N0DOT, NODCF, C4,     C1,     T2COF,  MCOF,  ETA,   DELM0,
                      OMGCOF, PERIGE, C5,    SINM0, D2,     D3,     D4,     T3COF, T4COF, T5COF,
                      A0DP,   AYCOF,  LCOF,  N0DP,  X3THM1, X1MTH2, X7THM1, COSI0, SINI0};
}

// This is the second stage of the SGP4 algorithm, that performs the actual time
// propagation and returns the Cartesian state. 's' is the array of intermediate quantities
// computed by sgp4_init(). 'inputs' is the set of 8 expressions representing the input
// quantities for the sgp4 algorithm (if empty, variables with predefined names will be
// used as inputs).
std::vector<expression> sgp4_time_prop(const std::vector<expression> &inputs, const auto &s)
{
    assert(inputs.empty() || inputs.size() == 8u);

    // This is an atan2() implementation that returns angles
    // in the [0, 2pi] range.
    constexpr auto ACTAN = [](const auto &a, const auto &b) {
        const auto ret = atan2(a, b);
        return select(gte(ret, 0.), ret, 2. * heyoka::pi + ret);
    };

    // Expressions representing the orbital elements, bstar and tsince.
    // NOTE: in this function we do not use n0, i.e., the first element of 'inputs'.
    const auto [E0, I0, NODE0, OMEGA0, M0, BSTAR, TSINCE] = [&inputs]() {
        if (inputs.empty()) {
            return make_vars("e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");
        } else {
            return std::array{inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7]};
        }
    }();

    // Fetch the expressions for the intermediate quantities.
    const auto &[MDOT, OMGDOT, N0DOT, NODCF, C4, C1, T2COF, MCOF, ETA, DELM0, OMGCOF, PERIGE, C5, SINM0, D2, D3, D4,
                 T3COF, T4COF, T5COF, A0DP, AYCOF, LCOF, N0DP, X3THM1, X1MTH2, X7THM1, COSI0, SINI0]
        = s;

    // First error code check: un-Kozaied mean motion is not positive.
    auto ERRORCODE = select(lte(N0DP, 0.), 2., 0.);

    // Update for secular gravity and atmospheric drag.
    auto MP = M0 + MDOT * TSINCE;
    auto OMEGA = OMEGA0 + OMGDOT * TSINCE;
    const auto NODE = NODE0 + (N0DOT + NODCF * TSINCE) * TSINCE;
    auto TEMPE = C4 * TSINCE;
    auto TEMPA = 1. - C1 * TSINCE;
    auto TEMPL = T2COF;
    const auto TEMPF = MCOF * (pow(1. + ETA * cos(MP), 3.) - DELM0) + OMGCOF * TSINCE;
    // The conditional updates.
    MP = MP + select(gte(PERIGE, SIMPHT), TEMPF, 0.);
    OMEGA = OMEGA - select(gte(PERIGE, SIMPHT), TEMPF, 0.);
    TEMPE = TEMPE + select(gte(PERIGE, SIMPHT), C5 * (sin(MP) - SINM0), 0.);
    TEMPA = TEMPA - select(gte(PERIGE, SIMPHT), (D2 + (D3 + D4 * TSINCE) * TSINCE) * pow(TSINCE, 2.), 0.);
    TEMPL = TEMPL + select(gte(PERIGE, SIMPHT), (T3COF + (T4COF + T5COF * TSINCE) * TSINCE) * TSINCE, 0.);
    const auto A = A0DP * pow(TEMPA, 2.);
    const auto N = KE / pow(A, 3. / 2);
    auto E = E0 - TEMPE * BSTAR;

    // Second error code check: invalid eccentricity.
    ERRORCODE = select(eq(ERRORCODE, 0.), select(logical_or({gte(E, 1.), lt(E, -0.001)}), 1., 0.), ERRORCODE);

    // NOTE: fix for low eccentricity.
    E = select(lt(E, 1e-6), 1e-6, E);
    TEMPL = TEMPL * pow(TSINCE, 2.);

    // Long period periodics.
    const auto AXN = E * cos(OMEGA);
    const auto AB = A * (1. - pow(E, 2.));
    const auto AYN = AYCOF / AB + E * sin(OMEGA);

    // Solve Kepler's equation.
    // NOTE: the original celestrak report #3 (on page 13) says that this step is about solving
    // Kepler's equations for E + omega. This is a quantity similar to the eccentric
    // longitude F = F(h, k, lambda), implemented in heyoka as kepF():
    //
    // https://articles.adsabs.harvard.edu//full/1972CeMec...5..303B/0000309.000.html
    //
    // Indeed, the numerical iteration proposed in the report is nothing but a
    // Newton-Raphson step for the computation of F(h=AYN, k=AXN, lambda=CAPU). Also, it can
    // be shown that E + omega satisfies the same Kepler-like equation as F does, with
    // k replaced by e*cos(omega) and h replaced by e*sin(omega). Thus, we avoid
    // here the proposed iteration and compute F using directly the heyoka function.
    const auto CAPU = LCOF * AXN / AB + MP + OMEGA + N0DP * TEMPL;
    const auto EPWNEW = kepF(AYN, AXN, CAPU);
    const auto SINEPW = sin(EPWNEW);
    const auto COSEPW = cos(EPWNEW);
    const auto ESINE = AXN * SINEPW - AYN * COSEPW;
    const auto ECOSE = AXN * COSEPW + AYN * SINEPW;

    // Short period preliminary quantities
    const auto ELSQ = pow(AXN, 2.) + pow(AYN, 2.);
    const auto TEMPS = 1. - ELSQ;
    const auto PL = A * TEMPS;

    // Third error code check: pl < 0.
    ERRORCODE = select(eq(ERRORCODE, 0.), select(lt(PL, 0.), 4., 0.), ERRORCODE);

    const auto R = A * (1. - ECOSE);
    const auto RDOT = KE * sqrt(A) * ESINE / R;
    const auto RFDOT = KE * sqrt(PL) / R;
    const auto BETAL = sqrt(TEMPS);
    const auto TEMP3 = ESINE / (1. + BETAL);
    const auto COSU = (COSEPW - AXN + AYN * TEMP3) * A / R;
    const auto SINU = (SINEPW - AYN - AXN * TEMP3) * A / R;
    const auto U = ACTAN(SINU, COSU);
    const auto SIN2U = 2. * SINU * COSU;
    const auto COS2U = 2. * pow(COSU, 2.) - 1.;
    const auto TEMP1 = CK2 / PL;
    const auto TEMP2 = TEMP1 / PL;

    // Update for short periodics.
    const auto RK = R * (1. - 1.5 * TEMP2 * BETAL * X3THM1) + .5 * TEMP1 * X1MTH2 * COS2U;
    const auto UK = U - .25 * TEMP2 * X7THM1 * SIN2U;
    const auto NODEK = NODE + 1.5 * TEMP2 * COSI0 * SIN2U;
    const auto IK = I0 + 1.5 * TEMP2 * COSI0 * SINI0 * COS2U;
    const auto RDOTK = RDOT - N * TEMP1 * X1MTH2 * SIN2U;
    const auto RFDOTK = RFDOT + N * TEMP1 * (X1MTH2 * COS2U + 1.5 * X3THM1);

    // Orientation vectors.
    const auto SINUK = sin(UK);
    const auto COSUK = cos(UK);
    const auto SINIK = sin(IK);
    const auto COSIK = cos(IK);
    const auto SINNOK = sin(NODEK);
    const auto COSNOK = cos(NODEK);
    const auto MX = -SINNOK * COSIK;
    const auto MY = COSNOK * COSIK;
    const auto UX = MX * SINUK + COSNOK * COSUK;
    const auto UY = MY * SINUK + SINNOK * COSUK;
    const auto UZ = SINIK * SINUK;
    const auto VX = MX * COSUK - COSNOK * SINUK;
    const auto VY = MY * COSUK - SINNOK * SINUK;
    const auto VZ = SINIK * COSUK;

    // Position and velocity.
    const auto PV1 = RK * UX;
    const auto PV2 = RK * UY;
    const auto PV3 = RK * UZ;
    const auto PV4 = RDOTK * UX + RFDOTK * VX;
    const auto PV5 = RDOTK * UY + RFDOTK * VY;
    const auto PV6 = RDOTK * UZ + RFDOTK * VZ;

    // Rescaling factor for the Cartesian velocities.
    const auto vel_fac = KMPER / 60.;

    // Last error check: RK < 1.0.
    ERRORCODE = select(eq(ERRORCODE, 0.), select(lt(RK, 1.0), 6., 0.), ERRORCODE);

    return {PV1 * KMPER, PV2 * KMPER, PV3 * KMPER, PV4 * vel_fac, PV5 * vel_fac, PV6 * vel_fac, ERRORCODE};
}

} // namespace

// NOTE: the sgp4 algorithm is described in detail in this report:
//
// https://celestrak.org/NORAD/documentation/spacetrk.pdf
//
// The report contains the original fortran code, but this implementation
// is based on the "modern fortran" code from here:
//
// https://aim.hamptonu.edu/archive/cips/documentation/software/common/astron_lib/
//
// (which is easier to read because it avoids GOTOs).
//
// We also incorporate the updates to the original SGP4 algorithm from the official C++ code:
//
// https://celestrak.org/software/vallado-sw.php
//
// These involve a handful of checks for special cases (low eccentricity or
// inclination close to 180 degrees) that may cause numerical issues, and an updated
// way to "un-Kozai" the mean motion. These updates are noted in the comments.
//
// NOTE: 'inputs' is the set of expressions to be used as inputs for the sgp4 algorithm.
// These must represent, in order:
//
// - n0: the mean motion from the TLE (in [rad / min]),
// - e0: the eccentricity from the TLE,
// - i0: the inclination from the TLE (in [rad]),
// - node0: the right ascension of the ascending node from the TLE (in [rad]),
// - omega0: the argument of perigee from the TLE (in [rad]),
// - m0: the mean anomaly from the TLE (in [rad]),
// - bstar: the BSTAR drag term from the TLE (in the same unit as given in the TLE),
// - tsince: the time elapsed from the TLE epoch (in [min]).
//
// If 'inputs' is empty, variables with predefined names will be used as inputs.
std::vector<expression> sgp4(const std::vector<expression> &inputs)
{
    if (!inputs.empty() && inputs.size() != 8u) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid number of inputs passed to the sgp4() function: 8 "
                                                "expressions are expected but {} were provided instead",
                                                inputs.size()));
    }

    const auto init = sgp4_init(inputs);
    return sgp4_time_prop(inputs, init);
}

namespace detail
{

// NOTE: here we are going to build two vector-valued functions.
//
// The first one takes in input the original elements + bstar, and outputs:
// - the original elements (minus n0, which is not needed any more) + bstar,
// - the intermediate quantities returned by sgp4_init(), which are all functions
//   of the original elements + bstar,
// - the derivatives of the intermediate quantities wrt the diff arguments (if requested).
//
// The second function takes in input the outputs of the first function, and returns
// the propagated Cartesian state (with the tsince passed in as heyoka::time), and the
// derivatives of the Cartesian state with respect to the diff arguments (if requested).
sgp4_prop_funcs sgp4_build_funcs(std::uint32_t order)
{
    // Initialise the input quantities for the sgp4 algorithm.
    // NOTE: these are all the default ones, apart from tsince for which we use heyoka::time
    // instead of a 'tsince' variable.
    const auto sgp4_inputs = []() {
        const auto tmp = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar");

        // Add tsince as heyoka::time.
        std::vector<expression> retval(tmp.begin(), tmp.end());
        retval.push_back(heyoka::time);

        return retval;
    }();

    // The differentiation arguments. These are the sgp4 inputs minus tsince.
    // NOTE: if no derivatives are requested, this will remain unused.
    const auto diff_args = std::vector(sgp4_inputs.begin(), sgp4_inputs.end() - 1);

    // Compute the expressions for the intermediate quantities as functions of the orbital elements + bstar
    // (remember that tsince is ignored by sgp4_init()).
    const auto iqs_exprs = sgp4_init(sgp4_inputs);

    // Start assembling the init function's outputs. We begin with the original elements and bstar,
    // excluding n0, which will not be needed in the time propagation.
    // NOTE: we want to include (some of) the original sgp4 inputs in the output of the init
    // function because like this we can directly forward the init function outputs to the
    // time propagation function, which also needs (some of) the origina sgp4 inputs.
    std::vector func_init(sgp4_inputs.begin() + 1, sgp4_inputs.end() - 1);

    // Add the iqs and their derivatives, if requested.
    std::optional<dtens> diqs;
    if (order > 0u) {
        // Compute the dtens.
        diqs.emplace(diff_tensors(std::vector(iqs_exprs.begin(), iqs_exprs.end()), diff_args, kw::diff_order = order));

        // Append the iqs and their derivatives.
        const auto tv = *diqs | std::views::transform([](const auto &p) { return p.second; });
        func_init.insert(func_init.end(), tv.begin(), tv.end());
    } else {
        // Append the iqs without derivatives.
        func_init.insert(func_init.end(), iqs_exprs.begin(), iqs_exprs.end());
    }

    // Begin assembling the time propagation function and its arguments.
    std::vector<expression> func_tprop;
    auto func_tprop_args = std::vector(sgp4_inputs.begin() + 1, sgp4_inputs.end() - 1);

    // Variables representing the intermediate quantities.
    // NOTE: the names here are not important, as long as they are unique
    // and as long as they do not collide with the names used for the sgp4 inputs.
    const auto iqs_vars = make_vars("MDOT", "OMGDOT", "N0DOT", "NODCF", "C4", "C1", "T2COF", "MCOF", "ETA", "DELM0",
                                    "OMGCOF", "PERIGE", "C5", "SINM0", "D2", "D3", "D4", "T3COF", "T4COF", "T5COF",
                                    "A0DP", "AYCOF", "LCOF", "N0DP", "X3THM1", "X1MTH2", "X7THM1", "COSI0", "SINI0");

    // Add the cartesian state and its derivatives, if requested.
    std::optional<dtens> cart_dfun_dtens;
    if (order > 0u) {
        // In order to compute the derivatives, we need to transform the variables iqs_vars into dfuns
        // of the diff arguments.

        // Create the vector of arguments for use in the dfuns.
        const auto dfun_args = std::make_shared<std::vector<expression>>(diff_args);

        // Create the dfuns.
        std::array<expression, 29> iqs_dfuns;
        std::ranges::transform(iqs_vars, iqs_dfuns.begin(), [&dfun_args](const auto &ex) {
            return dfun(std::get<variable>(ex.value()).name(), dfun_args);
        });

        // Formulate the expressions for the Cartesian state in terms of the dfuns.
        const auto cart_dfun = sgp4_time_prop(sgp4_inputs, iqs_dfuns);

        // Compute the derivatives of cart_dfun wrt the diff arguments.
        cart_dfun_dtens.emplace(diff_tensors(cart_dfun, diff_args, kw::diff_order = order));

        // We now need to build a subs map to replace the dfuns in the expressions
        // stored in cart_dfun_dtens with variables.
        // NOTE: in order to do this, we iterate over diqs in order to fetch
        // the multiindices of the derivatives of the iqs wrt the diff arguments.
        std::map<expression, expression> dfun_subs_map;
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        for (const auto &[key, _] : *diqs) {
            const auto &[iq_idx, diff_idx] = key;

            assert(iq_idx < iqs_vars.size());

            // Fetch the name of the intermediate quantity we are operating on.
            const auto &iq_var_name = std::get<variable>(iqs_vars[iq_idx].value()).name();

            // Reconstruct its derivative as a dfun.
            auto df = dfun(iq_var_name, dfun_args, diff_idx);

            // Build the variable that will replace the dfun.
            // NOTE: this will be a unique name by construction: all the names in iqs_var
            // are unique, diff_idx is unique and we are at not risk of potential colliding
            // names being injected by bogus func diff implementations because we are only
            // using math functions builtin in heyoka.
            auto var = expression(fmt::format("∂{}{}", diff_idx, iq_var_name));

            // Append the variable as input argument for func_tprop.
            func_tprop_args.push_back(var);

            // Add the entry to the subs map.
            assert(!dfun_subs_map.contains(df));
            dfun_subs_map.emplace(std::move(df), std::move(var));
        }

        // Append the derivatives in cart_dfun to func_tprop.
        func_tprop.reserve(cart_dfun_dtens->size());
        std::ranges::transform(*cart_dfun_dtens, std::back_inserter(func_tprop),
                               [](const auto &p) { return p.second; });

        // Run the substitution.
        func_tprop = subs(func_tprop, dfun_subs_map);
    } else {
        // No derivatives requested, the function will contain only
        // the Cartesian state as a function of iqs_vars.
        func_tprop = sgp4_time_prop(sgp4_inputs, iqs_vars);
        func_tprop_args.insert(func_tprop_args.end(), iqs_vars.begin(), iqs_vars.end());
    }

    return {.init = {std::move(func_init), std::vector(sgp4_inputs.begin(), sgp4_inputs.end() - 1)},
            .tprop = {std::move(func_tprop), std::move(func_tprop_args)},
            .dt = std::move(cart_dfun_dtens)};
}

// Compile in parallel the init (f1) and tprop (f2) functions.
void sgp4_compile_funcs(const std::function<void()> &f1, const std::function<void()> &f2)
{
    oneapi::tbb::parallel_invoke(f1, f2);
}

} // namespace detail

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
struct sgp4_propagator<T>::impl {
    std::vector<T> m_sat_buffer;
    std::vector<T> m_init_buffer;
    cfunc<T> m_cf_tprop;
    cfunc<T> m_cf_init;
    std::optional<dtens> m_dtens;
    // NOTE: this is a buffer used to convert dates to tsinces in the call operator
    // overloads taking dates in input.
    std::vector<T> m_tms_vec;

    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_sat_buffer;
        ar & m_init_buffer;
        ar & m_cf_tprop;
        ar & m_cf_init;
        ar & m_dtens;
        // NOTE: no need to save the content of m_tms_vec.
    }
};

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_impl;
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T>::sgp4_propagator() noexcept = default;

// Detect deep-space sgp4 gpes. The units of measurement here are the
// usual ones:
//
// - n0: the mean motion from the TLE (in [rad / min]),
// - e0: the eccentricity from the TLE,
// - i0: the inclination from the TLE (in [rad]).
bool gpe_is_deep_space(double n0, double e0, double i0)
{
    using std::cos;
    using std::pow;
    using std::sqrt;

    // NOTE: this chunk of code is taken from the official C++ source.
    const auto cosio = cos(i0);
    const auto cosio2 = cosio * cosio;
    const auto eccsq = e0 * e0;
    const auto omeosq = 1 - eccsq;
    const auto rteosq = sqrt(omeosq);
    const auto d1 = 0.75 * J2 * (3 * cosio2 - 1) / (rteosq * omeosq);
    const auto ak = pow(KE / n0, 2. / 3);
    auto del = d1 / (ak * ak);
    const auto adel = ak * (1 - del * del - del * (1. / 3 + 134 * del * del / 81));
    del = d1 / (adel * adel);
    const auto no_unkozai = n0 / (1 + del);

    return 2 * boost::math::constants::pi<double>() / no_unkozai >= 225;
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T>::sgp4_propagator(ptag, std::tuple<std::vector<T>, cfunc<T>, cfunc<T>, std::optional<dtens>> tup)
{
    auto &[sat_buffer, cf_init, cf_tprop, dt] = tup;

    assert(sat_buffer.size() % 9u == 0u); // LCOV_EXCL_LINE
    const auto n_sats = sat_buffer.size() / 9u;

    // Prepare the init buffer - this will contain the values of the intermediate quantities
    // for all satellites, and their derivatives (if requested).
    std::vector<T> init_buffer;
    init_buffer.resize(boost::safe_numerics::safe<decltype(init_buffer.size())>(n_sats) * cf_init.get_nouts());

    // Prepare the in/out spans for the invocation of cf_init.
    // NOTE: for initialisation we only need to read the elements and the bstars from sat_buffer,
    // the epochs do not matter. Hence, 7 rows instead of 9.
    const typename cfunc<T>::in_2d init_input(sat_buffer.data(), 7, boost::numeric_cast<std::size_t>(n_sats));
    const typename cfunc<T>::out_2d init_output(init_buffer.data(),
                                                boost::numeric_cast<std::size_t>(cf_init.get_nouts()),
                                                boost::numeric_cast<std::size_t>(n_sats));

    // Evaluate the intermediate quantities and their derivatives.
    cf_init(init_output, init_input);

    // Build and assign the implementation.
    m_impl = std::make_unique<impl>(impl{
        std::move(sat_buffer), std::move(init_buffer), std::move(cf_tprop), std::move(cf_init), std::move(dt), {}});
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T>::sgp4_propagator(const sgp4_propagator &other) : m_impl(std::make_unique<impl>(*other.m_impl))
{
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T>::sgp4_propagator(sgp4_propagator &&) noexcept = default;

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T> &sgp4_propagator<T>::operator=(const sgp4_propagator &other)
{
    if (this != &other) {
        *this = sgp4_propagator(other);
    }

    return *this;
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T> &sgp4_propagator<T>::operator=(sgp4_propagator &&) noexcept = default;

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
sgp4_propagator<T>::~sgp4_propagator() = default;

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
std::uint32_t sgp4_propagator<T>::get_nsats() const
{
    return boost::numeric_cast<std::uint32_t>(m_impl->m_sat_buffer.size() / 9u);
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
std::uint32_t sgp4_propagator<T>::get_nouts() const noexcept
{
    return m_impl->m_cf_tprop.get_nouts();
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>> sgp4_propagator<T>::get_sat_data() const
{
    return mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>>{
        m_impl->m_sat_buffer.data(), boost::numeric_cast<std::size_t>(m_impl->m_sat_buffer.size() / 9u)};
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::replace_sat_data(mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>> new_data)
{
    // Cache nsats.
    const auto nsats = get_nsats();

    if (new_data.extent(1) != nsats) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid array provided to replace_sat_data(): the number of "
                                                "columns ({}) does not match the number of satellites ({})",
                                                new_data.extent(1), nsats));
    }

    // Fetch references to sat_buffer and init_buffer.
    auto &sat_buffer = m_impl->m_sat_buffer;
    auto &init_buffer = m_impl->m_init_buffer;

    // Write the new data into sat_buffer.
    const mdspan<T, extents<std::size_t, 9, std::dynamic_extent>> buffer_span(sat_buffer.data(), new_data.extent(1));
    for (std::size_t i = 0; i < buffer_span.extent(0); ++i) {
        for (std::size_t j = 0; j < buffer_span.extent(1); ++j) {
            buffer_span(i, j) = new_data(i, j);
        }
    }

    // Fetch a reference to cf_init.
    auto &cf_init = m_impl->m_cf_init;

    // Prepare the in/out spans for the invocation of cf_init.
    // NOTE: for initialisation we only need to read the elements and the bstars from sat_buffer,
    // the epochs do not matter. Hence, 7 rows instead of 9.
    // NOTE: static casts are ok, we already inited once during construction.
    const typename cfunc<T>::in_2d init_input(sat_buffer.data(), 7, static_cast<std::size_t>(nsats));
    const typename cfunc<T>::out_2d init_output(init_buffer.data(), static_cast<std::size_t>(cf_init.get_nouts()),
                                                static_cast<std::size_t>(nsats));

    // Evaluate the intermediate quantities and their derivatives.
    // NOTE: here is where we could potentially throw, as cf_init() may end up being
    // parallelised which could lead in principle to exceptions being thrown by
    // the TBB primitives.
    cf_init(init_output, init_input);
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::check_with_diff(const char *fname) const
{
    if (get_diff_order() == 0u) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}()' cannot be invoked on an sgp4 propagator without derivatives", fname));
    }
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
std::uint32_t sgp4_propagator<T>::get_diff_order() const noexcept
{
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return m_impl->m_dtens ? m_impl->m_dtens->get_order() : static_cast<std::uint32_t>(0);
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
const std::vector<expression> &sgp4_propagator<T>::get_diff_args() const
{
    check_with_diff(__func__);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return m_impl->m_dtens->get_args();
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
std::pair<std::uint32_t, std::uint32_t> sgp4_propagator<T>::get_dslice(std::uint32_t order) const
{
    check_with_diff(__func__);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &dt = *m_impl->m_dtens;

    const auto rng = dt.get_derivatives(order);

    return {boost::numeric_cast<std::uint32_t>(dt.index_of(rng.begin())),
            boost::numeric_cast<std::uint32_t>(dt.index_of(rng.end()))};
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
std::pair<std::uint32_t, std::uint32_t> sgp4_propagator<T>::get_dslice(std::uint32_t component,
                                                                       std::uint32_t order) const
{
    check_with_diff(__func__);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &dt = *m_impl->m_dtens;

    const auto rng = dt.get_derivatives(component, order);

    return {boost::numeric_cast<std::uint32_t>(dt.index_of(rng.begin())),
            boost::numeric_cast<std::uint32_t>(dt.index_of(rng.end()))};
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
const dtens::sv_idx_t &sgp4_propagator<T>::get_mindex(std::uint32_t i) const
{
    check_with_diff(__func__);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const auto &dt = *m_impl->m_dtens;

    if (i >= dt.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Cannot fetch the multiindex of the derivative at index {}: the index "
                                                "is not less than the total number of derivatives ({})",
                                                i, dt.size()));
    }

    return (dt.begin() + boost::numeric_cast<decltype(dt.begin() - dt.begin())>(i))->first;
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::operator()(out_2d out, in_1d<T> tms)
{
    const auto n_sats = get_nsats();
    assert(n_sats != 0u); // LCOV_EXCL_LINE

    // Prepare the init buffer span.
    assert(m_impl->m_init_buffer.size() % n_sats == 0u); // LCOV_EXCL_LINE
    const auto n_init_rows = boost::numeric_cast<std::size_t>(m_impl->m_init_buffer.size() / n_sats);
    const typename cfunc<T>::in_2d init_span(m_impl->m_init_buffer.data(), n_init_rows,
                                             boost::numeric_cast<std::size_t>(n_sats));

    // Run the propagation.
    m_impl->m_cf_tprop(out, init_span, kw::time = tms);
}

namespace detail
{

namespace
{

// Helper to convert a Julian date into a time delta suitable for use in the SGP4 algorithm.
// The reference epochs for all satellites are stored in sat_buffer, the dates are passed in
// via the 'dates' argument, n_sats is the total number of satellites and i is the index
// of the satellite on which we are operating. The double-length UTC Julian dates are first converted
// to TAI Julian dates using an erfa routine. The resulting TAI dates are then normalised and
// subtracted to yield the final time delta.
template <typename SizeType, typename Dates, typename T>
T sgp4_date_to_tdelta(SizeType i, Dates dates, const std::vector<T> &sat_buffer, std::uint32_t n_sats)
{
    using dfloat = heyoka::detail::dfloat<T>;

    // Convert the target propagation date to TAI using erfa.
    const auto [tai_prop_date_hi, tai_prop_date_lo] = jd_utc_to_tai(dates(i).jd, dates(i).frac);
    // Do the same for the satellite epoch.
    const auto [tai_epoch_hi, tai_epoch_lo] = jd_utc_to_tai(sat_buffer[static_cast<SizeType>(7) * n_sats + i],
                                                            sat_buffer[static_cast<SizeType>(8) * n_sats + i]);

    // Normalise the TAI propagation date into a double-length number.
    // NOTE: we use Knuth's EFT here in order to ensure correctness
    // even if the fractional part of the date is greater in magnitude
    // than the main part.
    const auto norm_prop_date
        = heyoka::detail::eft_add_knuth(static_cast<T>(tai_prop_date_hi), static_cast<T>(tai_prop_date_lo));
    const auto date = dfloat(norm_prop_date.first, norm_prop_date.second);

    // Do the same for the epoch.
    const auto norm_epoch = heyoka::detail::eft_add_knuth(static_cast<T>(tai_epoch_hi), static_cast<T>(tai_epoch_lo));
    const auto epoch = dfloat(norm_epoch.first, norm_epoch.second);

    // Compute the time delta in double-length arithmetic, truncate to a single-length
    // number and convert to minutes.
    return static_cast<T>(date - epoch) * 1440;
}

} // namespace

} // namespace detail

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::operator()(out_2d out, in_1d<date> dates)
{
    // Check the dates array.
    const auto n_sats = get_nsats();
    if (dates.extent(0) != n_sats) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid array of dates passed to the call operator of an sgp4_propagator: the number of "
                        "satellites is {}, while the number of dates is {}",
                        n_sats, dates.extent(0)));
    }

    // We need to convert the dates into time deltas suitable for use in the other call operator overload.

    // Prepare the memory buffer.
    auto &tms_vec = m_impl->m_tms_vec;
    using tms_vec_size_t = decltype(tms_vec.size());
    tms_vec.resize(boost::numeric_cast<tms_vec_size_t>(dates.extent(0)));

    // NOTE: we have a rough estimate of ~500 flops for a single execution of sgp4_date_to_tdelta().
    // Taking the usual figure of ~10'000 clock cycles as minimum threshold to parallelise, we enable
    // parallelisation if we have 20 satellites or more.
    if (n_sats >= 20u) {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<tms_vec_size_t>(0, tms_vec.size()),
                                  [&tms_vec, dates, this, n_sats](const auto &range) {
                                      for (auto i = range.begin(); i != range.end(); ++i) {
                                          tms_vec[i]
                                              = detail::sgp4_date_to_tdelta(i, dates, m_impl->m_sat_buffer, n_sats);
                                      }
                                  });
    } else {
        for (tms_vec_size_t i = 0; i < tms_vec.size(); ++i) {
            tms_vec[i] = detail::sgp4_date_to_tdelta(i, dates, m_impl->m_sat_buffer, n_sats);
        }
    }

    // Create the view.
    const in_1d<T> tms(tms_vec.data(), dates.extent(0));

    // Call the other overload.
    this->operator()(out, tms);
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::operator()(out_3d out, in_2d<T> tms)
{
    // Check the dimensionalities of out and tms.
    const auto n_evals = out.extent(0);
    if (n_evals != tms.extent(0)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid dimensions detected in batch-mode sgp4 propagation: the number of evaluations "
                        "inferred from the output array is {}, which is not consistent with the number of evaluations "
                        "inferred from the times array ({})",
                        n_evals, tms.extent(0)));
    }

    const auto n_sats = get_nsats();
    assert(n_sats != 0u); // LCOV_EXCL_LINE

    // Prepare the init buffer span.
    assert(m_impl->m_init_buffer.size() % n_sats == 0u); // LCOV_EXCL_LINE
    const auto n_init_rows = boost::numeric_cast<std::size_t>(m_impl->m_init_buffer.size() / n_sats);
    const typename cfunc<T>::in_2d init_span(m_impl->m_init_buffer.data(), n_init_rows,
                                             boost::numeric_cast<std::size_t>(n_sats));

    // NOTE: here we are unconditionally enabling parallel operations, even though in principle with very
    // few satellites or n_evals we could have some unnecessary overhead. Coming up with a cost model for when
    // to enable parallel operations is not easy, as I do not see an easy way of estimating the propagation
    // cost in the presence of derivatives. Probably it does not matter too much as a single sgp4 propagation
    // without derivatives is most likely already in the ~1000 flops range.

    // The functor to be run in the parallel loop below.
    auto par_iter = [out, tms, init_span](cfunc<T> &cf, const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
            // Create the spans for this iteration.
            const auto out_span
                = std::experimental::submdspan(out, i, std::experimental::full_extent, std::experimental::full_extent);
            const auto tm_span = std::experimental::submdspan(tms, i, std::experimental::full_extent);

            // Run the propagation.
            cf(out_span, init_span, kw::time = tm_span);
        }
    };

    // NOTE: in compact mode concurrent parallel operations on the same cfunc
    // object are not safe.
    if (m_impl->m_cf_tprop.get_compact_mode()) {
        // Construct the thread-specific storage for parallel operations.
        using ets_t = oneapi::tbb::enumerable_thread_specific<cfunc<T>, oneapi::tbb::cache_aligned_allocator<cfunc<T>>,
                                                              oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
        ets_t ets_cfunc([this]() { return m_impl->m_cf_tprop; });

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_evals),
                                  [&ets_cfunc, &par_iter](const auto &range) {
                                      // Fetch the thread-local cfunc.
                                      auto &cf = ets_cfunc.local();

                                      // NOTE: there are well-known pitfalls when using thread-specific
                                      // storage with nested parallelism:
                                      //
                                      // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                                      //
                                      // If the cfunc itself is performing parallel operations, then the current thread
                                      // will block as execution in the parallel region of the cfunc begins. The
                                      // blocked thread could then grab another task from the parallel for loop
                                      // we are currently in, and it would then start operating for a second time
                                      // on the same cf object.
                                      oneapi::tbb::this_task_arena::isolate([&]() { par_iter(cf, range); });
                                  });
    } else {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_evals),
                                  [&par_iter, this](const auto &range) { par_iter(m_impl->m_cf_tprop, range); });
    }
}

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
void sgp4_propagator<T>::operator()(out_3d out, in_2d<date> dates)
{
    // Check the dates array.
    const auto n_sats = get_nsats();
    if (dates.extent(1) != n_sats) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid array of dates passed to the batch-mode call operator of an sgp4_propagator: the number of "
            "satellites is {}, while the number of dates is per evaluation is {}",
            n_sats, dates.extent(1)));
    }

    // We need to convert the dates into time deltas suitable for use in the other call operator overload.

    // Prepare the memory buffer.
    const auto n_evals = dates.extent(0);
    auto &tms_vec = m_impl->m_tms_vec;
    using tms_vec_size_t = decltype(tms_vec.size());
    tms_vec.resize(boost::safe_numerics::safe<tms_vec_size_t>(n_evals) * dates.extent(1));

    // NOTE: we have a rough estimate of ~500 flops for a single execution of sgp4_date_to_tdelta().
    // Taking the usual figure of ~10'000 clock cycles as minimum threshold to parallelise, we enable
    // parallelisation if tms_vec.size() (i.e., n_evals * n_sats) is 20 or more.
    if (tms_vec.size() >= 20u) {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<std::size_t>(0, dates.extent(0), 0, dates.extent(1)),
            [&tms_vec, dates, this, n_sats](const auto &range) {
                for (auto i = range.rows().begin(); i != range.rows().end(); ++i) {
                    const auto cur_dates = std::experimental::submdspan(dates, i, std::experimental::full_extent);

                    for (auto j = range.cols().begin(); j != range.cols().end(); ++j) {
                        tms_vec[static_cast<tms_vec_size_t>(i) * n_sats + j] = detail::sgp4_date_to_tdelta(
                            static_cast<tms_vec_size_t>(j), cur_dates, m_impl->m_sat_buffer, n_sats);
                    }
                }
            });
    } else {
        for (std::size_t i = 0; i < dates.extent(0); ++i) {
            const auto cur_dates = std::experimental::submdspan(dates, i, std::experimental::full_extent);

            for (std::size_t j = 0; j < dates.extent(1); ++j) {
                tms_vec[static_cast<tms_vec_size_t>(i) * n_sats + j] = detail::sgp4_date_to_tdelta(
                    static_cast<tms_vec_size_t>(j), cur_dates, m_impl->m_sat_buffer, n_sats);
            }
        }
    }

    // Create the view.
    const in_2d<T> tms(tms_vec.data(), dates.extent(0), dates.extent(1));

    // Call the other overload.
    this->operator()(out, tms);
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC sgp4_propagator<float>;
template class HEYOKA_DLL_PUBLIC sgp4_propagator<double>;

std::pair<double, double> jd_utc_to_tai(double utc1, double utc2)
{
    // NOTE: the docs of the erfa routine indicate that we do not have
    // to normalise the input UTC double-length Julian dates.
    std::pair<double, double> retval;
    const auto res = eraUtctai(utc1, utc2, &retval.first, &retval.second);
    // LCOV_EXCL_START
    if (res == -1) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Cannot convert the invalid UTC Julian date ({}, {}) to TAI", utc1, utc2));
    }
    // LCOV_EXCL_STOP

    return retval;
}

std::pair<double, double> jd_tai_to_utc(double tai1, double tai2)
{
    // NOTE: the docs of the erfa routine indicate that we do not have
    // to normalise the input TAI double-length Julian dates.
    std::pair<double, double> retval;
    const auto res = eraTaiutc(tai1, tai2, &retval.first, &retval.second);
    // LCOV_EXCL_START
    if (res == -1) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Cannot convert the invalid TAI Julian date ({}, {}) to UTC", tai1, tai2));
    }
    // LCOV_EXCL_STOP

    return retval;
}

} // namespace model

HEYOKA_END_NAMESPACE
