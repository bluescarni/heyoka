// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/model/sgp4.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

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
// Several numerical evaluations of this model have been compared to the
// results from the Python sgp4 implementation:
//
// https://pypi.org/project/sgp4/
//
// Agreement seems ok, with a positional error which starts at <1mm for TSINCE=0,
// somehow increasing to the 1cm level after 1 day of propagation (at least for some test
// cases). This is far less than the expected 1-3km/day error by which the satellites
// deviate from the ideal orbits described in TLE files. In any case, in the future and
// if needed, we can always refer to the "official" code on celestrak (on which the Python
// sgp4 module is based):
//
// https://celestrak.org/software/vallado-sw.php
std::pair<std::vector<expression>, std::vector<expression>> sgp4()
{
    // Several math wrappers used in the original fortran code.
    // Yay caps!
    constexpr auto ABS = [](const auto &x) { return select(gte(x, 0.), x, -x); };

    constexpr auto ACTAN = [](const auto &a, const auto &b) {
        const auto ret = atan2(a, b);
        return select(gte(ret, 0.), ret, 2. * heyoka::pi + ret);
    };

    constexpr auto MAX = [](const auto &a, const auto &b) { return select(gt(a, b), a, b); };

    constexpr auto MIN = [](const auto &a, const auto &b) { return select(lt(a, b), a, b); };

    // Constants.
    const auto KE = 0.743669161e-1;
    const auto TOTHRD = 2 / 3.;
    const auto J2 = 1.082616e-3;
    const auto CK2 = .5 * J2;
    const auto KMPER = 6378.135;
    const auto S0 = 20. / KMPER;
    const auto S1 = 78. / KMPER;
    const auto Q0 = 120. / KMPER;
    const auto J3 = -0.253881e-5;
    const auto A3OVK2 = -J3 / CK2;
    // NOTE: not sure if this is a mistake in the original code
    // or what else, but this constant (and this constant only)
    // was defined in single precision, rather than double.
    // Defining this constant directly in double precision
    // (rather than going through the cast) results
    // in subtle differences in several intermediate quantities computed
    // in the implementation. It probably does not matter too much in the
    // end as this is just the J4 coefficient and its overall effect on
    // the dynamics is miniscule, but still... why?
    const auto J4 = static_cast<double>(-1.65597e-6f);
    const auto CK4 = -.375 * J4;
    const auto SIMPHT = 220. / KMPER;

    // The inputs.
    const auto [N0, I0, E0, BSTAR, OMEGA0, M0, TSINCE, NODE0]
        = make_vars("n0", "i0", "e0", "bstar", "omega0", "m0", "tsince", "node0");

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
    const auto A0DP = A0 / (1. - DEL0);
    const auto PERIGE = A0DP * (1. - E0) - 1.;
    const auto S = MIN(MAX(S0, PERIGE - S1), S1);
    const auto S4 = 1. + S;
    const auto PINVSQ = 1. / pow(A0DP * BETA02, 2.);
    const auto XI = 1. / (A0DP - S4);
    const auto ETA = A0DP * XI * E0;
    const auto ETASQ = pow(ETA, 2.);
    const auto EETA = E0 * ETA;
    const auto PSISQ = ABS(1. - ETASQ);
    const auto COEF = pow((Q0 - S) * XI, 4.);
    const auto COEF1 = COEF / (sqrt(PSISQ) * pow(PSISQ, 3.));
    const auto C1 = BSTAR * COEF1 * N0DP
                    * (A0DP * (1. + 1.5 * ETASQ + EETA * (4. + ETASQ))
                       + 0.75 * CK2 * XI / PSISQ * X3THM1 * (8. + 3. * ETASQ * (8. + ETASQ)));

    const auto SINI0 = sin(I0);
    const auto C3 = COEF * XI * A3OVK2 * N0DP * SINI0 / E0;
    const auto X1MTH2 = 1. - THETA2;
    const auto C4 = 2. * N0DP * COEF1 * A0DP * BETA02
                    * (ETA * (2. + .5 * ETASQ) + E0 * (.5 + 2. * ETASQ)
                       - 2. * CK2 * XI / (A0DP * PSISQ)
                             * (-3. * X3THM1 * (1. - 2. * EETA + ETASQ * (1.5 - .5 * EETA))
                                + .75 * X1MTH2 * (2. * ETASQ - EETA * (1. + ETASQ)) * cos(2. * OMEGA0)));
    const auto C5 = 2. * COEF1 * A0DP * BETA02 * (1. + 2.75 * (ETASQ + EETA) + EETA * ETASQ);
    const auto THETA4 = pow(THETA2, 2.);
    auto TEMP1 = 3. * CK2 * PINVSQ * N0DP;
    auto TEMP2 = TEMP1 * CK2 * PINVSQ;
    auto TEMP3 = 1.25 * CK4 * pow(PINVSQ, 2.) * N0DP;
    const auto MDOT = N0DP + .5 * TEMP1 * BETA0 * X3THM1 + .0625 * TEMP2 * BETA0 * (13. - 78. * THETA2 + 137. * THETA4);
    const auto OMGDOT = -.5 * TEMP1 * (1. - 5. * THETA2) + 0.0625 * TEMP2 * (7. - 114. * THETA2 + 395. * THETA4)
                        + TEMP3 * (3. - 36. * THETA2 + 49. * THETA4);
    const auto HDOT1 = -TEMP1 * COSI0;
    const auto N0DOT = HDOT1 + (.5 * TEMP2 * (4. - 19. * THETA2) + 2. * TEMP3 * (3. - 7. * THETA2)) * COSI0;
    const auto OMGCOF = BSTAR * C3 * cos(OMEGA0);
    const auto MCOF = -TOTHRD * COEF * BSTAR / EETA;
    const auto NODCF = 3.5 * BETA02 * HDOT1 * C1;
    const auto T2COF = 1.5 * C1;
    const auto LCOF = .125 * A3OVK2 * SINI0 * (3. + 5. * COSI0) / (1. + COSI0);
    const auto AYCOF = .25 * A3OVK2 * SINI0;
    const auto DELM0 = pow(1. + ETA * cos(M0), 3.);
    const auto SINM0 = sin(M0);
    const auto X7THM1 = 7. * THETA2 - 1.;

    // For perigee less than 220 kilometers, the equations are
    // truncated to linear variation in sqrt A and quadratic
    // variation in mean anomaly.  Also, the C3 term, the
    // delta OMEGA term, and the delta M term are dropped.
    const auto C1SQ = pow(C1, 2.);
    const auto D2 = 4. * A0DP * XI * C1SQ;
    const auto TEMP0 = D2 * XI * C1 / 3.;
    const auto D3 = (17. * A0DP + S4) * TEMP0;
    const auto D4 = .5 * TEMP0 * A0DP * XI * (221. * A0DP + 31. * S4) * C1;
    const auto T3COF = D2 + 2. * C1SQ;
    const auto T4COF = .25 * (3. * D3 + C1 * (12. * D2 + 10. * C1SQ));
    const auto T5COF = .2 * (3. * D4 + 12. * C1 * D3 + 6. * pow(D2, 2.) + 15. * C1SQ * (2. * D2 + C1SQ));

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
    const auto E = E0 - TEMPE * BSTAR;
    TEMPL = TEMPL * pow(TSINCE, 2.);

    // Long period periodics.
    const auto AXN = E * cos(OMEGA);
    const auto AB = A * (1. - pow(E, 2.));
    const auto AYN = AYCOF / AB + E * sin(OMEGA);

    // Solve Kepler's equation.
    // NOTE: the original report (on page 13) says that this step is about solving
    // Kepler's equations for E + omega. This is a quantity similar to the eccentric
    // longitude F = F(h, k, lambda), implemented in heyoka as kepF():
    //
    // https://articles.adsabs.harvard.edu//full/1972CeMec...5..303B/0000309.000.html
    //
    // Indeed, the numerical iteration proposed in the report is nothing but a
    // Newton-Raphson step for the computation of F(h=AYN, k=AXN, lambda=CAPU). Thus, we avoid
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
    const auto R = A * (1. - ECOSE);
    const auto RDOT = KE * sqrt(A) * ESINE / R;
    const auto RFDOT = KE * sqrt(PL) / R;
    const auto BETAL = sqrt(TEMPS);
    TEMP3 = ESINE / (1. + BETAL);
    const auto COSU = (COSEPW - AXN + AYN * TEMP3) * A / R;
    const auto SINU = (SINEPW - AYN - AXN * TEMP3) * A / R;
    const auto U = ACTAN(SINU, COSU);
    const auto SIN2U = 2. * SINU * COSU;
    const auto COS2U = 2. * pow(COSU, 2.) - 1.;
    TEMP1 = CK2 / PL;
    TEMP2 = TEMP1 / PL;

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

    return {{N0, E0, I0, NODE0, OMEGA0, M0, BSTAR, TSINCE},
            {PV1 * KMPER, PV2 * KMPER, PV3 * KMPER, PV4 * (KMPER / 60.), PV5 * (KMPER / 60.), PV6 * (KMPER / 60.)}};
}

} // namespace model

HEYOKA_END_NAMESPACE
