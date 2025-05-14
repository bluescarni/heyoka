// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/core.h>

#include <oneapi/tbb/parallel_invoke.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/analytical_theories_helpers.hpp>
#include <heyoka/detail/elp2000/elp2000_10_15.hpp>
#include <heyoka/detail/elp2000/elp2000_16_21.hpp>
#include <heyoka/detail/elp2000/elp2000_1_3.hpp>
#include <heyoka/detail/elp2000/elp2000_22_36.hpp>
#include <heyoka/detail/elp2000/elp2000_4_9.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/elp2000.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

namespace
{

// Polynomial coefficients of the time-dependent arguments.

// Mean mean longitude.
constexpr std::array W1 = {3.8103444305883079, 8399.6847317739157, -2.8547283984772807e-05, 3.2017095500473753e-08,
                           -1.5363745554361197e-10};

// NOTE: this is the linear part of W1 plus the precession constant.
constexpr std::array zeta = {W1[0], W1[1] + 0.024381748353014515};

// Delaunay arguments and their linear versions.
constexpr std::array D = {5.1984667410274437, 7771.3771468120494, -2.8449351621188683e-05, 3.1973462269173901e-08,
                          -1.5436467606527627e-10};
constexpr std::array D_lin = {D[0], D[1]};

constexpr std::array lp = {6.2400601269714615, 628.30195516800313, -2.680534842854624e-06, 7.1267611123101784e-10};
constexpr std::array lp_lin = {lp[0], lp[1]};

constexpr std::array l
    = {2.3555558982657985, 8328.6914269553617, 0.00015702775761561094, 2.5041111442988642e-07, -1.1863390776750345e-09};
constexpr std::array l_lin = {l[0], l[1]};

constexpr std::array F
    = {1.6279052333714679, 8433.4661581308319, -5.9392100004323707e-05, -4.9499476841283623e-09, 2.021673050226765e-11};
constexpr std::array F_lin = {F[0], F[1]};

// Planetary longitudes and mean motions.
constexpr std::array Me = {4.4026088424029615, 2608.7903141574106};
constexpr std::array V_ = {3.1761466969075944, 1021.3285546211089};
constexpr std::array Ma = {6.2034809133999449, 334.06124314922965};
constexpr std::array J = {0.59954649738867349, 52.969096509472053};
constexpr std::array S = {0.87401675651848076, 21.329909543800007};
constexpr std::array U_ = {5.4812938716049908, 7.4781598567143535};
constexpr std::array N = {5.3118862867834666, 3.8133035637584562};

// Mean heliocentric mean longitude of the Earth-Moon barycenter
// NOTE: this is needed only in the linear formulation to compute
// the planetary perturbations.
constexpr std::array T = {1.753470343150658, 628.30758496215537};

} // namespace

// NOTE: don't check for coverage here as in order to hit all branches
// we would need very low threshold level, which is not feasible
// in debug mode.
// LCOV_EXCL_START

// Spherical coordinates, inertial mean ecliptic of date.
// NOLINTNEXTLINE(readability-function-size,hicpp-function-size,google-readability-function-size)
std::vector<expression> elp2000_spherical_impl(const expression &tm, double thresh)
{
    using heyoka::detail::ccpow;
    using heyoka::detail::ex_cinv;
    using heyoka::detail::horner_eval;
    using heyoka::detail::pairwise_cmul;
    using heyoka::detail::trig_eval_dict_t;
    using heyoka::detail::uncvref_t;

    if (!std::isfinite(thresh) || thresh < 0.) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid threshold value passed to elp2000_spherical(): "
                                                "the value must be finite and non-negative, but it is {} instead",
                                                thresh));
    }

    // Evaluate the arguments.
    const auto W1_eval = horner_eval(W1, tm);
    const auto zeta_eval = horner_eval(zeta, tm);
    const auto D_eval = horner_eval(D, tm);
    const auto D_lin_eval = horner_eval(D_lin, tm);
    const auto lp_eval = horner_eval(lp, tm);
    const auto lp_lin_eval = horner_eval(lp_lin, tm);
    const auto l_eval = horner_eval(l, tm);
    const auto l_lin_eval = horner_eval(l_lin, tm);
    const auto F_eval = horner_eval(F, tm);
    const auto F_lin_eval = horner_eval(F_lin, tm);
    const auto Me_eval = horner_eval(Me, tm);
    const auto V_eval = horner_eval(V_, tm);
    const auto Ma_eval = horner_eval(Ma, tm);
    const auto J_eval = horner_eval(J, tm);
    const auto S_eval = horner_eval(S, tm);
    const auto U_eval = horner_eval(U_, tm);
    const auto N_eval = horner_eval(N, tm);
    const auto T_eval = horner_eval(T, tm);

    // Seed the trig eval dictionary with powers of 0, 1 and -1 for each
    // trigonometric argument.
    trig_eval_dict_t trig_eval;

    auto seed_trig_eval = [&trig_eval](const expression &arg) {
        const auto [it, flag] = trig_eval.insert({arg, {}});
        assert(flag);
        auto &pd = it->second;

        const std::array c_arg = {cos(arg), sin(arg)};

        pd[0] = {1_dbl, 0_dbl};
        pd[1] = c_arg;
        pd[-1] = ex_cinv(c_arg);
    };

    seed_trig_eval(zeta_eval);
    seed_trig_eval(D_eval);
    seed_trig_eval(D_lin_eval);
    seed_trig_eval(lp_eval);
    seed_trig_eval(lp_lin_eval);
    seed_trig_eval(l_eval);
    seed_trig_eval(l_lin_eval);
    seed_trig_eval(F_eval);
    seed_trig_eval(F_lin_eval);
    seed_trig_eval(Me_eval);
    seed_trig_eval(V_eval);
    seed_trig_eval(Ma_eval);
    seed_trig_eval(J_eval);
    seed_trig_eval(S_eval);
    seed_trig_eval(U_eval);
    seed_trig_eval(N_eval);
    seed_trig_eval(T_eval);

    // Temporary accumulation list for complex products.
    std::vector<std::array<expression, 2>> tmp_cprod;

    // Several constants used in the corrections to the A coefficients
    // for the main problem. See section 7 in the PDF.
    // NOTE: these are all kept in the original units of measures,
    // will be converting on-the-fly as needed after the computation of
    // the correction.
    // NOTE: in some of these numbers there seem to be slight differences
    // or inconsistencies between the PDF and the Fortran code. We use the
    // values from the PDF.
    const auto a0 = 384747.980674;
    const auto nu = 1732559343.18;
    const auto np = 129597742.34;
    const auto m = np / nu;
    const auto dnu = 0.55604;
    const auto dnp = -0.0642;
    const auto alpha = std::cbrt(m * m * 3.040423956e-6);
    const auto alpha2_m3 = (2 * alpha) / (3 * m);
    const auto B15_fac = (dnp - m * dnu) / nu;
    const auto B2_fac = -0.08066 / 206264.81;
    const auto B3_fac = 0.01789 / 206264.81;
    const auto B4_fac = -0.12879 / 206264.81;
    const auto arcsec = 4.8481368110953598e-06;

    // Longitude.
    std::vector<expression> V_terms{W1_eval}, V_terms_t1, V_terms_t2;

    // ELP1.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_1)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_1); ++i) {
            const auto &[cur_A, B1, B2, B3, B4, B5] = elp2000_A_B_1[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_1[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                // Compute the correction to A.
                auto corr = ((B1 + B5 * alpha2_m3) * B15_fac) + (B2_fac * B2) + (B3_fac * B3) + (B4_fac * B4);
                corr *= arcsec;

                V_terms.push_back((cur_A + corr) * cprod[1]);
            }
        }
    }

    // ELP4.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_4)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_4); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_4[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_4[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP7.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_7)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_7); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_7[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_7[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP10.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_10)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_10); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_10[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_10[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP13.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_13)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_13); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_13[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_13[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP16.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_16)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_16); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_16[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_16[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP19.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_19)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_19); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_19[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_19[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP22.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_22)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_22); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_22[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_22[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP25.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_25)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_25); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_25[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_25[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP28.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_28)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_28); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_28[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_28[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP31.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_31)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_31); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_31[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_31[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP34.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_34)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_34); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_34[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_34[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                V_terms_t2.push_back(cur_A * cprod[1]);
            }
        }
    }

    // Latitude.
    std::vector<expression> U_terms, U_terms_t1, U_terms_t2;

    // ELP2.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_2)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_2); ++i) {
            const auto &[cur_A, B1, B2, B3, B4, B5] = elp2000_A_B_2[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_2[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                // Compute the correction to A.
                auto corr = ((B1 + B5 * alpha2_m3) * B15_fac) + (B2_fac * B2) + (B3_fac * B3) + (B4_fac * B4);
                corr *= arcsec;

                U_terms.push_back((cur_A + corr) * cprod[1]);
            }
        }
    }

    // ELP5.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_5)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_5); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_5[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_5[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP8.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_8)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_8); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_8[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_8[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP11.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_11)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_11); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_11[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_11[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP14.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_14)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_14); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_14[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_14[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP17.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_17)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_17); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_17[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_17[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP20.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_20)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_20); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_20[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_20[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP23.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_23)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_23); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_23[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_23[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP26.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_26)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_26); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_26[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_26[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP29.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_29)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_29); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_29[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_29[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP32.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_32)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_32); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_32[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_32[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP35.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_35)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_35); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_35[i];

            if (std::abs(cur_A) > thresh) {
                const auto &cur_idx_v = elp2000_idx_35[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                U_terms_t2.push_back(cur_A * cprod[1]);
            }
        }
    }

    // Distance.
    std::vector<expression> r_terms, r_terms_t1, r_terms_t2;

    // ELP3.
    {
        const std::array args = {D_eval, lp_eval, l_eval, F_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_3)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_3); ++i) {
            const auto &[cur_A, B1, B2, B3, B4, B5] = elp2000_A_B_3[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_3[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                // Compute the correction to A.
                auto corr = ((B1 + B5 * alpha2_m3) * B15_fac) + (B2_fac * B2) + (B3_fac * B3) + (B4_fac * B4);
                corr -= 2 * cur_A * dnu / (3 * nu);

                r_terms.push_back((cur_A + corr) * cprod[0]);
            }
        }
    }

    // ELP6.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_6)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_6); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_6[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_6[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP9.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_9)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_9); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_9[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_9[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP12.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_12)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_12); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_12[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_12[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP15.
    {
        const std::array args
            = {Me_eval, V_eval, T_eval, Ma_eval, J_eval, S_eval, U_eval, N_eval, D_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_15)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_15); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_15[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_15[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP18.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_18)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_18); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_18[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_18[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP21.
    {
        const std::array args = {Me_eval, V_eval,     T_eval,      Ma_eval,    J_eval,    S_eval,
                                 U_eval,  D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_21)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_21); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_21[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_21[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP24.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_24)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_24); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_24[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_24[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP27.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_27)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_27); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_27[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_27[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms_t1.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP30.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_30)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_30); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_30[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_30[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP33.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_33)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_33); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_33[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_33[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms.push_back(cur_A * cprod[1]);
            }
        }
    }

    // ELP36.
    {
        const std::array args = {zeta_eval, D_lin_eval, lp_lin_eval, l_lin_eval, F_lin_eval};
        static_assert(std::extent_v<uncvref_t<decltype(elp2000_idx_36)>, 1> == std::tuple_size_v<decltype(args)>);

        for (std::size_t i = 0; i < std::size(elp2000_idx_36); ++i) {
            const auto &[cur_phi, cur_A] = elp2000_phi_A_36[i];

            if (std::abs(cur_A / a0) > thresh) {
                const auto &cur_idx_v = elp2000_idx_36[i];
                tmp_cprod.clear();

                for (std::size_t j = 0; j < std::size(cur_idx_v); ++j) {
                    if (cur_idx_v[j] != 0) {
                        tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx_v[j]));
                    }
                }

                if (cur_phi != 0.) {
                    tmp_cprod.push_back({expression{std::cos(cur_phi)}, expression{std::sin(cur_phi)}});
                }

                auto cprod = pairwise_cmul(tmp_cprod);

                r_terms_t2.push_back(cur_A * cprod[1]);
            }
        }
    }

    std::vector<expression> retval;
    retval.resize(3);

    oneapi::tbb::parallel_invoke(
        [&]() {
            expression a, b, c;
            oneapi::tbb::parallel_invoke([&]() { a = sum(r_terms); }, [&]() { b = sum(r_terms_t1); },
                                         [&]() { c = sum(r_terms_t2); });
            retval[0] = horner_eval({a, b, c}, tm);
        },
        [&]() {
            expression a, b, c;
            oneapi::tbb::parallel_invoke([&]() { a = sum(U_terms); }, [&]() { b = sum(U_terms_t1); },
                                         [&]() { c = sum(U_terms_t2); });
            retval[1] = horner_eval({a, b, c}, tm);
        },
        [&]() {
            expression a, b, c;
            oneapi::tbb::parallel_invoke([&]() { a = sum(V_terms); }, [&]() { b = sum(V_terms_t1); },
                                         [&]() { c = sum(V_terms_t2); });
            retval[2] = horner_eval({a, b, c}, tm);
        });

    return retval;
}

// LCOV_EXCL_STOP

// Cartesian coordinates, inertial mean ecliptic of date.
std::vector<expression> elp2000_cartesian_impl(const expression &tm, double thresh)
{
    const auto sph = elp2000_spherical_impl(tm, thresh);

    const auto &r = sph[0];
    const auto &U = sph[1];
    const auto &V = sph[2];

    const auto cU = cos(U);
    const auto sU = sin(U);

    const auto cV = cos(V);
    const auto sV = sin(V);

    const auto rcU = r * cU;

    return {rcU * cV, rcU * sV, r * sU};
}

namespace
{

// Laskar's P and Q series' coefficients.
constexpr std::array LP = {0., 0.10180391e-4, 0.47020439e-6, -0.5417367e-9, -0.2507948e-11, 0.463486e-14};
constexpr std::array LQ = {0., -0.113469002e-3, 0.12372674e-6, 0.12654170e-8, -0.1371808e-11, -0.320334e-14};

} // namespace

// Cartesian coordinates, inertial mean ecliptic and equinox of J2000.
std::vector<expression> elp2000_cartesian_e2000_impl(const expression &tm, double thresh)
{
    namespace hd = heyoka::detail;

    const auto cart = elp2000_cartesian_impl(tm, thresh);

    const auto &x = cart[0];
    const auto &y = cart[1];
    const auto &z = cart[2];

    // Evaluate Laskar's series.
    const auto P = hd::horner_eval(LP, tm);
    const auto Q = hd::horner_eval(LQ, tm);

    const auto P2 = pow(P, 2_dbl);
    const auto Q2 = pow(Q, 2_dbl);
    const auto PQ = P * Q;
    const auto sqrP2Q2 = sqrt(1. - P2 - Q2);

    const auto xe2000 = sum({(1. - 2. * P2) * x, 2. * PQ * y, 2. * P * sqrP2Q2 * z});
    const auto ye2000 = sum({2. * PQ * x, (1. - 2. * Q2) * y, -2. * Q * sqrP2Q2 * z});
    const auto ze2000 = sum({-2. * P * sqrP2Q2 * x, 2. * Q * sqrP2Q2 * y, (1. - 2. * P2 - 2. * Q2) * z});

    return {xe2000, ye2000, ze2000};
}

// Cartesian coordinates, FK5 (i.e., mean equator and rotational mean equinox of J2000).
std::vector<expression> elp2000_cartesian_fk5_impl(const expression &tm, double thresh)
{
    const auto cart_e2000 = elp2000_cartesian_e2000_impl(tm, thresh);

    const auto &xe2000 = cart_e2000[0];
    const auto &ye2000 = cart_e2000[1];
    const auto &ze2000 = cart_e2000[2];

    const auto xq2000 = sum({xe2000, 0.000000437913 * ye2000, -0.000000189859 * ze2000});
    const auto yq2000 = sum({-0.000000477299 * xe2000, 0.917482137607 * ye2000, -0.397776981701 * ze2000});
    const auto zq2000 = sum({0.397776981701 * ye2000, 0.917482137607 * ze2000});

    return {xq2000, yq2000, zq2000};
}

} // namespace detail

std::array<double, 2> get_elp2000_mus()
{
    return {3.986005e14, 4902794214578.239};
}

} // namespace model

HEYOKA_END_NAMESPACE
