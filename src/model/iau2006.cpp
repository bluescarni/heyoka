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
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/analytical_theories_helpers.hpp>
#include <heyoka/detail/iau2006/X.hpp>
#include <heyoka/detail/iau2006/Y.hpp>
#include <heyoka/detail/iau2006/s.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/iau2006.hpp>

// NOTE: this is an implementation of the IAU-2006/2000 analytical precession/nutation
// theory. The formulation is explained in detail in Vallado, 3.7.1, and in chapter 5 of the
// IERS conventions:
//
// https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf
//
// The datasets used in this implementation are available online at:
//
// https://iers-conventions.obspm.fr/content/chapter5/additional_info
//
// (files tab5.2a.txt, tab5.2b.txt and tab5.2d.txt).

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// Polynomial coefficients of the lunisolar arguments.
//
// NOTE: these are taken from https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf
// (section 5.7.2 and following) and double-checked against Vallado. The unit of measurement
// is *arcseconds*. These are all order-4 polynomials.
constexpr std::array arg_l = {485868.24903600005, 1717915923.2178, 31.8792, 0.051635, -0.00024470};
constexpr std::array arg_lp = {1287104.793048, 129596581.0481, -0.5532, 0.000136, -0.00001149};
constexpr std::array arg_F = {335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417};
constexpr std::array arg_D = {1072260.7036920001, 1602961601.2090, -6.3706, 0.006593, -0.00003169};
constexpr std::array arg_Om = {450160.39803599997, -6962890.5431, 7.4722, 0.007702, -0.00005939};

// Polynomial coefficients of the planetary arguments.
// NOTE: these values are in *radians* (yep, different from the lunisolar arguments).
constexpr std::array arg_L_Me = {4.402608842, 2608.7903141574};
constexpr std::array arg_L_Ve = {3.176146697, 1021.3285546211};
constexpr std::array arg_L_E = {1.753470314, 628.3075849991};
constexpr std::array arg_L_Ma = {6.203480913, 334.0612426700};
constexpr std::array arg_L_J = {0.599546497, 52.9690962641};
constexpr std::array arg_L_Sa = {0.874016757, 21.3299104960};
constexpr std::array arg_L_U = {5.481293872, 7.4781598567};
constexpr std::array arg_L_Ne = {5.311886287, 3.8133035638};
// NOTE: this is the general precession in longitude, the only one among
// these arguments with an order-2 polynomial expression.
constexpr std::array arg_p_A = {0., 0.02438175, 0.00000538691};

// Coefficients of the polynomial parts of the expressions for X, Y and s.
// The unit of measurement is *arcseconds* for X and Y, *microarcseconds* for s.
constexpr std::array poly_X = {-0.016617, 2004.191898, -0.4297829, -0.19861834, 0.000007578, 0.0000059285};
constexpr std::array poly_Y = {-0.006951, -0.025896, -22.4072747, 0.00190059, 0.001112526, 0.0000001358};
constexpr std::array poly_s = {94.0, 3808.65, -122.68, -72574.11, 27.98, 15.62};

} // namespace

std::array<expression, 3> iau2006_impl(const expression &tm, double thresh)
{
    using heyoka::detail::ccpow;
    using heyoka::detail::ex_cinv;
    using heyoka::detail::horner_eval;
    using heyoka::detail::pairwise_cmul;
    using heyoka::detail::trig_eval_dict_t;

    if (!std::isfinite(thresh) || thresh < 0.) {
        throw std::invalid_argument(fmt::format("Invalid threshold value passed to iau2006(): the value must be finite "
                                                "and non-negative, but it is {} instead",
                                                thresh));
    }

    // A couple of useful conversion factors.
    const auto arcsec2rad = boost::math::constants::pi<double>() / (180 * 3600.);
    const auto uas2rad = boost::math::constants::pi<double>() / (180 * 3600. * 1e6);

    // Evaluate the polynomial parts of X, Y and s.
    const auto poly_X_eval = horner_eval(poly_X, tm) * arcsec2rad;
    const auto poly_Y_eval = horner_eval(poly_Y, tm) * arcsec2rad;
    const auto poly_s_eval = horner_eval(poly_s, tm) * uas2rad;

    // Evaluate the lunisolar arguments.
    const auto l_eval = horner_eval(arg_l, tm) * arcsec2rad;
    const auto lp_eval = horner_eval(arg_lp, tm) * arcsec2rad;
    const auto F_eval = horner_eval(arg_F, tm) * arcsec2rad;
    const auto D_eval = horner_eval(arg_D, tm) * arcsec2rad;
    const auto Om_eval = horner_eval(arg_Om, tm) * arcsec2rad;

    // Evaluate the planetary arguments.
    // NOTE: these do not need unit conversion as they are already expressed in radians.
    const auto L_Me_eval = horner_eval(arg_L_Me, tm);
    const auto L_Ve_eval = horner_eval(arg_L_Ve, tm);
    const auto L_E_eval = horner_eval(arg_L_E, tm);
    const auto L_Ma_eval = horner_eval(arg_L_Ma, tm);
    const auto L_J_eval = horner_eval(arg_L_J, tm);
    const auto L_Sa_eval = horner_eval(arg_L_Sa, tm);
    const auto L_U_eval = horner_eval(arg_L_U, tm);
    const auto L_Ne_eval = horner_eval(arg_L_Ne, tm);
    const auto p_A_eval = horner_eval(arg_p_A, tm);

    // Store the evaluated arguments in an array for later use.
    const std::array<expression, 14> args = {l_eval,   lp_eval,   F_eval,   D_eval,    Om_eval,  L_Me_eval, L_Ve_eval,
                                             L_E_eval, L_Ma_eval, L_J_eval, L_Sa_eval, L_U_eval, L_Ne_eval, p_A_eval};

    // Seed the trig eval dictionary with powers of 0, 1 and -1 for each argument.
    trig_eval_dict_t trig_eval;
    for (const auto &arg : args) {
        const auto [it, flag] = trig_eval.insert({arg, {}});
        assert(flag);
        auto &pd = it->second;

        const std::array c_arg = {cos(arg), sin(arg)};

        pd[0] = {1_dbl, 0_dbl};
        pd[1] = c_arg;
        pd[-1] = ex_cinv(c_arg);
    }

    // Temporary accumulation list for complex products.
    std::vector<std::array<expression, 2>> tmp_cprod;

    // Helper to construct a trigonometric series from the coefficients and indices of the iau2006 solution.
    //
    // The iau2006 solution for X, Y and s is built from trigonometric series in which terms have the form
    //
    // C_s * sin(arg) + C_c * cos(arg),
    //
    // where C_s/C_c are numerical coefficients and arg is an integral linear combination of the arguments in
    // 'args', i.e.,
    //
    // arg = i0*l + i1*lp + i2*F + ...
    //
    // In order to avoid repeated calls to sin/cos, we compute instead (denoting with 'j' the imaginary unit)
    //
    // exp(j*arg) = exp(j*i0*l) * exp(j*i1*lp) * ...
    //
    // so that cos(arg) = Re[exp(j*arg)] and sin(arg) = Im[exp(j*arg)], while iteratively caching the integral
    // powers of the complex exponentials of the arguments. Like this, we compute sin/cos once per argument and
    // we transform the rest of the computation in a sequance of complex multiplications.
    //
    // In this function, out is the vector that will contain all the terms of the trigonometric series. cfs is the
    // dataset of sin/cos coefficients. idxs is the dataset of integral indices.
    const auto trig_builder
        = [&tmp_cprod, arcsec2rad, thresh, &args, &trig_eval](auto &out, const auto &cfs, const auto &idxs) {
              assert(out.empty());

              for (std::size_t i = 0; i < cfs.extent(0); ++i) {
                  // Fetch the sin/cos coefficients and convert them to arcseconds.
                  auto sin_cf = cfs(i, 0) / 1e6;
                  auto cos_cf = cfs(i, 1) / 1e6;

                  // Check if the term is too small.
                  if (std::sqrt((sin_cf * sin_cf) + (cos_cf * cos_cf)) < thresh) {
                      continue;
                  }

                  // Convert the coefficients to rad.
                  sin_cf *= arcsec2rad;
                  cos_cf *= arcsec2rad;

                  // Compute the complex powers of the arguments.
                  tmp_cprod.clear();
                  for (auto j = 0u; j < 14u; ++j) {
                      const auto cur_idx = idxs(i, j);
                      if (cur_idx != 0) {
                          tmp_cprod.push_back(ccpow(args[j], trig_eval, cur_idx));
                      }
                  }

                  // Assemble them.
                  const auto cprod = pairwise_cmul(tmp_cprod);

                  // Add the terms.
                  out.push_back(cos_cf * cprod[0]);
                  out.push_back(sin_cf * cprod[1]);
              }
          };

    // Build the Poisson series for X.
    std::vector<expression> X_terms_0, X_terms_1, X_terms_2, X_terms_3, X_terms_4;
    trig_builder(X_terms_0, iau2006_X_cfs_0, iau2006_X_args_idxs_0);
    trig_builder(X_terms_1, iau2006_X_cfs_1, iau2006_X_args_idxs_1);
    trig_builder(X_terms_2, iau2006_X_cfs_2, iau2006_X_args_idxs_2);
    trig_builder(X_terms_3, iau2006_X_cfs_3, iau2006_X_args_idxs_3);
    trig_builder(X_terms_4, iau2006_X_cfs_4, iau2006_X_args_idxs_4);

    // Build the Poisson series for Y.
    std::vector<expression> Y_terms_0, Y_terms_1, Y_terms_2, Y_terms_3, Y_terms_4;
    trig_builder(Y_terms_0, iau2006_Y_cfs_0, iau2006_Y_args_idxs_0);
    trig_builder(Y_terms_1, iau2006_Y_cfs_1, iau2006_Y_args_idxs_1);
    trig_builder(Y_terms_2, iau2006_Y_cfs_2, iau2006_Y_args_idxs_2);
    trig_builder(Y_terms_3, iau2006_Y_cfs_3, iau2006_Y_args_idxs_3);
    trig_builder(Y_terms_4, iau2006_Y_cfs_4, iau2006_Y_args_idxs_4);

    // Build the Poisson series for s.
    std::vector<expression> s_terms_0, s_terms_1, s_terms_2, s_terms_3, s_terms_4;
    trig_builder(s_terms_0, iau2006_s_cfs_0, iau2006_s_args_idxs_0);
    trig_builder(s_terms_1, iau2006_s_cfs_1, iau2006_s_args_idxs_1);
    trig_builder(s_terms_2, iau2006_s_cfs_2, iau2006_s_args_idxs_2);
    trig_builder(s_terms_3, iau2006_s_cfs_3, iau2006_s_args_idxs_3);
    trig_builder(s_terms_4, iau2006_s_cfs_4, iau2006_s_args_idxs_4);

    // Sum the Poisson series, multiply them by powers of the time and sum again.
    const auto X_trig
        = horner_eval(std::array{sum(X_terms_0), sum(X_terms_1), sum(X_terms_2), sum(X_terms_3), sum(X_terms_4)}, tm);
    const auto Y_trig
        = horner_eval(std::array{sum(Y_terms_0), sum(Y_terms_1), sum(Y_terms_2), sum(Y_terms_3), sum(Y_terms_4)}, tm);
    const auto s_trig
        = horner_eval(std::array{sum(s_terms_0), sum(s_terms_1), sum(s_terms_2), sum(s_terms_3), sum(s_terms_4)}, tm);

    // Assemble the return values.
    auto X = poly_X_eval + X_trig;
    auto Y = poly_Y_eval + Y_trig;
    auto s = sum({poly_s_eval, s_trig, -(X * Y) / 2.});

    return {X, Y, s};
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
