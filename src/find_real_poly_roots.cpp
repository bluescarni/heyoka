// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <span>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/find_real_poly_roots.hpp>

HEYOKA_BEGIN_NAMESPACE

// Find the real roots of a polynomial within an interval.
//
// The polynomial p(t) is given by its coefficients cfs in the monomial basis, in ascending degree order, so that the
// polynomial degree is cfs.size() - 1. This function stores in ret the real roots of p(t) for t in [0, h) if h>0, and
// for t in (h, 0] if h<0. If h==0, ret will be empty.
//
// Every real root of p(t) in this range is reported. Roots that cannot be resolved at the working precision (e.g.,
// multiple roots, or roots closer to each other than the precision allows) may be reported as a single root, and such a
// report may be a false positive.
//
// lb is a non-negative value in the same units as h. Roots in the following ranges are not reported:
//
// - t in [0, lb) if h>0;
// - t in (-lb, 0] if h<0.
//
// ws must have been constructed for the degree of p. cfs and h must be finite, and lb must be non-NaN and non-negative.
// For mppp::real, cfs and h must have the precision of ws, and lb is rounded to it.
template <typename T>
void find_real_poly_roots(std::vector<real_poly_root<T>> &ret, frpr_workspace<T> &ws, const std::span<const T> cfs,
                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                          const T &h, T lb)
{
    using std::abs;
    using std::isfinite;
    using std::isnan;

    // Small helper to reduce typing below.
    constexpr auto is_finite = [](const auto &x) { return isfinite(x); };

    // Validate the input arguments.
    if (cfs.empty()) [[unlikely]] {
        throw std::invalid_argument(
            "Cannot invoke find_real_poly_roots() with an empty list of polynomial coefficients");
    }

    if (ws.m_impl->get_degree() != cfs.size() - 1u) [[unlikely]] {
        throw std::invalid_argument(
            // LCOV_EXCL_START
            fmt::format("find_real_poly_roots() was invoked with inconsistent arguments: the polynomial "
                        "degree from the workspace is {}, while the polynomial degree inferred from the list of "
                        "coefficients is {}",
                        // LCOV_EXCL_STOP
                        ws.m_impl->get_degree(), cfs.size() - 1u));
    }

#if defined(HEYOKA_HAVE_REAL)

    // Precision checks for mppp::real.
    if constexpr (std::same_as<T, mppp::real>) {
        const auto expected_prec = ws.m_impl->get_prec();
        if (h.get_prec() != expected_prec || std::ranges::any_of(cfs, [expected_prec](const auto &x) {
                return x.get_prec() != expected_prec;
            })) [[unlikely]] {
            throw std::invalid_argument(fmt::format("One or more multiprecision arguments with invalid precision were "
                                                    "passed to find_real_poly_roots() - the expected precision is {}",
                                                    expected_prec));
        }

        // NOTE: for lb we want to forcibly set the precision instead of checking, since, if not specified, lb will get
        // the precision from the zero int literal its default value is constructed from.
        lb.prec_round(expected_prec);
    }

#endif

    // Finiteness checks.
    if (!is_finite(h)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The h value passed to find_real_poly_roots() must be finite, but a value of {} was passed instead", h));
    }
    if (!std::ranges::all_of(cfs, is_finite)) [[unlikely]] {
        throw std::invalid_argument(
            "The list of coefficients passed to find_real_poly_roots() contains non-finite values");
    }
    // NOTE: lb is allowed to be +inf.
    if (isnan(lb)) [[unlikely]] {
        throw std::invalid_argument("The lb value passed to find_real_poly_roots() must not be NaN");
    }
    if (lb < 0) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The lb value passed to find_real_poly_roots() must not be negative, but a value of {} was provided", lb));
    }

    // Clear out the return value.
    ret.clear();

    // If h is zero, no roots are reported.
    if (h == 0) {
        return;
    }

    // Convert lb to the tau variable.
    const auto lb_tau = lb / abs(h);

    // No roots are reported if lb covers the entire [0, 1) range.
    if (lb_tau >= 1) {
        return;
    }
}

#define HEYOKA_FIND_REAL_POLY_ROOTS_INST(F)                                                                            \
    template HEYOKA_DLL_PUBLIC void find_real_poly_roots<F>(std::vector<real_poly_root<F>> &, frpr_workspace<F> &,     \
                                                            std::span<const F>, const F &, F)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(float);
HEYOKA_FIND_REAL_POLY_ROOTS_INST(double);
HEYOKA_FIND_REAL_POLY_ROOTS_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(mppp::real);

#endif

#undef HEYOKA_FIND_REAL_POLY_ROOTS_INST

HEYOKA_END_NAMESPACE
