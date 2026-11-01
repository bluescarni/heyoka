// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/find_real_poly_roots.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// NOTE: helpers to compute the next/previous representable floating-point numbers.
template <typename T>
T frpr_nextup(T x)
{
    using std::nextafter;
    // NOTE: for mppp::real, the second argument will be an infinity in single precision. This is ok, as
    // mppp::nextafter() ignores the precision of the second argument and outputs a value with the same precision as the
    // first argument.
    return nextafter(std::move(x), static_cast<T>(std::numeric_limits<float>::infinity()));
}

template <typename T>
T frpr_nextdown(T x)
{
    using std::nextafter;
    return nextafter(std::move(x), static_cast<T>(-std::numeric_limits<float>::infinity()));
}

} // namespace

} // namespace detail

// Find the real roots of a polynomial within an interval.
//
// The polynomial p(t) is given by its coefficients cfs in the monomial basis, in ascending degree order, so that the
// polynomial degree is cfs.size() - 1. This function stores in ret the real roots of p(t) for t in [0, h) if h>0, and
// for t in (h, 0] if h<0. If h==0, ret will be empty. If p is the zero polynomial, a single root at t = 0 with dir 0 is
// reported, unless lb != 0.
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
// The exclusion is decided on the computed values of the roots, which are subject to rounding: a root lying within
// rounding error of lb (or -lb) may or may not be reported. If lb is zero, no root is excluded.
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

    // Cache for re-use.
    const auto cfs_size = cfs.size();

    if (ws.m_impl->get_degree() != cfs_size - 1u) [[unlikely]] {
        throw std::invalid_argument(
            // LCOV_EXCL_START
            fmt::format("find_real_poly_roots() was invoked with inconsistent arguments: the polynomial "
                        "degree from the workspace is {}, while the polynomial degree inferred from the list of "
                        "coefficients is {}",
                        // LCOV_EXCL_STOP
                        ws.m_impl->get_degree(), cfs_size - 1u));
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

    // Determine the index of the last nonzero coefficient in the polynomial. If all the coefficients are zero,
    // last_nz_idx is set to cfs_size.
    auto last_nz_idx = cfs_size;
    for (std::size_t i = 0; i < cfs_size; ++i) {
        if (cfs[i] != 0) {
            last_nz_idx = i;
        }
    }

    // Special handling for the zero polynomial: a single root at t = 0 with zero direction (the derivative is exactly
    // zero), unless lb != 0. isolate_real_roots() requires a nonzero polynomial.
    if (last_nz_idx == cfs_size) {
        if (lb == 0) {
            ret.emplace_back(detail::num_zero_like(h), 0);
        }

        return;
    }

    // Special casing for constant non-zero polynomials.
    if (last_nz_idx == 0u) {
        return;
    }

    // Special casing for linear polynomials.
    if (last_nz_idx == 1u) {
        // Compute the root.
        //
        // NOTE: for mppp::real, this is guaranteed to keep the same precision as the original coefficients.
        const auto root = -cfs[0] / cfs[1];
        // Compute a rigorous enclosure for the root. The division is correctly rounded, so the exact root differs from
        // root by less than one ulp, and it lies within [nextdown(root), nextup(root)].
        const auto root_lo = detail::frpr_nextdown(root);
        const auto root_hi = detail::frpr_nextup(root);

        // The direction of the root, that is, the sign of the derivative dp/dt = c1 (which is nonzero here, by
        // definition of last_nz_idx).
        assert(cfs[1] != 0);
        const auto dir = (cfs[1] < 0) ? -1 : 1;

        if (h > 0) {
            if (root_hi < 0 || root_lo >= h) {
                // The root is outside [0, h), exit early.
                return;
            }

            if (root_lo >= 0 && root_hi < h) {
                // The root falls within [0, h), add it if it is outside [0, lb).
                if (root >= lb) {
                    ret.emplace_back(root, dir);
                }

                return;
            }

            // NOTE: if we end up here, it means that the root straddles the bounds of [0, h). We fall through to the
            // general case.
        } else {
            if (root_lo > 0 || root_hi <= h) {
                // The root is outside (h, 0], exit early.
                return;
            }

            if (root_lo > h && root_hi <= 0) {
                // The root falls within (h, 0], add it if it is outside (-lb, 0].
                if (root <= -lb) {
                    ret.emplace_back(root, dir);
                }

                return;
            }

            // NOTE: if we end up here, it means that the root straddles the bounds of (h, 0]. We fall through to the
            // general case.
        }
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
