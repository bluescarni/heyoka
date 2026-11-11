// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <mp++/real.hpp>

#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/find_real_poly_roots.hpp>

HEYOKA_BEGIN_NAMESPACE

// Find the real roots of a polynomial within an interval.
//
// The polynomial p(t) is given by its coefficients cfs in the monomial basis, in ascending degree order, so that the
// polynomial degree is cfs.size() - 1. This function stores in ret the real roots of p(t) for t in [0, h) if h>0, and
// for t in (h, 0] if h<0. If h==0, ret will be empty. If p is the zero polynomial, a single root at t = 0 with dir 0 is
// reported, unless lb != 0.
//
// Every real root of p(t) in this range is reported. Roots that cannot be resolved at the working precision (e.g.,
// multiple roots, or roots closer to each other than the precision allows) may be reported as a single root, and such a
// report may be a false positive. If the location of a root cannot be narrowed down to the working precision (this can
// happen, e.g., for roots of high multiplicity), a std::runtime_error is thrown.
//
// Each reported root value is a point where p is zero within rounding error, but its distance from the exact root is
// not certified. Reports that may be false positives always have an empty dir.
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
// For mppp::real, cfs and h must have the precision of ws, and lb is rounded to it. If any of these requirements is not
// met, a std::invalid_argument is thrown.
//
// The roots in ret are reported in increasing t for h > 0, decreasing t for h < 0. Consecutive roots in ret may have
// the same value. All reported roots are guaranteed to be finite. Roots at zero are always reported as +0.
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
    //
    // NOTE: for mppp::real, lb_tau is guaranteed to have the same precision as lb and h (i.e., the working precision).
    const auto abs_h = abs(h);
    const auto lb_tau = lb / abs_h;

    // No roots are reported if lb covers the entire range. The check on lb_tau also ensures the lb_tau < 1 precondition
    // of isolate_real_roots().
    if (lb >= abs_h || lb_tau >= 1) {
        return;
    }

    // Zero constant with the workspace precision for mppp::real.
    const auto zero = detail::num_zero_like(h);

    // Sign of h (guaranteed to be nonzero at this point).
    const auto h_sgn = (h > 0) ? 1 : -1;

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
            ret.emplace_back(zero, 0);
        }

        return;
    }

    // Special casing for constant non-zero polynomials.
    if (last_nz_idx == 0u) {
        return;
    }

    // Negative of lb, used below.
    const auto mlb = -lb;

    // Special casing for linear polynomials.
    if (last_nz_idx == 1u) {
        // Compute the root.
        //
        // NOTE: for mppp::real, this is guaranteed to keep the same precision as the original coefficients.
        const auto root = -cfs[0] / cfs[1];
        // Compute a rigorous enclosure for the root. The division is correctly rounded, so the exact root differs from
        // root by less than one ulp, and it lies within [nextdown(root), nextup(root)].
        const auto root_lo = detail::num_nextdown(root);
        const auto root_hi = detail::num_nextup(root);

        // The direction of the root, that is, the sign of the derivative dp/dt = c1 (which is nonzero here, by
        // definition of last_nz_idx).
        assert(cfs[1] != 0);
        const auto dir = (cfs[1] < 0) ? -1 : 1;

        if (h_sgn == 1) {
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
                if (root <= mlb) {
                    ret.emplace_back(root, dir);
                }

                return;
            }

            // NOTE: if we end up here, it means that the root straddles the bounds of (h, 0]. We fall through to the
            // general case.
        }
    }

    // We have exhausted the special cases, we can handle the general case now.
    //
    // Run the real root isolation algorithm.
    const auto &ivals = detail::isolate_real_roots(ws, cfs, h, lb_tau);

    // This is ensured to be set by isolate_real_roots().
    assert(ws.m_impl->m_rescaled_poly);
    const auto *const rescaled_poly_data = ws.m_impl->m_rescaled_poly->get_span().data();

    // Helper to add a root to ret.
    //
    // An input root from the [0, 1) interval will be rescaled back to the original [0, h)/(h, 0] interval and clamped
    // before being added to ret. If the root falls within the [0, lb)/(-lb, 0] range, it is ignored.
    const auto add_root = [&ret, &h, h_sgn, lb, &mlb, &zero](T root_tau, const std::optional<int> &sign) {
        assert(root_tau >= 0);
        assert(root_tau < 1);

        // Step 1: rescale the value.
        auto root = std::move(root_tau) * h;

        // Step 2: check for finiteness.
        //
        // NOTE: this is likely unreachable currently, but let's be defensive about this.
        if (!isfinite(root)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::runtime_error(fmt::format("find_real_poly_roots() produced the non-finite root {}", root));
            // LCOV_EXCL_STOP
        }

        // Step 3: clamp the root to the [0, h)/(h, 0] interval, and handle lb.
        if (h_sgn == 1) {
            assert(root >= 0);
            if (root >= h) {
                // LCOV_EXCL_START
                // NOTE: unreachable in practice: since root_tau < 1, root_tau * h can round to h only if |h| is at the
                // bottom of the exponent range (e.g., subnormal).
                root = detail::num_nextdown(h);
                // LCOV_EXCL_STOP
            }

            if (root < lb) {
                return;
            }
        } else {
            assert(root <= 0);
            if (root <= h) {
                // LCOV_EXCL_START
                // NOTE: see above.
                root = detail::num_nextup(h);
                // LCOV_EXCL_STOP
            }

            if (root > mlb) {
                return;
            }
        }

        // Normalise -0 to 0.
        if (root == 0) {
            root = zero;
        }

        // Compute the direction if possible and add the root.
        if (sign) {
            const auto dir = h_sgn * (*sign);
            ret.emplace_back(std::move(root), dir);
        } else {
            ret.emplace_back(std::move(root), std::nullopt);
        }
    };

    // Fetch the root finding function from the JIT.
    auto *const root_finder = ws.m_impl->m_jit_data->m_root_finder_func;

    // Helper to check if an undecided interval is too wide.
    //
    // Undecided intervals not smaller than min_width (in the tau variable) will throw an exception, smaller ones are
    // reported as roots. min_width must match the one used in isolate_real_roots().
    const auto check_undecided_too_wide = [min_width = detail::num_eps_like(h)](const detail::irr_interval<T> &ival) {
        assert(ival.type == detail::irr_interval_type::undecided);
        // NOTE: undecided intervals never have a sign.
        assert(!ival.sign);

        if (ival.hi - ival.lo >= min_width) [[unlikely]] {
            throw std::runtime_error(fmt::format(
                "find_real_poly_roots() was unable to isolate the real roots of a polynomial: the interval "
                "[{}, {}) of the normalised variable t/h could not be resolved (this can happen, e.g., with "
                "roots of high multiplicity)",
                ival.lo, ival.hi));
        }
    };

    // Handle each item in ivals.
    for (decltype(ivals.size()) i = 0; i < ivals.size(); ++i) {
        const auto &[cur_lo, cur_hi, cur_type, cur_sign] = ivals[i];

        if (cur_type == detail::irr_interval_type::isolating) {
            // Isolating interval or exact root.

            // Handle an exact root first. This is flagged by cur_lo == cur_hi.
            if (cur_lo == cur_hi) {
                // Add the root and move on.
                add_root(cur_lo, cur_sign);
                continue;
            }

            // NOTE: isolating intervals must always have a sign. This sign has to be 1 or -1.
            assert(cur_sign);
            assert(*cur_sign == 1 || *cur_sign == -1);

            // Check if there are numbers between cur_lo and cur_hi. If this is not the case, we cannot run the root
            // finding function.
            using std::nextafter;
            if (nextafter(cur_lo, cur_hi) == cur_hi) {
                // LCOV_EXCL_START
                // NOTE: unreachable in practice: an isolating interval with adjacent bounds requires bisecting down to
                // the minimum width in [1/2, 1), which happens only around roots (or complex pairs) closer to each
                // other than ~2**-p.
                //
                // No numbers between lo and hi, report a root at lo directly.
                add_root(cur_lo, cur_sign);
                // LCOV_EXCL_STOP
            } else {
                // lo and hi are suitable for root finding, run it.
                //
                // NOTE: the root finder requires lo/2, hi/2 and lo/2 + hi/2 to be exactly representable. Isolating
                // intervals come from bisections of [0, 1) (in both isolation passes), so they have width 2^-j with j
                // <= p (p = binary digits of T) and bounds that are multiples of 2^-j. For j < p, the halves and the
                // midpoint are multiples of 2^-p in [0, 1), hence representable. For j = p, they are multiples of
                // 2^-(p+1), representable below 0.5, while in [0.5, 1) an interval of width 2^-p has adjacent bounds,
                // which are handled above.
                const std::array bounds = {cur_lo, cur_hi};
                // NOTE: the root finder requires the sign of the exact poly at lo. This is the opposite of the sign of
                // the interval, which is the sign of the derivative at the root.
                const std::int32_t sign = -*cur_sign;
                auto root = zero;
                root_finder(rescaled_poly_data, bounds.data(), &sign, &root);

                // Add the root.
                add_root(std::move(root), cur_sign);
            }

            // Move to the next interval.
            continue;
        }

        // The current interval is undecided. Check if it is too wide.
        check_undecided_too_wide(ivals[i]);

        // The current interval is a small undecided interval. Check if it is the first of a run of adjacent small
        // undecided intervals (each starting exactly where the previous one ends). If it is, we will merge the run and
        // report a single root.
        //
        // NOTE: eff_hi will end up pointing to the last hi bound in case of a run of consecutive small undecided
        // intervals. Otherwise, it will remain cur_hi.
        const auto *eff_hi = &cur_hi;
        while (i + 1u < ivals.size() && ivals[i + 1u].type == detail::irr_interval_type::undecided
               && ivals[i + 1u].lo == ivals[i].hi) {
            ++i;

            // Check the next interval in the run.
            check_undecided_too_wide(ivals[i]);

            // Re-assign eff_hi.
            eff_hi = &ivals[i].hi;
        }

        // At this point [cur_lo, *eff_hi) is either a single small undecided interval, or it is a larger merged
        // undecided interval. We report it as a single root at its midpoint.
        auto candidate = detail::num_div_2(cur_lo) + detail::num_div_2(*eff_hi);
        if (cur_lo > candidate || candidate >= *eff_hi) {
            // If rounding places the midpoint outside the range, report the root at cur_lo.
            candidate = cur_lo;
        }
        add_root(std::move(candidate), std::nullopt);
    }
}

#define HEYOKA_FIND_REAL_POLY_ROOTS_INST(F)                                                                            \
    template HEYOKA_DLL_PUBLIC void find_real_poly_roots<F>(std::vector<real_poly_root<F>> &, frpr_workspace<F> &,     \
                                                            std::span<const F>, const F &, F)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(float);
HEYOKA_FIND_REAL_POLY_ROOTS_INST(double);

#if defined(HEYOKA_HAVE_LONG_DOUBLE)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(long double);

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_FIND_REAL_POLY_ROOTS_INST(mppp::real128);

#endif

HEYOKA_FIND_REAL_POLY_ROOTS_INST(mppp::real);

#undef HEYOKA_FIND_REAL_POLY_ROOTS_INST

HEYOKA_END_NAMESPACE
