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
#include <span>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/poly_root_finding/vector_cache.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/find_real_poly_roots.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Small helper to divide an input value by 2, with mppp::real specialisation.
template <typename T>
T isolate_real_roots_div_2(T x)
{
    return x / 2;
}

#if defined(HEYOKA_HAVE_REAL)

// NOTE: the specialisation for mppp::real is guaranteed to produce a result with the same precision as x. It also uses
// a specialised division primitive for performance.
template <>
mppp::real isolate_real_roots_div_2<mppp::real>(mppp::real x)
{
    return mppp::div_2ui(std::move(x), 1u);
}

#endif

} // namespace

// Isolate the real roots for a polynomial.
//
// The polynomial p(t) is given by its coefficients cfs in the monomial basis, in ascending degree order, so that the
// polynomial degree is cfs.size() - 1. This function isolates the real roots of q(tau) = p(h*tau) for tau in [0, 1),
// and returns a reference to the list of resulting intervals. The list is stored in ws. All the values in the returned
// intervals refer to q and tau.
//
// In terms of t, tau in [0, 1) corresponds to t in [0, h) if h>0, and to t in (h, 0] if h<0.
//
// Intervals lying entirely within [0, lb_tau) are not reported.
//
// Preconditions:
//
// - ws was constructed for the degree of p;
// - cfs and h are finite, and h is nonzero;
// - lb_tau is in [0, 1);
// - for mppp::real, cfs, h and lb_tau have the precision of ws.
template <typename T>
const std::vector<irr_interval<T>> &isolate_real_roots(frpr_workspace<T> &ws, const std::span<const T> cfs,
                                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                       const T &h, const T &lb_tau)
{
    // Small helper to reduce typing in the debug checks.
    [[maybe_unused]] constexpr auto is_finite = [](const auto &x) {
        using std::isfinite;
        return isfinite(x);
    };

    // Check preconditions.
    assert(!cfs.empty());
    assert(ws.m_impl->get_degree() == cfs.size() - 1u);

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::same_as<T, mppp::real>) {
        [[maybe_unused]] const auto expected_prec = ws.m_impl->get_prec();
        assert(h.get_prec() == expected_prec);
        assert(std::ranges::all_of(cfs, [expected_prec](const auto &x) { return x.get_prec() == expected_prec; }));
        assert(lb_tau.get_prec() == expected_prec);
    }

#endif

    assert(is_finite(h));
    assert(h != 0);
    assert(std::ranges::all_of(cfs, is_finite));
    assert(is_finite(lb_tau));
    assert(lb_tau >= 0 && lb_tau < 1);

    // Compute the minimum width (in the [0, 1) range of tau) of the intervals that can be bisected in the real-root
    // isolation algorithm. Intervals narrower than this that would need further bisection are reported as undecided.
    const auto min_width = num_eps_like(h);

    // Fetch the list of intervals and clear it out.
    auto &isol_ivals = ws.m_impl->m_intervals;
    isol_ivals.clear();

    // Fetch the working list and clear it out.
    auto &wlist = ws.m_impl->m_wlist;
    wlist.clear();

    // Fetch the function pointers.
    auto *const bernstein_f = ws.m_impl->m_jit_data->m_bernstein_func;
    auto *const decasteljau_func_f = ws.m_impl->m_jit_data->m_decasteljau_func;

    // Fetch vectors to store the results of the conversion of cfs to the Bernstein basis.
    cached_vector<T> bern_poly(ws.m_impl->m_vector_cache), rescaled_poly(ws.m_impl->m_vector_cache);

    // Run the conversion.
    bernstein_f(bern_poly.get_span().data(), rescaled_poly.get_span().data(), cfs.data(), &h);

    // Seed wlist.
    wlist.emplace_back(num_zero_like(h), num_one_like(h), std::move(bern_poly));

    // Run the bisection loop.
    //
    // Within the bisection loop, the objective is to analyze a sub-interval [lo, hi) of the original [0, 1) interval in
    // the tau variable, and to determine, if possible, if the sub-interval contains roots of the polynomial. The main
    // tool is the Descartes rule of signs, applied to the input polynomial in tau after its conversion to the Bernstein
    // basis defined over the [lo, hi] interval. For reference, see Theorem 2.22 here:
    //
    // https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/26046/1/Dissertation_8200_Eige_Arno_2008.pdf
    //
    // One complication arises because the sign rule provides information about the roots over the open interval (lo,
    // hi), whereas our goal is to analyze the half-open interval [lo, hi). Thus, the sign rule must be complemented
    // with explicit checks of the behaviour of the polynomial at lo.
    while (!wlist.empty()) {
        // Pop the current item from wlist.
        const auto [lo, hi, cur_poly] = std::move(wlist.back());
        wlist.pop_back();

        // Fetch the current poly data.
        const auto cur_poly_span = cur_poly.get_span();

        // Count the min/max number of sign changes.
        const auto [V_min, V_max] = ed_count_sign_changes_ival(cur_poly_span);

        // NOTE: in order to determine the behaviour of the polynomial at lo, we can look at the first Bernstein
        // coefficient b_0 (for which we have an interval enclosure). The value of the polynomial at lo equals b_0, and
        // thus we have the following possibilities:
        //
        // - b_0 does *not* include zero -> no root at lo, otherwise
        // - b_0 is exactly [0, 0] -> there is one root at lo (unknown multiplicity), otherwise
        // - b_0 contains 0 -> a root at lo *may* be present (inconclusive).

        // Extract the components of b_0 (recall that the lower bound is stored negated, hence called mlo).
        const auto &b0_mlo = cur_poly_span[0];
        const auto &b0_hi = cur_poly_span[1];

        // Check if b_0 is exactly [0, 0].
        const auto zero_b_0 = (b0_mlo == 0) && (b0_hi == 0);

        // Check if b_0 straddles 0 (i.e., it is not [0, 0] but it contains 0).
        const auto b_0_sz = !zero_b_0 && (b0_mlo >= 0) && (b0_hi >= 0);

        // Check the most common case first: the number of sign changes in (lo, hi) is exactly determined to be 0 (i.e.,
        // no roots in (lo, hi)) and b_0 has a known sign (i.e., no roots at lo).
        if ((V_min == V_max) && (V_min == 0u) && !zero_b_0 && !b_0_sz) {
            // Simply ignore the interval.
            continue;
        }

        // Check if the current interval [lo, hi) is isolating:
        //
        // - the number of sign changes in (lo, hi) is exactly determined to be 1, and
        // - the b_0 coefficient is not [0, 0], nor does it straddle 0.
        //
        // This means that there is no root at lo and exactly 1 root in (lo, hi).
        if ((V_min == V_max) && (V_min == 1u) && !zero_b_0 && !b_0_sz) {
            // Determine the sign of the interval, that is, the sign of the derivative of the polynomial at the root.
            //
            // NOTE: the polynomial has exactly one root r in (lo, hi), so it has a constant sign s on (lo, r) and the
            // opposite sign on (r, hi). The direction of the root is thus -s. Since lo is the left boundary of (lo, r)
            // and the polynomial is continuous, s is the sign of its value at lo, that is, the sign of b_0.
            const auto ival_sign = (b0_hi < 0) ? 1 : -1;

            // Add to the return value.
            isol_ivals.emplace_back(lo, hi, irr_interval_type::isolating, ival_sign);

            // Continue to the next item in wlist.
            continue;
        }

        // Determine if we have a single root at lo. We do if the number of sign changes in (lo, hi) is exactly
        // determined to be 0 (i.e., no roots in (lo, hi)) and b_0 is exactly [0, 0].
        if ((V_min == V_max) && (V_min == 0u) && zero_b_0) {
            // NOTE: make sure that we do not add a root before lb_tau.
            if (lo >= lb_tau) {
                isol_ivals.emplace_back(lo, lo, irr_interval_type::isolating, 0);
            }

            // Continue to the next item in wlist.
            continue;
        }

        // Determine if the interval is undecided:
        //
        // - V_min != V_max (meaning that the number of sign changes cannot be determined exactly, so we have no certain
        //   info on the roots in (lo, hi)), or
        // - b_0 straddles 0 (meaning that we don't know if a root occurs exactly at lo).
        //
        // We will attempt to remove the undecidedness on these intervals with a second bisection pass.
        if (V_min != V_max || b_0_sz) {
            isol_ivals.emplace_back(lo, hi, irr_interval_type::undecided, 0);

            // Continue to the next item in wlist.
            continue;
        }

        // If we end up here, it means that either we have 2 or more sign changes, or we have 1 sign change and b_0 is
        // exactly [0, 0]. Either way, we need to bisect: with 2 or more sign changes, this is the standard bisection
        // condition, in the other case we bisect in order to separate the root at lo from the root in (lo, hi).
        //
        // In both cases, if b_0 is [0, 0], the root at lo is not reported here: it will be reported once, by the left
        // half starting at lo whose sign-change count drops to 0.
        assert(V_min == V_max);
        assert(V_min >= 2u || zero_b_0);

        // NOTE: the bisection won't take place if the interval is too small. In such a case, we report the interval as
        // undecided.
        if (hi - lo < min_width) {
            isol_ivals.emplace_back(lo, hi, irr_interval_type::undecided, 0);

            // Continue to the next item in wlist.
            continue;
        }

        // Fetch a couple of vectors for the split.
        cached_vector<T> left_poly(ws.m_impl->m_vector_cache), right_poly(ws.m_impl->m_vector_cache);

        // Do the split.
        decasteljau_func_f(left_poly.get_span().data(), right_poly.get_span().data(), cur_poly_span.data());

        // Compute the midpoint for the bisection: m = lo/2 + hi/2.
        //
        // NOTE: mid is computed exactly. By induction from [0, 1), [lo, hi) = [k*2**-d, (k+1)*2**-d) for some integers
        // k and d, so the exact midpoint is (2k+1)*2**-(d+1). The integer 2k+1 fits in the p-bit significand:
        //
        // - hi = (k+1)*2**-d <= 1 gives k+1 <= 2**d, hence 2k+1 < 2**(d+1);
        // - the width check above (2**-d >= eps = 2**(1-p)) gives d+1 <= p.
        //
        // lo/2 and hi/2 are exact, and so is their sum, since its exact value is representable.
        const auto mid = isolate_real_roots_div_2(lo) + isolate_real_roots_div_2(hi);
        assert(lo < mid && mid < hi);

        // NOTE: don't add the left split if it falls entirely before lb_tau.
        if (lb_tau < mid) {
            wlist.emplace_back(lo, mid, std::move(left_poly));
        }
        wlist.emplace_back(mid, hi, std::move(right_poly));
    }

    return isol_ivals;
}

#define HEYOKA_ISOLATE_REAL_ROOTS_INST(F)                                                                              \
    template HEYOKA_DLL_PUBLIC const std::vector<irr_interval<F>> &isolate_real_roots<F>(                              \
        frpr_workspace<F> &, std::span<const F>, const F &, const F &)

HEYOKA_ISOLATE_REAL_ROOTS_INST(float);
HEYOKA_ISOLATE_REAL_ROOTS_INST(double);
HEYOKA_ISOLATE_REAL_ROOTS_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_ISOLATE_REAL_ROOTS_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_ISOLATE_REAL_ROOTS_INST(mppp::real);

#endif

#undef HEYOKA_ISOLATE_REAL_ROOTS_INST

} // namespace detail

HEYOKA_END_NAMESPACE
