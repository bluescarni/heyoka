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
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

namespace
{

template <typename T>
using ivals_t = std::vector<detail::irr_interval<T>>;

using rat_t = boost::multiprecision::cpp_rational;

// Run the real-root isolation on the polynomial with coefficients cfs (ascending degree order) and return a copy of
// the list of intervals.
template <typename T>
ivals_t<T> run_isolation(const std::vector<T> &cfs, T h, T lb_tau)
{
    frpr_workspace<T> ws(static_cast<std::uint32_t>(cfs.size() - 1u));

    return detail::isolate_real_roots(ws, std::span<const T>(cfs), h, lb_tau);
}

// Check if two lists of intervals are identical.
template <typename T>
bool ivals_equal(const ivals_t<T> &a, const ivals_t<T> &b)
{
    return std::ranges::equal(a, b, [](const auto &x, const auto &y) {
        return x.lo == y.lo && x.hi == y.hi && x.type == y.type && x.sign == y.sign;
    });
}

// Check if ivals contains undecided intervals.
template <typename T>
bool has_undecided(const ivals_t<T> &ivals)
{
    return std::ranges::any_of(ivals, [](const auto &iv) { return iv.type == detail::irr_interval_type::undecided; });
}

// Check if ivals contains undecided intervals not smaller than the minimum width used in the isolation (that is, the
// epsilon of T). The first isolation pass may produce such intervals, while a successful extended-precision pass does
// not. T must be a fixed-precision floating-point type.
template <typename T>
bool has_wide_undecided(const ivals_t<T> &ivals)
{
    return std::ranges::any_of(ivals, [](const auto &iv) {
        return iv.type == detail::irr_interval_type::undecided && iv.hi - iv.lo >= std::numeric_limits<T>::epsilon();
    });
}

// Check if ivals contains the entry for a root exactly at x, with the given (possibly empty) sign.
template <typename T>
bool has_root_at(const ivals_t<T> &ivals, T x, std::optional<int> sign)
{
    return std::ranges::any_of(ivals, [x, sign](const auto &iv) {
        return iv.type == detail::irr_interval_type::isolating && iv.lo == x && iv.hi == x && iv.sign == sign;
    });
}

// Check that ivals is a correct output for a polynomial whose real roots in [0, 1), excluding those within the
// cooldown, are exactly the distinct values in roots, each paired with the sign of the derivative at the root (zero
// for a multiple root):
//
// - an isolating interval (lo, hi) with lo < hi contains exactly one root, and its sign is the sign of that root;
// - an entry with lo == hi is exactly at a root, and its sign is either empty or the sign of that root;
// - an undecided interval has an empty sign;
// - every root is covered by an isolating interval, by an entry with lo == hi or by an undecided interval [lo, hi).
template <typename T>
void check_correct(const ivals_t<T> &ivals, const std::vector<std::pair<rat_t, int>> &roots)
{
    using detail::irr_interval_type;

    for (const auto &iv : ivals) {
        if (iv.type != irr_interval_type::isolating) {
            REQUIRE(!iv.sign);
            continue;
        }

        const auto lo = to_rat(iv.lo), hi = to_rat(iv.hi);

        if (lo == hi) {
            REQUIRE(std::ranges::count_if(roots, [&](const auto &r) { return r.first == lo; }) == 1);

            const auto it = std::ranges::find_if(roots, [&](const auto &r) { return r.first == lo; });
            REQUIRE((!iv.sign || *iv.sign == it->second));
        } else {
            REQUIRE(std::ranges::count_if(roots, [&](const auto &r) { return lo < r.first && r.first < hi; }) == 1);

            const auto it = std::ranges::find_if(roots, [&](const auto &r) { return lo < r.first && r.first < hi; });
            REQUIRE(iv.sign == it->second);
        }
    }

    for (const auto &[r, sign] : roots) {
        REQUIRE(std::ranges::any_of(ivals, [&](const auto &iv) {
            const auto lo = to_rat(iv.lo), hi = to_rat(iv.hi);

            if (iv.type == irr_interval_type::isolating) {
                return lo == hi ? r == lo : (lo < r && r < hi);
            } else {
                return lo <= r && r < hi;
            }
        }));
    }
}

} // namespace

// NOTE: in the tests below, the roots in the tau variable are the roots in t divided by h.

TEST_CASE("simple roots")
{
    // p(t) = (t - 3)(t - 6)(t - 8.5), roots at tau = 3/10, 6/10, 17/20. The derivative is positive at the first and
    // last roots, negative at the middle one.
    const auto ivals = run_isolation({-153., 94.5, -17.5, 1.}, 10., 0.);

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    check_correct(ivals, {{rat_t(3, 10), 1}, {rat_t(6, 10), -1}, {rat_t(17, 20), 1}});
}

TEST_CASE("root at zero")
{
    // p(t) = t*(t - 3), roots at tau = 0 and tau = 3/10. The derivative is negative at the first root, positive at the
    // second one. All the interval computations in the isolation are exact for this polynomial, so that the sign of the
    // root at tau = 0 is determined.
    const std::vector cfs = {0., -3., 1.};

    auto ivals = run_isolation(cfs, 10., 0.);

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., -1));
    check_correct(ivals, {{rat_t(0), -1}, {rat_t(3, 10), 1}});

    // With a cooldown, the root at zero is not reported.
    ivals = run_isolation(cfs, 10., 0.1);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(!has_root_at(ivals, 0., -1));
    check_correct(ivals, {{rat_t(3, 10), 1}});
}

TEST_CASE("interval starting within the cooldown")
{
    // p(t) = t - 3, root at tau = 3/10, with a cooldown ending at tau = 1/10. The initial interval [0, 1) is already
    // isolating, and it is reported even though it starts within the cooldown: only the intervals lying entirely within
    // the cooldown are not reported.
    const auto ivals = run_isolation({-3., 1.}, 10., 0.1);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(ivals[0].type == detail::irr_interval_type::isolating);
    REQUIRE(ivals[0].lo == 0.);
    REQUIRE(ivals[0].hi == 1.);

    // The derivative is positive at the root.
    check_correct(ivals, {{rat_t(3, 10), 1}});
}

TEST_CASE("root at split point")
{
    // p(t) = (t - 5)(t - 8), roots at tau = 1/2 (the first bisection point) and tau = 4/5. The derivative is negative
    // at the first root, positive at the second one. All the interval computations in the isolation are exact for this
    // polynomial, so that its value at tau = 1/2 is computed as exactly [0, 0] and the root is reported as a root at
    // the lower bound of [1/2, 3/4), with a determined sign.
    auto ivals = run_isolation({40., -13., 1.}, 10., 0.);

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0.5, -1));
    check_correct(ivals, {{rat_t(1, 2), -1}, {rat_t(4, 5), 1}});

    // p(t) = (t - 2)(t - 5), roots at tau = 1/5 and tau = 1/2 (the first bisection point). The derivative is negative
    // at the first root, positive at the second one. As above, the computations are exact, and the root at tau = 1/2 is
    // reported as a root at the lower bound of [1/2, 1), with a determined sign.
    ivals = run_isolation({10., -7., 1.}, 10., 0.);

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0.5, 1));
    check_correct(ivals, {{rat_t(1, 5), -1}, {rat_t(1, 2), 1}});
}

TEST_CASE("multiple root at split point")
{
    // p(t) = (t - 5)**2, double root at tau = 1/2 (the first bisection point). All the interval computations in the
    // isolation are exact for this polynomial, so that the root is reported as a root at the lower bound of [1/2, 1).
    // The derivative at the root is zero, and so is the sign (which is not empty, since b_1 is computed as exactly
    // [0, 0]).
    const auto ivals = run_isolation({25., -10., 1.}, 10., 0.);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(has_root_at(ivals, 0.5, 0));
    check_correct(ivals, {{rat_t(1, 2), 0}});
}

TEST_CASE("multiple root below the minimum width")
{
    // p(t) = (t - 3)**2, with a double root at tau = 3/10. All the computations in the isolation are exact for this
    // polynomial, thus the sign-change count stays exactly 2 on the intervals containing the root, which are bisected
    // down to the minimum width. The root is then reported in a single undecided interval smaller than the minimum
    // width.
    const auto ivals = run_isolation({9., -6., 1.}, 10., 0.);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(has_undecided(ivals));
    REQUIRE(!has_wide_undecided(ivals));
    check_correct(ivals, {{rat_t(3, 10), 0}});
}

TEST_CASE("no roots")
{
    // p(t) = t**2 + 1.
    const auto ivals = run_isolation({1., 0., 1.}, 1., 0.);

    REQUIRE(ivals.empty());
}

TEST_CASE("constant polys")
{
    // Nonzero constants have no roots.
    REQUIRE(run_isolation({1.}, 1., 0.).empty());
    REQUIRE(run_isolation({-1.}, 1., 0.).empty());
}

TEST_CASE("negative h")
{
    // p(t) = t + 3, root at t = -3, that is, tau = 3/10. q(tau) = p(-10*tau) = 3 - 10*tau is decreasing, hence the
    // sign is -1.
    const auto ivals = run_isolation({3., 1.}, -10., 0.);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(!has_undecided(ivals));
    check_correct(ivals, {{rat_t(3, 10), -1}});
}

TEST_CASE("close roots")
{
    // p(t) = (3t - 1)(3t - 1 - delta)(t + 1) = 9t**3 + (3 - 3*delta)t**2 - (5 + 2*delta)t + 1 + delta, with
    // delta = 2**-50. The coefficients are exactly representable, and the roots in [0, 1) are 1/3 and (1 + delta)/3,
    // about 1.3 epsilons apart. The factor (t + 1) makes the conversion to the Bernstein basis inexact (it divides by
    // 3): without it, all the computations in the first isolation pass would be exact for this polynomial, and the
    // first pass alone would separate the roots. Here instead, the first pass cannot separate them: the sign-change
    // count becomes inexact, and they end up in undecided intervals wider than the minimum width. The
    // extended-precision pass (if available) separates the roots into two isolating intervals.
    const auto delta = std::ldexp(1., -50);
    const auto ivals = run_isolation({1. + delta, -5. - 2. * delta, 3. - 3. * delta, 9.}, 1., 0.);

    // The derivative is negative at the smaller root and positive at the larger one.
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));

#else

    REQUIRE(has_wide_undecided(ivals));

#endif
}

TEST_CASE("extended precision fallback")
{
    // p(t) = (t - 3)**6, with a root of multiplicity 6 at t = 3. The coefficients are exactly representable. h has a
    // full 53-bit significand, so that the rescaling of the polynomial is inexact also in extended precision. With,
    // e.g., h = 10, all the computations would be exact for this polynomial, and the root would be bisected down to the
    // minimum width with an exact sign-change count. Here instead, the first isolation pass reports the root in
    // undecided intervals much wider than the minimum width. The extended-precision pass (if available) exceeds its
    // iteration cap around the root, and the results of the first pass are returned.
    const auto h = 10.1;
    const auto ivals = run_isolation({729., -1458., 1215., -540., 135., -18., 1.}, h, 0.);

    // The root is at tau = 3/h, and the derivative is zero there.
    check_correct(ivals, {{rat_t(3) / to_rat(h), 0}});

    // There is at least one undecided interval not smaller than the minimum width. If the extended-precision pass had
    // succeeded, all the undecided intervals would be smaller than the minimum width.
    REQUIRE(has_wide_undecided(ivals));
}

TEST_CASE("extended precision double root")
{
    // p(t) = (t - 3)**2, with a double root at t = 3, that is, tau = 3/h. h has a full 53-bit significand, so that the
    // rescaling is inexact in the first pass. A double root cannot be isolated: the first pass reports it in undecided
    // intervals much wider than the minimum width, while the extended-precision pass (if available) bisects it down to
    // undecided intervals smaller than the minimum width.
    const auto h = 10.1;
    const auto ivals = run_isolation({9., -6., 1.}, h, 0.);

    // The derivative is zero at the root.
    check_correct(ivals, {{rat_t(3) / to_rat(h), 0}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(has_undecided(ivals));
    REQUIRE(!has_wide_undecided(ivals));

#else

    REQUIRE(has_wide_undecided(ivals));

#endif
}

TEST_CASE("extended precision root at split point")
{
    // p(t) = (t - 5)(t - 8)(t - 9), roots at tau = 1/2 (the first bisection point), 4/5 and 9/10. The conversion to the
    // Bernstein basis is inexact for this polynomial (it divides by 3), so that in both passes the value of the
    // polynomial at tau = 1/2 is computed as an interval straddling zero. The first pass thus reports [0, 1/2) and
    // [1/2, 1) as undecided. The extended-precision pass (if available) bisects them: the root at tau = 1/2 is left in
    // undecided intervals smaller than the minimum width, while the other two roots are isolated.
    const auto ivals = run_isolation({-360., 157., -22., 1.}, 10., 0.);

    // The derivative is positive at the first and last roots, negative at the middle one.
    check_correct(ivals, {{rat_t(1, 2), 1}, {rat_t(4, 5), -1}, {rat_t(9, 10), 1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(std::ranges::count_if(
                ivals, [](const auto &iv) { return iv.type == detail::irr_interval_type::isolating && iv.lo < iv.hi; })
            == 2);
    REQUIRE(has_undecided(ivals));
    REQUIRE(!has_wide_undecided(ivals));

#else

    REQUIRE(has_wide_undecided(ivals));

#endif
}

TEST_CASE("extended precision cooldown")
{
    // p(t) = t*(3t - 1)(3t - 1 - delta), with the two close roots of the "close roots" test and an additional root at
    // tau = 0. As in the "close roots" test, the conversion to the Bernstein basis is inexact (it divides by 3), and
    // the extended-precision pass is needed to separate the two close roots. Here we check that the extended-precision
    // pass (if available) reports the root at tau = 0 only if it is not within the cooldown.
    const auto delta = std::ldexp(1., -50);
    const std::vector cfs = {0., 1. + delta, -(6. + 3. * delta), 9.};

    // No cooldown. The derivative is positive at tau = 0 and at the larger close root, negative at the smaller one.
    auto ivals = run_isolation(cfs, 1., 0.);
    check_correct(ivals, {{rat_t(0), 1}, {rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., 1));

#endif

    // With a cooldown, the root at tau = 0 is not reported.
    ivals = run_isolation(cfs, 1., 0.1);
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));

#endif
}

TEST_CASE("extended precision root at lo")
{
    // In both polynomials below, the close roots of the "close roots" test are needed to trigger the
    // extended-precision pass, and there is an additional root at tau = 0. Here we check the sign of the root at
    // tau = 0 in the extended-precision pass (if available).
    const auto delta = std::ldexp(1., -50);

    // p(t) = -t*(3t - 1)(3t - 1 - delta). The conversion to the Bernstein basis divides by 3. The derivative is
    // negative at tau = 0 and at the larger close root, positive at the smaller one.
    auto ivals = run_isolation({0., -(1. + delta), 6. + 3. * delta, -9.}, 1., 0.);
    check_correct(ivals, {{rat_t(0), -1}, {rat_t(1, 3), 1}, {(1 + to_rat(delta)) / 3, -1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., -1));

#else

    REQUIRE(has_wide_undecided(ivals));

#endif

    // p(t) = t**2*(3t - 1)(3t - 1 - delta), with a double root at tau = 0. The conversion to the Bernstein basis
    // divides by 6. The first two Bernstein coefficients are exactly [0, 0] on all the intervals starting at tau = 0,
    // thus the sign of the root at tau = 0 is zero. The derivative is negative at the smaller close root, positive at
    // the larger one.
    ivals = run_isolation({0., 0., 1. + delta, -(6. + 3. * delta), 9.}, 1., 0.);
    check_correct(ivals, {{rat_t(0), 0}, {rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., 0));

#else

    REQUIRE(has_wide_undecided(ivals));

#endif
}

#if defined(HEYOKA_HAVE_REAL)

// Test the extended-precision pass with mppp::real.
TEST_CASE("extended precision real")
{
    using mppp::real;

    // The polynomial from the "close roots" test, at the precision of double.
    const auto prec = 53;
    const auto delta = std::ldexp(1., -50);

    std::vector<real> cfs;
    for (const auto x : {1. + delta, -5. - 2. * delta, 3. - 3. * delta, 9.}) {
        cfs.emplace_back(x, prec);
    }

    frpr_workspace<real> ws(3, prec);
    const real h{1, prec}, lb_tau{0, prec};
    const auto &ivals = detail::isolate_real_roots(ws, std::span<const real>(cfs), h, lb_tau);

    // As in the "close roots" test, the extended-precision pass separates the roots.
    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

    // The bounds of the intervals have the precision of the workspace.
    for (const auto &iv : ivals) {
        REQUIRE(iv.lo.get_prec() == prec);
        REQUIRE(iv.hi.get_prec() == prec);
    }
}

#endif

// Test the copy/move/serialisation semantics of the workspace.
TEST_CASE("workspace semantics")
{
    // The polynomial from the "simple roots" test, and the list of intervals computed with a fresh workspace.
    const std::vector cfs = {-153., 94.5, -17.5, 1.};
    const auto ref = run_isolation(cfs, 10., 0.);
    REQUIRE(ref.size() == 3u);

    // A polynomial of a different degree.
    const std::vector cfs2 = {1., 0., 1.};

    // Helper to check that w is a working workspace for polynomials of degree 3: it reproduces ref, and
    // find_real_poly_roots() accepts polynomials of degree 3 and rejects polynomials of a different degree.
    const auto check_ws = [&](frpr_workspace<double> &w) {
        REQUIRE(ivals_equal(detail::isolate_real_roots(w, std::span<const double>(cfs), 10., 0.), ref));

        std::vector<real_poly_root<double>> ret;
        REQUIRE_NOTHROW(find_real_poly_roots(ret, w, std::span<const double>(cfs), 10.));
        REQUIRE_THROWS_AS(find_real_poly_roots(ret, w, std::span<const double>(cfs2), 10.), std::invalid_argument);
    };

    frpr_workspace<double> ws(3);
    check_ws(ws);

    // Copy construction. The copy has its own buffers: using it does not alter the results stored in the original.
    {
        auto ws2 = ws;
        check_ws(ws2);

        const auto &res = detail::isolate_real_roots(ws, std::span<const double>(cfs), 10., 0.);

        // p(t) = t**3 + 1, no roots in [0, 1).
        const std::vector cfs3 = {1., 0., 0., 1.};
        REQUIRE(detail::isolate_real_roots(ws2, std::span<const double>(cfs3), 1., 0.).empty());

        REQUIRE(ivals_equal(res, ref));

        check_ws(ws);
    }

    // Move construction. A moved-from workspace can be assigned to and used again.
    {
        auto ws2 = ws;
        auto ws3 = std::move(ws2);
        check_ws(ws3);

        ws2 = ws3;
        check_ws(ws2);
    }

    // Copy assignment, into a workspace of a different degree. Self-assignment.
    {
        frpr_workspace<double> ws2(2);
        ws2 = ws;
        check_ws(ws2);

        auto &ws2_ref = ws2;
        ws2 = ws2_ref;
        check_ws(ws2);
    }

    // Move assignment, into a workspace of a different degree.
    {
        frpr_workspace<double> ws2(2);
        auto ws3 = ws;
        ws2 = std::move(ws3);
        check_ws(ws2);
    }

    // Serialisation round trip, into a workspace of a different degree.
    {
        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ws;
        }

        frpr_workspace<double> ws2(2);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ws2;
        }

        check_ws(ws2);

        // The original is unaffected.
        check_ws(ws);
    }
}

// Test the input checks of find_real_poly_roots().
TEST_CASE("input checks")
{
    using Catch::Matchers::Message;

    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    // The precision argument is ignored for fixed-precision types.
    REQUIRE_NOTHROW(frpr_workspace<double>(2, -5));

    frpr_workspace<double> ws(2);
    const std::vector cfs = {1., 2., 3.};
    const auto span = std::span<const double>(cfs);
    std::vector<real_poly_root<double>> ret;

    // Empty list of coefficients.
    REQUIRE_THROWS_MATCHES(
        find_real_poly_roots(ret, ws, std::span<const double>{}, 1.), std::invalid_argument,
        Message("Cannot invoke find_real_poly_roots() with an empty list of polynomial coefficients"));

    // Degree mismatch with the workspace.
    const std::vector cfs3 = {1., 2., 3., 4.};
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, std::span<const double>(cfs3), 1.), std::invalid_argument,
                           Message("find_real_poly_roots() was invoked with inconsistent arguments: the polynomial "
                                   "degree from the workspace is 2, while the polynomial degree inferred from the list "
                                   "of coefficients is 3"));

    // Non-finite h.
    REQUIRE_THROWS_MATCHES(
        find_real_poly_roots(ret, ws, span, inf), std::invalid_argument,
        Message("The h value passed to find_real_poly_roots() must be finite, but a value of inf was "
                "passed instead"));
    REQUIRE_THROWS_MATCHES(
        find_real_poly_roots(ret, ws, span, nan), std::invalid_argument,
        Message("The h value passed to find_real_poly_roots() must be finite, but a value of nan was "
                "passed instead"));

    // Non-finite coefficients.
    for (const auto x : {inf, -inf, nan}) {
        auto bad_cfs = cfs;
        bad_cfs[1] = x;
        REQUIRE_THROWS_MATCHES(
            find_real_poly_roots(ret, ws, std::span<const double>(bad_cfs), 1.), std::invalid_argument,
            Message("The list of coefficients passed to find_real_poly_roots() contains non-finite values"));
    }

    // Invalid lb.
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, 1., nan), std::invalid_argument,
                           Message("The lb value passed to find_real_poly_roots() must not be NaN"));
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, 1., -1.), std::invalid_argument,
                           Message("The lb value passed to find_real_poly_roots() must not be negative, but a value of "
                                   "-1 was provided"));

    // Valid inputs for which no roots are reported, and ret must be cleared out: h == 0, lb == +inf, and lb covering
    // the whole range for either sign of h.
    for (const auto &[h, lb] : {std::pair{0., 0.}, std::pair{1., inf}, std::pair{1., 1.}, std::pair{-1., 2.}}) {
        ret.resize(1);
        REQUIRE_NOTHROW(find_real_poly_roots(ret, ws, span, h, lb));
        REQUIRE(ret.empty());
    }
}

#if defined(HEYOKA_HAVE_REAL)

// Test the input checks of find_real_poly_roots() specific to mppp::real.
TEST_CASE("input checks real")
{
    using Catch::Matchers::Message;

    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    // Invalid precision for the workspace.
    REQUIRE_THROWS_MATCHES(frpr_workspace<mppp::real>(2, 0), std::invalid_argument,
                           Message(fmt::format("Invalid precision value 0 specified to the constructor of a "
                                               "frpr_workspace: the value must be within the [{}, {}] bounds",
                                               mppp::real_prec_min(), mppp::real_prec_max())));

    const auto prec = 30;

    frpr_workspace<mppp::real> ws(2, prec);
    const std::vector cfs = {mppp::real{1., prec}, mppp::real{2., prec}, mppp::real{3., prec}};
    const auto span = std::span<const mppp::real>(cfs);
    const mppp::real h{1., prec};
    std::vector<real_poly_root<mppp::real>> ret;

    const auto prec_msg = Message("One or more multiprecision arguments with invalid precision were passed to "
                                  "find_real_poly_roots() - the expected precision is 30");

    // h with the wrong precision.
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, mppp::real{1., prec + 1}), std::invalid_argument,
                           prec_msg);

    // A coefficient with the wrong precision.
    auto bad_cfs = cfs;
    bad_cfs[1] = mppp::real{2., prec + 1};
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, std::span<const mppp::real>(bad_cfs), h),
                           std::invalid_argument, prec_msg);

    // Non-finite h and coefficients.
    //
    // NOTE: the message for a non-finite h contains the formatted value, which depends on the formatting of
    // mppp::real: we check only the exception type.
    REQUIRE_THROWS_AS(find_real_poly_roots(ret, ws, span, mppp::real{inf, prec}), std::invalid_argument);
    bad_cfs = cfs;
    bad_cfs[1] = mppp::real{nan, prec};
    REQUIRE_THROWS_MATCHES(
        find_real_poly_roots(ret, ws, std::span<const mppp::real>(bad_cfs), h), std::invalid_argument,
        Message("The list of coefficients passed to find_real_poly_roots() contains non-finite values"));

    // lb is accepted with any precision, including the precision of its default value.
    REQUIRE_NOTHROW(find_real_poly_roots(ret, ws, span, h));
    REQUIRE_NOTHROW(find_real_poly_roots(ret, ws, span, h, mppp::real{0.5, prec + 10}));

    // Invalid lb.
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, h, mppp::real{nan, prec}), std::invalid_argument,
                           Message("The lb value passed to find_real_poly_roots() must not be NaN"));
    REQUIRE_THROWS_AS(find_real_poly_roots(ret, ws, span, h, mppp::real{-1., prec}), std::invalid_argument);
}

#endif
