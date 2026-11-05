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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <mp++/real.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

namespace
{

template <typename T>
using ivals_t = std::vector<detail::irr_interval<T>>;

using rat_t = boost::multiprecision::cpp_rational;

// The floating-point types tested in the find_real_poly_roots() tests.
const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Check whether the exact value r lies within k floating-point values of x.
template <typename T>
bool within_steps(const T &x, const rat_t &r, unsigned k)
{
    using std::nextafter;

    // NOTE: the directions are given via the infinity of double converted to T, because std::numeric_limits is not
    // specialised for mppp::real (its infinity() is zero).
    const auto inf = static_cast<T>(std::numeric_limits<double>::infinity());

    auto below = x;
    auto above = x;
    for (unsigned i = 0; i < k; ++i) {
        below = nextafter(below, -inf);
        above = nextafter(above, inf);
    }

    return to_rat(below) <= r && r <= to_rat(above);
}

// Run the real-root isolation on the polynomial with coefficients cfs (ascending degree order) and return a copy of
// the list of intervals.
template <typename T>
ivals_t<T> run_isolation(const std::vector<T> &cfs, T h, T lb_tau)
{
    frpr_workspace<T> ws(static_cast<std::uint32_t>(cfs.size() - 1u));

    return detail::isolate_real_roots(ws, std::span<const T>(cfs), h, lb_tau);
}

// Run find_real_poly_roots() on the polynomial with coefficients cfs (ascending degree order) and return the list of
// roots.
template <typename T>
std::vector<real_poly_root<T>> run_find_roots(const std::vector<T> &cfs, T h, T lb)
{
    frpr_workspace<T> ws(static_cast<std::uint32_t>(cfs.size() - 1u));
    std::vector<real_poly_root<T>> ret;
    find_real_poly_roots(ret, ws, std::span<const T>(cfs), h, lb);

    return ret;
}

// Check if two lists of intervals are identical.
template <typename T>
bool ivals_equal(const ivals_t<T> &a, const ivals_t<T> &b)
{
    return std::ranges::equal(a, b, [](const auto &x, const auto &y) {
        return x.lo == y.lo && x.hi == y.hi && x.type == y.type && x.sign == y.sign;
    });
}

// Check if two lists of roots returned by find_real_poly_roots() are identical.
template <typename T>
bool roots_equal(const std::vector<real_poly_root<T>> &a, const std::vector<real_poly_root<T>> &b)
{
    return std::ranges::equal(a, b, [](const auto &x, const auto &y) { return x.t == y.t && x.dir == y.dir; });
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

// Check that ivals is a correct output for a polynomial whose real roots in [0, 1), excluding those within
// [0, lb_tau), are exactly the distinct values in roots, each paired with the sign of the derivative at the root
// (zero for a multiple root):
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

    // With a lower bound lb_tau = 0.1, the root at zero is not reported.
    ivals = run_isolation(cfs, 10., 0.1);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(!has_root_at(ivals, 0., -1));
    check_correct(ivals, {{rat_t(3, 10), 1}});
}

TEST_CASE("interval starting below the lower bound")
{
    // p(t) = t - 3, root at tau = 3/10, with a lower bound lb_tau = 1/10. The initial interval [0, 1) is already
    // isolating, and it is reported even though it starts below the lower bound: only the intervals lying entirely
    // within [0, lb_tau) are not reported.
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
    // extended-precision pass separates the roots into two isolating intervals.
    const auto delta = std::ldexp(1., -50);
    const auto ivals = run_isolation({1. + delta, -5. - 2. * delta, 3. - 3. * delta, 9.}, 1., 0.);

    // The derivative is negative at the smaller root and positive at the larger one.
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
}

TEST_CASE("extended precision fallback")
{
    // p(t) = (t - 3)**6, with a root of multiplicity 6 at t = 3. The coefficients are exactly representable. h has a
    // full 53-bit significand, so that the rescaling of the polynomial is inexact also in extended precision. With,
    // e.g., h = 10, all the computations would be exact for this polynomial, and the root would be bisected down to the
    // minimum width with an exact sign-change count. Here instead, the first isolation pass reports the root in
    // undecided intervals much wider than the minimum width. The extended-precision pass exceeds its
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
    // intervals much wider than the minimum width, while the extended-precision pass bisects it down to
    // undecided intervals smaller than the minimum width.
    const auto h = 10.1;
    const auto ivals = run_isolation({9., -6., 1.}, h, 0.);

    // The derivative is zero at the root.
    check_correct(ivals, {{rat_t(3) / to_rat(h), 0}});

    REQUIRE(has_undecided(ivals));
    REQUIRE(!has_wide_undecided(ivals));
}

TEST_CASE("extended precision root at split point")
{
    // p(t) = (t - 5)(t - 8)(t - 9), roots at tau = 1/2 (the first bisection point), 4/5 and 9/10. The conversion to the
    // Bernstein basis is inexact for this polynomial (it divides by 3), so that in both passes the value of the
    // polynomial at tau = 1/2 is computed as an interval straddling zero. The first pass thus reports [0, 1/2) and
    // [1/2, 1) as undecided. The extended-precision pass bisects them: the root at tau = 1/2 is left in
    // undecided intervals smaller than the minimum width, while the other two roots are isolated.
    const auto ivals = run_isolation({-360., 157., -22., 1.}, 10., 0.);

    // The derivative is positive at the first and last roots, negative at the middle one.
    check_correct(ivals, {{rat_t(1, 2), 1}, {rat_t(4, 5), -1}, {rat_t(9, 10), 1}});

    REQUIRE(std::ranges::count_if(
                ivals, [](const auto &iv) { return iv.type == detail::irr_interval_type::isolating && iv.lo < iv.hi; })
            == 2);
    REQUIRE(has_undecided(ivals));
    REQUIRE(!has_wide_undecided(ivals));
}

TEST_CASE("extended precision lower bound")
{
    // p(t) = t*(3t - 1)(3t - 1 - delta), with the two close roots of the "close roots" test and an additional root at
    // tau = 0. As in the "close roots" test, the conversion to the Bernstein basis is inexact (it divides by 3), and
    // the extended-precision pass is needed to separate the two close roots. Here we check that the extended-precision
    // pass reports the root at tau = 0 only if it is not within [0, lb_tau).
    const auto delta = std::ldexp(1., -50);
    const std::vector cfs = {0., 1. + delta, -(6. + 3. * delta), 9.};

    // No lower bound. The derivative is positive at tau = 0 and at the larger close root, negative at the smaller one.
    auto ivals = run_isolation(cfs, 1., 0.);
    check_correct(ivals, {{rat_t(0), 1}, {rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., 1));

    // With a lower bound lb_tau = 0.1, the root at tau = 0 is not reported.
    ivals = run_isolation(cfs, 1., 0.1);
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
}

TEST_CASE("extended precision root at lo")
{
    // In both polynomials below, the close roots of the "close roots" test are needed to trigger the
    // extended-precision pass, and there is an additional root at tau = 0. Here we check the sign of the root at
    // tau = 0 in the extended-precision pass.
    const auto delta = std::ldexp(1., -50);

    // p(t) = -t*(3t - 1)(3t - 1 - delta). The conversion to the Bernstein basis divides by 3. The derivative is
    // negative at tau = 0 and at the larger close root, positive at the smaller one.
    auto ivals = run_isolation({0., -(1. + delta), 6. + 3. * delta, -9.}, 1., 0.);
    check_correct(ivals, {{rat_t(0), -1}, {rat_t(1, 3), 1}, {(1 + to_rat(delta)) / 3, -1}});

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., -1));

    // p(t) = t**2*(3t - 1)(3t - 1 - delta), with a double root at tau = 0. The conversion to the Bernstein basis
    // divides by 6. The first two Bernstein coefficients are exactly [0, 0] on all the intervals starting at tau = 0,
    // thus the sign of the root at tau = 0 is zero. The derivative is negative at the smaller close root, positive at
    // the larger one.
    ivals = run_isolation({0., 0., 1. + delta, -(6. + 3. * delta), 9.}, 1., 0.);
    check_correct(ivals, {{rat_t(0), 0}, {rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});

    REQUIRE(ivals.size() == 3u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0., 0));
}

TEST_CASE("undecided interval without real roots")
{
    // p(t) = t**2 - 2*a*t + c0, with a = 2**-38 + 2**-54 + 2**-90 and c0 the double just above a**2. The coefficients
    // are exactly representable. The roots are a +- i*b, with b**2 = c0 - a**2 > 0 and b ~ 2**-64: there are no real
    // roots, but the complex pair is much closer to the real axis than the minimum width. The intervals containing a
    // keep 2 sign changes down to the minimum width, thus the root isolation reports an undecided interval around a,
    // although there is no real root in it (a false positive).
    //
    // NOTE: a must be far (relative to b**2/eps) from the bisection points, otherwise the intervals starting or ending
    // close to a would have no sign changes and the complex pair would be resolved.
    const auto c0 = 0x1.0002000100003p-76;
    const auto c1 = -0x1.0001000000001p-37;
    const auto a = -to_rat(c1) / 2;

    // No real roots.
    REQUIRE(to_rat(c0) > a * a);

    const auto ivals = run_isolation({c0, c1, 1.}, 1., 0.);

    // No root is certified, but there is an undecided interval containing a.
    check_correct(ivals, {});
    REQUIRE(std::ranges::any_of(ivals, [&a](const auto &iv) {
        return iv.type == detail::irr_interval_type::undecided && to_rat(iv.lo) <= a && a < to_rat(iv.hi);
    }));

    // The final undecided interval is smaller than the minimum width.
    REQUIRE(ivals.size() == 1u);
    REQUIRE(!has_wide_undecided(ivals));
}

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

// Test the input checks of find_real_poly_roots() specific to mppp::real.
TEST_CASE("input checks real")
{
    using Catch::Matchers::Message;

    const auto inf = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();

    // Invalid precision for the workspace.
    REQUIRE_THROWS_MATCHES(frpr_workspace<mppp::real>(2, 0), std::invalid_argument,
                           Message(fmt::format("Invalid precision value 0 specified to the constructor of an "
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

TEST_CASE("find_real_poly_roots zero polynomial")
{
    // The zero polynomial is reported as a single root at t = 0 with zero direction, for any degree and either sign of
    // h, unless lb is nonzero.
    for (const auto &cfs : {std::vector{0.}, std::vector{0., 0., 0.}}) {
        for (const auto h : {1., -1.}) {
            const auto ret = run_find_roots(cfs, h, 0.);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t == 0);
            REQUIRE(ret[0].dir == 0);

            REQUIRE(run_find_roots(cfs, h, 0.5).empty());
        }
    }
}

TEST_CASE("find_real_poly_roots constant polynomials")
{
    // Nonzero constants have no roots, for any degree and either sign of h.
    for (const auto &cfs : {std::vector{2.}, std::vector{-2., 0., 0.}}) {
        for (const auto h : {1., -1.}) {
            REQUIRE(run_find_roots(cfs, h, 0.).empty());
        }
    }
}

TEST_CASE("find_real_poly_roots linear polynomials")
{
    // Helper to check that the polynomial with coefficients cfs has a single root at t with direction dir.
    const auto check_root = [](const std::vector<double> &cfs, double h, double lb, double t, int dir) {
        const auto ret = run_find_roots(cfs, h, lb);
        REQUIRE(ret.size() == 1u);
        REQUIRE(ret[0].t == t);
        REQUIRE(ret[0].dir == dir);
    };

    // Roots within the range, for both signs of the derivative and of h. The roots are computed exactly here.
    check_root({-3., 1.}, 10., 0., 3., 1);
    check_root({6., -2.}, 10., 0., 3., -1);
    check_root({3., 1.}, -10., 0., -3., 1);
    check_root({-6., -2.}, -10., 0., -3., -1);

    // A polynomial of higher degree whose coefficients of degree 2 and higher are all zero.
    check_root({-3., 1., 0., 0.}, 10., 0., 3., 1);

    // Roots outside the range.
    REQUIRE(run_find_roots({-30., 1.}, 10., 0.).empty());
    REQUIRE(run_find_roots({3., 1.}, 10., 0.).empty());
    REQUIRE(run_find_roots({-3., 1.}, -10., 0.).empty());
    REQUIRE(run_find_roots({30., 1.}, -10., 0.).empty());

    // Roots in [0, lb) (h > 0) or in (-lb, 0] (h < 0) are not reported, roots exactly at lb or -lb are.
    REQUIRE(run_find_roots({-3., 1.}, 10., 5.).empty());
    check_root({-3., 1.}, 10., 3., 3., 1);
    REQUIRE(run_find_roots({3., 1.}, -10., 5.).empty());
    check_root({3., 1.}, -10., 3., -3., 1);

    // The return value is cleared out before adding the roots.
    std::vector<real_poly_root<double>> ret(3);
    frpr_workspace<double> ws(1);
    const std::vector cfs = {-3., 1.};
    find_real_poly_roots(ret, ws, std::span<const double>(cfs), 10.);
    REQUIRE(ret.size() == 1u);
    REQUIRE(ret[0].t == 3);
}

TEST_CASE("find_real_poly_roots linear polynomials real")
{
    using mppp::real;

    const auto prec = 100;
    frpr_workspace<real> ws(1, prec);
    const real h{1, prec};
    std::vector<real_poly_root<real>> ret;

    // p(t) = 3t - 1, with a root at t = 1/3 (not exactly representable). The root has the precision of the workspace.
    const std::vector cfs = {real{-1, prec}, real{3, prec}};
    find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);
    REQUIRE(ret.size() == 1u);
    REQUIRE(ret[0].t.get_prec() == prec);
    REQUIRE(ret[0].t == real{1, prec} / 3);
    REQUIRE(ret[0].dir == 1);

    // p(t) = t - 2, with a root outside the range.
    const std::vector cfs2 = {real{-2, prec}, real{1, prec}};
    find_real_poly_roots(ret, ws, std::span<const real>(cfs2), h);
    REQUIRE(ret.empty());
}

TEST_CASE("find_real_poly_roots simple roots")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check that the roots of the polynomial with coefficients cfs are, in order, within k
        // floating-point values of the exact values in roots, with the given directions.
        //
        // NOTE: k = 40 has been measured by exact emulation of the root finder and of the conversion t = h*tau (the
        // maximum distance is 34 floating-point values, at t = 2 and t = -2).
        const auto check = [](const std::vector<fp_t> &cfs, fp_t h, const std::vector<std::pair<rat_t, int>> &roots) {
            const auto ret = run_find_roots(cfs, h, fp_t(0));
            REQUIRE(ret.size() == roots.size());
            for (std::size_t i = 0; i < roots.size(); ++i) {
                REQUIRE(within_steps(ret[i].t, roots[i].first, 40));
                REQUIRE(ret[i].dir == roots[i].second);
            }
        };

        // p(t) = (t - 1)(t - 2)(2t - 5) with h = 3. In the tau variable the roots are 1/3, 2/3 and 5/6, which are
        // not dyadic and thus are found by the root finder. With an integer h, the rescaled polynomial is computed
        // exactly.
        check({fp_t(-10), fp_t(19), fp_t(-11), fp_t(2)}, fp_t(3), {{rat_t(1), 1}, {rat_t(2), -1}, {rat_t(5, 2), 1}});

        // The mirror image (t + 1)(t + 2)(2t + 5) with h = -3: same roots in the tau variable, reported in decreasing
        // order of t.
        check({fp_t(10), fp_t(19), fp_t(11), fp_t(2)}, fp_t(-3), {{rat_t(-1), 1}, {rat_t(-2), -1}, {rat_t(-5, 2), 1}});
    });
}

TEST_CASE("find_real_poly_roots simple roots real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots simple roots" test, plus the precision of the roots.
        //
        // NOTE: k = 40 has been measured by exact emulation for all the precisions used here.
        const auto check
            = [prec](const std::vector<real> &cfs, const real &h, const std::vector<std::pair<rat_t, int>> &roots) {
                  frpr_workspace<real> ws(static_cast<std::uint32_t>(cfs.size() - 1u), prec);
                  std::vector<real_poly_root<real>> ret;
                  find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);

                  REQUIRE(ret.size() == roots.size());
                  for (std::size_t i = 0; i < roots.size(); ++i) {
                      REQUIRE(ret[i].t.get_prec() == prec);
                      REQUIRE(within_steps(ret[i].t, roots[i].first, 40));
                      REQUIRE(ret[i].dir == roots[i].second);
                  }
              };

        check({r(-10), r(19), r(-11), r(2)}, r(3), {{rat_t(1), 1}, {rat_t(2), -1}, {rat_t(5, 2), 1}});
        check({r(10), r(19), r(11), r(2)}, r(-3), {{rat_t(-1), 1}, {rat_t(-2), -1}, {rat_t(-5, 2), 1}});
    }
}

// The polynomials in the "root at zero" tests: p(t) = t(t - 3) with h = 10, and p(t) = t(t + 3) with h = -10. In both
// cases the rescaled polynomial is q(tau) = 100 tau**2 - 30 tau, with an exact root at tau = 0 (found by the isolation
// as a root exactly at the lower bound of an interval) and a root at tau = 0.3 (found by the root finder). For h = -10,
// the root at tau = 0 is computed as -10 * 0 = -0, which must be reported as +0.
//
// NOTE: the tolerance k = 4 for the root at tau = 0.3 has been measured by exact emulation (the maximum distance is 1
// floating-point value).

TEST_CASE("find_real_poly_roots root at zero")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check that the polynomial with coefficients cfs has a root at +0 with direction dir0, followed by
        // a root within 4 floating-point values of root1 with direction -dir0.
        const auto check = [](const std::vector<fp_t> &cfs, fp_t h, int dir0, const rat_t &root1) {
            using std::signbit;

            const auto ret = run_find_roots(cfs, h, fp_t(0));
            REQUIRE(ret.size() == 2u);
            REQUIRE(ret[0].t == 0);
            REQUIRE(!signbit(ret[0].t));
            REQUIRE(ret[0].dir == dir0);
            REQUIRE(within_steps(ret[1].t, root1, 4));
            REQUIRE(ret[1].dir == -dir0);
        };

        check({fp_t(0), fp_t(-3), fp_t(1)}, fp_t(10), -1, rat_t(3));
        check({fp_t(0), fp_t(3), fp_t(1)}, fp_t(-10), 1, rat_t(-3));
    });
}

TEST_CASE("find_real_poly_roots root at zero real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots root at zero" test, plus the precision of the roots.
        const auto check = [prec](const std::vector<real> &cfs, const real &h, int dir0, const rat_t &root1) {
            frpr_workspace<real> ws(2, prec);
            std::vector<real_poly_root<real>> ret;
            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);

            REQUIRE(ret.size() == 2u);
            REQUIRE(ret[0].t == 0);
            REQUIRE(!signbit(ret[0].t));
            REQUIRE(ret[0].t.get_prec() == prec);
            REQUIRE(ret[0].dir == dir0);
            REQUIRE(ret[1].t.get_prec() == prec);
            REQUIRE(within_steps(ret[1].t, root1, 4));
            REQUIRE(ret[1].dir == -dir0);
        };

        check({r(0), r(-3), r(1)}, r(10), -1, rat_t(3));
        check({r(0), r(3), r(1)}, r(-10), 1, rat_t(-3));
    }
}

// The polynomials in the "root at split point" tests are those of the "root at split point" and "multiple root at
// split point" isolation tests, (t - 5)(t - 8), (t - 2)(t - 5) and (t - 5)**2 with h = 10, together with their mirror
// images (t + 5)(t + 8), (t + 2)(t + 5) and (t + 5)**2 with h = -10, which have the same rescaled polynomials. The
// roots at t = 5 and t = -5 (tau = 1/2) are exact roots at the lower bound of an interval, and they must be reported
// exactly. The double root has zero direction.
//
// NOTE: the tolerance k = 8 for the other roots has been measured by exact emulation (the maximum distance is 5
// floating-point values).

TEST_CASE("find_real_poly_roots root at split point")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check the roots of the polynomial with coefficients cfs. Each element of roots contains the exact
        // value, the direction and a flag signalling if the root must be reported exactly (otherwise, it must be within
        // 8 floating-point values of the exact value).
        const auto check
            = [](const std::vector<fp_t> &cfs, fp_t h, const std::vector<std::tuple<rat_t, int, bool>> &roots) {
                  const auto ret = run_find_roots(cfs, h, fp_t(0));
                  REQUIRE(ret.size() == roots.size());
                  for (std::size_t i = 0; i < roots.size(); ++i) {
                      const auto &[r, dir, exact] = roots[i];
                      REQUIRE(within_steps(ret[i].t, r, exact ? 0u : 8u));
                      REQUIRE(ret[i].dir == dir);
                  }
              };

        check({fp_t(40), fp_t(-13), fp_t(1)}, fp_t(10), {{rat_t(5), -1, true}, {rat_t(8), 1, false}});
        check({fp_t(40), fp_t(13), fp_t(1)}, fp_t(-10), {{rat_t(-5), 1, true}, {rat_t(-8), -1, false}});

        check({fp_t(10), fp_t(-7), fp_t(1)}, fp_t(10), {{rat_t(2), -1, false}, {rat_t(5), 1, true}});
        check({fp_t(10), fp_t(7), fp_t(1)}, fp_t(-10), {{rat_t(-2), 1, false}, {rat_t(-5), -1, true}});

        check({fp_t(25), fp_t(-10), fp_t(1)}, fp_t(10), {{rat_t(5), 0, true}});
        check({fp_t(25), fp_t(10), fp_t(1)}, fp_t(-10), {{rat_t(-5), 0, true}});
    });
}

TEST_CASE("find_real_poly_roots root at split point real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots root at split point" test, plus the precision of the roots.
        const auto check = [prec](const std::vector<real> &cfs, const real &h,
                                  const std::vector<std::tuple<rat_t, int, bool>> &roots) {
            frpr_workspace<real> ws(2, prec);
            std::vector<real_poly_root<real>> ret;
            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);

            REQUIRE(ret.size() == roots.size());
            for (std::size_t i = 0; i < roots.size(); ++i) {
                const auto &[rt, dir, exact] = roots[i];
                REQUIRE(ret[i].t.get_prec() == prec);
                REQUIRE(within_steps(ret[i].t, rt, exact ? 0u : 8u));
                REQUIRE(ret[i].dir == dir);
            }
        };

        check({r(40), r(-13), r(1)}, r(10), {{rat_t(5), -1, true}, {rat_t(8), 1, false}});
        check({r(40), r(13), r(1)}, r(-10), {{rat_t(-5), 1, true}, {rat_t(-8), -1, false}});

        check({r(10), r(-7), r(1)}, r(10), {{rat_t(2), -1, false}, {rat_t(5), 1, true}});
        check({r(10), r(7), r(1)}, r(-10), {{rat_t(-2), 1, false}, {rat_t(-5), -1, true}});

        check({r(25), r(-10), r(1)}, r(10), {{rat_t(5), 0, true}});
        check({r(25), r(10), r(1)}, r(-10), {{rat_t(-5), 0, true}});
    }
}

// The polynomials in the "double root" tests: (t - 3)**2 with h = 10 (the polynomial of the "multiple root below the
// minimum width" isolation test), and its mirror image (t + 3)**2 with h = -10, which has the same rescaled polynomial.
// All the computations in the isolation are exact, so that the double root at tau = 3/10 ends up in a single undecided
// interval [lo, hi) with hi - lo = 2**-p (p being the number of binary digits), which is reported as a root at its
// midpoint, t = h*mid with mid = lo/2 + hi/2, with an empty direction.
//
// NOTE: the tolerance k = 4 has been measured by exact computation of h*mid (the maximum distance is 1 floating-point
// value).

TEST_CASE("find_real_poly_roots double root")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check that the polynomial with coefficients cfs has a single root within 4 floating-point values of
        // root, with an empty direction.
        const auto check = [](const std::vector<fp_t> &cfs, fp_t h, const rat_t &root) {
            const auto ret = run_find_roots(cfs, h, fp_t(0));
            REQUIRE(ret.size() == 1u);
            REQUIRE(within_steps(ret[0].t, root, 4));
            REQUIRE(!ret[0].dir);
        };

        check({fp_t(9), fp_t(-6), fp_t(1)}, fp_t(10), rat_t(3));
        check({fp_t(9), fp_t(6), fp_t(1)}, fp_t(-10), rat_t(-3));
    });
}

TEST_CASE("find_real_poly_roots double root real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots double root" test, plus the precision of the root.
        const auto check = [prec](const std::vector<real> &cfs, const real &h, const rat_t &root) {
            frpr_workspace<real> ws(2, prec);
            std::vector<real_poly_root<real>> ret;
            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);

            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t.get_prec() == prec);
            REQUIRE(within_steps(ret[0].t, root, 4));
            REQUIRE(!ret[0].dir);
        };

        check({r(9), r(-6), r(1)}, r(10), rat_t(3));
        check({r(9), r(6), r(1)}, r(-10), rat_t(-3));
    }
}

// The polynomials in the "undecided interval straddling lb" tests: (3t - 1)**2 with h = 1, and its mirror image
// (3t + 1)**2 with h = -1, which has the same rescaled polynomial. As in the "double root" tests, all the computations
// in the isolation are exact, and the double root at tau = 1/3 ends up in a single undecided interval [lo, hi) with hi
// - lo = 2**-p, which is reported as a root at its midpoint mid = lo/2 + hi/2 (representable, since 1/3 is in
// [1/4, 1/2)). With |h| = 1, lb_tau = lb and t = h*tau are computed exactly. The checks:
//
// - with lb = lo, and with lb = mid (the interval straddles lb_tau), the root is reported at h*mid;
// - with lb = hi no root is reported (the interval lies entirely within [0, lb_tau)).

TEST_CASE("find_real_poly_roots undecided interval straddling lb")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        const auto check = [](const std::vector<fp_t> &cfs, fp_t h) {
            // Fetch the undecided interval from the isolation.
            const auto ivals = run_isolation(cfs, h, fp_t(0));
            REQUIRE(ivals.size() == 1u);
            REQUIRE(ivals[0].type == detail::irr_interval_type::undecided);
            const auto lo = ivals[0].lo, hi = ivals[0].hi;
            const auto mid = lo / 2 + hi / 2;
            REQUIRE(lo < mid);
            REQUIRE(mid < hi);

            auto ret = run_find_roots(cfs, h, mid);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t == h * mid);
            REQUIRE(!ret[0].dir);

            ret = run_find_roots(cfs, h, lo);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t == h * mid);
            REQUIRE(!ret[0].dir);

            REQUIRE(run_find_roots(cfs, h, hi).empty());
        };

        check({fp_t(1), fp_t(-6), fp_t(9)}, fp_t(1));
        check({fp_t(1), fp_t(6), fp_t(9)}, fp_t(-1));
    });
}

TEST_CASE("find_real_poly_roots undecided interval straddling lb real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots undecided interval straddling lb" test, plus the precision of
        // the roots.
        const auto check = [prec](const std::vector<real> &cfs, const real &h) {
            frpr_workspace<real> ws(2, prec);
            std::vector<real_poly_root<real>> ret;

            // Fetch the undecided interval from the isolation.
            const auto ivals = detail::isolate_real_roots(ws, std::span<const real>(cfs), h, real{0, prec});
            REQUIRE(ivals.size() == 1u);
            REQUIRE(ivals[0].type == detail::irr_interval_type::undecided);
            const auto lo = ivals[0].lo, hi = ivals[0].hi;
            // NOTE: div_2ui() keeps the precision.
            const auto mid = mppp::div_2ui(lo, 1u) + mppp::div_2ui(hi, 1u);
            REQUIRE(lo < mid);
            REQUIRE(mid < hi);

            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h, mid);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t.get_prec() == prec);
            REQUIRE(ret[0].t == h * mid);
            REQUIRE(!ret[0].dir);

            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h, lo);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t.get_prec() == prec);
            REQUIRE(ret[0].t == h * mid);
            REQUIRE(!ret[0].dir);

            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h, hi);
            REQUIRE(ret.empty());
        };

        check({r(1), r(-6), r(9)}, r(1));
        check({r(1), r(6), r(9)}, r(-1));
    }
}

TEST_CASE("find_real_poly_roots wide undecided interval")
{
    using Catch::Matchers::StartsWith;

    // p(t) = (t - 3)**6 with h = 10.1 (the polynomial of the "extended precision fallback" isolation test), and its
    // mirror image (t + 3)**6 with h = -10.1, which has the same rescaled polynomial. The isolation returns undecided
    // intervals wider than the minimum width, and find_real_poly_roots() throws.
    const std::vector cfs = {729., -1458., 1215., -540., 135., -18., 1.};
    const std::vector cfs_mirror = {729., 1458., 1215., 540., 135., 18., 1.};

    for (const auto &[c, h] : {std::pair{cfs, 10.1}, std::pair{cfs_mirror, -10.1}}) {
        REQUIRE_THROWS_AS(run_find_roots(c, h, 0.), std::runtime_error);
        REQUIRE_THROWS_WITH(run_find_roots(c, h, 0.),
                            StartsWith("find_real_poly_roots() was unable to isolate the real roots of a polynomial: "
                                       "the interval ["));
    }

    // With h = 10 (and h = -10 for the mirror image), all the computations in the isolation are exact, the root is
    // bisected down to the minimum width with an exact sign-change count, and it is reported as a single root with an
    // empty direction.
    //
    // NOTE: as in the "double root" tests, the root is reported at t = h*mid, where mid is the midpoint of the
    // undecided interval [lo, lo + 2**-53) containing tau = 3/10. The tolerance k = 4 has been measured by exact
    // computation of h*mid (the distance is at most 1 floating-point value).
    for (const auto &[c, h, root] : {std::tuple{cfs, 10., rat_t(3)}, std::tuple{cfs_mirror, -10., rat_t(-3)}}) {
        const auto ret = run_find_roots(c, h, 0.);
        REQUIRE(ret.size() == 1u);
        REQUIRE(within_steps(ret[0].t, root, 4));
        REQUIRE(!ret[0].dir);
    }
}

// The polynomials in the "merged undecided intervals" tests: (t - 5)(t - 8)(t - 9) with h = 10 (the polynomial of the
// "extended precision root at split point" isolation test), and its mirror image (t + 5)(t + 8)(t + 9) with h = -10,
// which has the same rescaled polynomial. The conversion to the Bernstein basis is inexact (it divides by 3), thus the
// value of the polynomial at tau = 1/2 is computed as an interval straddling zero in both isolation passes, and the
// root at tau = 1/2 ends up in the two adjacent undecided intervals [1/2 - 2**-p, 1/2) and [1/2, 1/2 + 2**-p). These
// are merged into a single undecided range whose midpoint is exactly 1/2: the root must be reported once, exactly at
// t = 5 (t = -5 for the mirror image), with an empty direction. The other two roots are found by the root finder.
//
// NOTE: the tolerance k = 160 for the other two roots has been measured by exact emulation (the maximum distance is
// 131 floating-point values).

TEST_CASE("find_real_poly_roots merged undecided intervals")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check that the polynomial with coefficients cfs has a root exactly at root0 with an empty
        // direction, followed by two roots within 160 floating-point values of root1 and root2, with directions -1
        // and 1.
        const auto check
            = [](const std::vector<fp_t> &cfs, fp_t h, const rat_t &root0, const rat_t &root1, const rat_t &root2) {
                  const auto ret = run_find_roots(cfs, h, fp_t(0));
                  REQUIRE(ret.size() == 3u);
                  REQUIRE(within_steps(ret[0].t, root0, 0));
                  REQUIRE(!ret[0].dir);
                  REQUIRE(within_steps(ret[1].t, root1, 160));
                  REQUIRE(ret[1].dir == -1);
                  REQUIRE(within_steps(ret[2].t, root2, 160));
                  REQUIRE(ret[2].dir == 1);
              };

        check({fp_t(-360), fp_t(157), fp_t(-22), fp_t(1)}, fp_t(10), rat_t(5), rat_t(8), rat_t(9));
        check({fp_t(360), fp_t(157), fp_t(22), fp_t(1)}, fp_t(-10), rat_t(-5), rat_t(-8), rat_t(-9));
    });
}

TEST_CASE("find_real_poly_roots merged undecided intervals real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots merged undecided intervals" test, plus the precision of the
        // roots.
        const auto check = [prec](const std::vector<real> &cfs, const real &h, const rat_t &root0, const rat_t &root1,
                                  const rat_t &root2) {
            frpr_workspace<real> ws(3, prec);
            std::vector<real_poly_root<real>> ret;
            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);

            REQUIRE(ret.size() == 3u);
            for (const auto &root : ret) {
                REQUIRE(root.t.get_prec() == prec);
            }
            REQUIRE(within_steps(ret[0].t, root0, 0));
            REQUIRE(!ret[0].dir);
            REQUIRE(within_steps(ret[1].t, root1, 160));
            REQUIRE(ret[1].dir == -1);
            REQUIRE(within_steps(ret[2].t, root2, 160));
            REQUIRE(ret[2].dir == 1);
        };

        check({r(-360), r(157), r(-22), r(1)}, r(10), rat_t(5), rat_t(8), rat_t(9));
        check({r(360), r(157), r(22), r(1)}, r(-10), rat_t(-5), rat_t(-8), rat_t(-9));
    }
}

// The polynomials in the "lb filter" tests are those of the "simple roots" tests: (t - 1)(t - 2)(2t - 5) with h = 3,
// and its mirror image (t + 1)(t + 2)(2t + 5) with h = -3. The values of lb are far from the roots (relative to the
// tolerance k = 40 of the "simple roots" tests):
//
// - lb = 0.5 excludes no root;
// - lb = 1.8 excludes the root at t = 1 (t = -1 for the mirror image);
// - lb = 2.1 excludes the roots at t = 1 and t = 2 (t = -1 and t = -2);
// - lb = 2.6 excludes all the roots.
//
// For lb = 1.8 and lb = 2.1, lb_tau lies strictly inside the isolating interval of the root at tau = 2/3 (on either
// side of the root), so that the exclusion of the root is decided after the root finding.

TEST_CASE("find_real_poly_roots lb filter")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        // Helper to check that the roots of the polynomial with coefficients cfs are, in order, within 40
        // floating-point values of the exact values in roots, with the given directions.
        const auto check
            = [](const std::vector<fp_t> &cfs, fp_t h, fp_t lb, const std::vector<std::pair<rat_t, int>> &roots) {
                  const auto ret = run_find_roots(cfs, h, lb);
                  REQUIRE(ret.size() == roots.size());
                  for (std::size_t i = 0; i < roots.size(); ++i) {
                      REQUIRE(within_steps(ret[i].t, roots[i].first, 40));
                      REQUIRE(ret[i].dir == roots[i].second);
                  }
              };

        // Helper to check that lb_tau = lb/|h| lies strictly inside an isolating interval.
        const auto check_straddle = [](const std::vector<fp_t> &cfs, fp_t h, fp_t lb) {
            using std::abs;

            const auto lb_tau = lb / abs(h);
            const auto ivals = run_isolation(cfs, h, lb_tau);
            REQUIRE(std::ranges::any_of(ivals, [&lb_tau](const auto &iv) {
                return iv.type == detail::irr_interval_type::isolating && iv.lo < lb_tau && lb_tau < iv.hi;
            }));
        };

        const std::vector cfs = {fp_t(-10), fp_t(19), fp_t(-11), fp_t(2)};
        const std::vector cfs_mirror = {fp_t(10), fp_t(19), fp_t(11), fp_t(2)};

        check(cfs, fp_t(3), fp_t(0.5), {{rat_t(1), 1}, {rat_t(2), -1}, {rat_t(5, 2), 1}});
        check(cfs, fp_t(3), fp_t(1.8), {{rat_t(2), -1}, {rat_t(5, 2), 1}});
        check(cfs, fp_t(3), fp_t(2.1), {{rat_t(5, 2), 1}});
        check(cfs, fp_t(3), fp_t(2.6), {});

        check(cfs_mirror, fp_t(-3), fp_t(0.5), {{rat_t(-1), 1}, {rat_t(-2), -1}, {rat_t(-5, 2), 1}});
        check(cfs_mirror, fp_t(-3), fp_t(1.8), {{rat_t(-2), -1}, {rat_t(-5, 2), 1}});
        check(cfs_mirror, fp_t(-3), fp_t(2.1), {{rat_t(-5, 2), 1}});
        check(cfs_mirror, fp_t(-3), fp_t(2.6), {});

        for (const auto lb : {fp_t(1.8), fp_t(2.1)}) {
            check_straddle(cfs, fp_t(3), lb);
            check_straddle(cfs_mirror, fp_t(-3), lb);
        }
    });
}

TEST_CASE("find_real_poly_roots lb filter real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](double v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots lb filter" test, plus the precision of the roots.
        const auto check = [prec](const std::vector<real> &cfs, const real &h, const real &lb,
                                  const std::vector<std::pair<rat_t, int>> &roots) {
            frpr_workspace<real> ws(3, prec);
            std::vector<real_poly_root<real>> ret;
            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h, lb);

            REQUIRE(ret.size() == roots.size());
            for (std::size_t i = 0; i < roots.size(); ++i) {
                REQUIRE(ret[i].t.get_prec() == prec);
                REQUIRE(within_steps(ret[i].t, roots[i].first, 40));
                REQUIRE(ret[i].dir == roots[i].second);
            }
        };

        const auto check_straddle = [prec](const std::vector<real> &cfs, const real &h, const real &lb) {
            frpr_workspace<real> ws(3, prec);

            const auto lb_tau = lb / abs(h);
            const auto ivals = detail::isolate_real_roots(ws, std::span<const real>(cfs), h, lb_tau);
            REQUIRE(std::ranges::any_of(ivals, [&lb_tau](const auto &iv) {
                return iv.type == detail::irr_interval_type::isolating && iv.lo < lb_tau && lb_tau < iv.hi;
            }));
        };

        const std::vector cfs = {r(-10), r(19), r(-11), r(2)};
        const std::vector cfs_mirror = {r(10), r(19), r(11), r(2)};

        check(cfs, r(3), r(0.5), {{rat_t(1), 1}, {rat_t(2), -1}, {rat_t(5, 2), 1}});
        check(cfs, r(3), r(1.8), {{rat_t(2), -1}, {rat_t(5, 2), 1}});
        check(cfs, r(3), r(2.1), {{rat_t(5, 2), 1}});
        check(cfs, r(3), r(2.6), {});

        check(cfs_mirror, r(-3), r(0.5), {{rat_t(-1), 1}, {rat_t(-2), -1}, {rat_t(-5, 2), 1}});
        check(cfs_mirror, r(-3), r(1.8), {{rat_t(-2), -1}, {rat_t(-5, 2), 1}});
        check(cfs_mirror, r(-3), r(2.1), {{rat_t(-5, 2), 1}});
        check(cfs_mirror, r(-3), r(2.6), {});

        for (const auto lb : {1.8, 2.1}) {
            check_straddle(cfs, r(3), r(lb));
            check_straddle(cfs_mirror, r(-3), r(lb));
        }
    }
}

// Linear polynomials whose root enclosure [nextdown(root), nextup(root)] straddles a bound of the range fall through to
// the general case:
//
// - p(t) = 3t and p(t) = -3t: the root is exactly at t = 0. The isolation reports it as a root exactly at tau = 0, and
//   it is reported at +0 for either sign of h, with the direction given by the sign of the derivative c1;
// - p(t) = t - 10 with h = 10, and its mirror image p(t) = t + 10 with h = -10: the root is exactly at t = h, outside
//   the range, and no root is reported.

TEST_CASE("find_real_poly_roots linear fall-through")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);
        using std::signbit;

        for (const auto h : {fp_t(10), fp_t(-10)}) {
            for (const auto c1 : {fp_t(3), fp_t(-3)}) {
                const auto ret = run_find_roots({fp_t(0), c1}, h, fp_t(0));
                REQUIRE(ret.size() == 1u);
                REQUIRE(ret[0].t == 0);
                REQUIRE(!signbit(ret[0].t));
                REQUIRE(ret[0].dir == ((c1 > 0) ? 1 : -1));
            }
        }

        REQUIRE(run_find_roots({fp_t(-10), fp_t(1)}, fp_t(10), fp_t(0)).empty());
        REQUIRE(run_find_roots({fp_t(10), fp_t(1)}, fp_t(-10), fp_t(0)).empty());
    });
}

TEST_CASE("find_real_poly_roots linear fall-through real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots linear fall-through" test, plus the precision of the roots.
        frpr_workspace<real> ws(1, prec);
        std::vector<real_poly_root<real>> ret;

        for (const auto &h : {r(10), r(-10)}) {
            for (const auto &c1 : {r(3), r(-3)}) {
                const std::vector cfs = {r(0), c1};
                find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);
                REQUIRE(ret.size() == 1u);
                REQUIRE(ret[0].t.get_prec() == prec);
                REQUIRE(ret[0].t == 0);
                REQUIRE(!signbit(ret[0].t));
                REQUIRE(ret[0].dir == ((c1 > 0) ? 1 : -1));
            }
        }

        const std::vector cfs1 = {r(-10), r(1)};
        find_real_poly_roots(ret, ws, std::span<const real>(cfs1), r(10));
        REQUIRE(ret.empty());

        const std::vector cfs2 = {r(10), r(1)};
        find_real_poly_roots(ret, ws, std::span<const real>(cfs2), r(-10));
        REQUIRE(ret.empty());
    }
}

// Workspace reuse in the general case: the results computed with a reused, copied or deserialised workspace must be
// identical to those computed with a fresh workspace. In particular, the rescaled polynomial used by the root finder
// must be the one of the current polynomial. The two polynomials, (t - 1)(t - 2)(2t - 5) with h = 3 (from the
// "find_real_poly_roots simple roots" test) and (t - 3)(t - 6)(t - 8.5) with h = 10 (from the "simple roots" isolation
// test), have different rescaled polynomials, and all their roots are found by the root finder. Every call also starts
// from a non-empty ret, which must be cleared out.

TEST_CASE("find_real_poly_roots workspace reuse")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        const std::vector cfs1 = {fp_t(-10), fp_t(19), fp_t(-11), fp_t(2)};
        const auto h1 = fp_t(3);
        const std::vector cfs2 = {fp_t(-153), fp_t(94.5), fp_t(-17.5), fp_t(1)};
        const auto h2 = fp_t(10);

        // The reference results, computed with fresh workspaces.
        const auto ref1 = run_find_roots(cfs1, h1, fp_t(0));
        const auto ref2 = run_find_roots(cfs2, h2, fp_t(0));
        REQUIRE(ref1.size() == 3u);
        REQUIRE(ref2.size() == 3u);

        // Helper to run find_real_poly_roots() with the workspace w, starting from a non-empty ret.
        const auto run = [](frpr_workspace<fp_t> &w, const std::vector<fp_t> &cfs, fp_t h) {
            std::vector<real_poly_root<fp_t>> ret(5);
            find_real_poly_roots(ret, w, std::span<const fp_t>(cfs), h);
            return ret;
        };

        // A workspace used for the two polynomials alternately.
        frpr_workspace<fp_t> ws(3);
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));
        REQUIRE(roots_equal(run(ws, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));

        // A copy of the workspace, and the original after the copy.
        auto ws_copy = ws;
        REQUIRE(roots_equal(run(ws_copy, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));

        // A deserialised workspace.
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oa(ss);
            oa << ws;
        }
        frpr_workspace<fp_t> ws_s11n(3);
        {
            boost::archive::binary_iarchive ia(ss);
            ia >> ws_s11n;
        }
        REQUIRE(roots_equal(run(ws_s11n, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws_s11n, cfs1, h1), ref1));
    });
}

TEST_CASE("find_real_poly_roots workspace reuse real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](double v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots workspace reuse" test.
        const std::vector cfs1 = {r(-10), r(19), r(-11), r(2)};
        const auto h1 = r(3);
        const std::vector cfs2 = {r(-153), r(94.5), r(-17.5), r(1)};
        const auto h2 = r(10);

        const auto run = [](frpr_workspace<real> &w, const std::vector<real> &cfs, const real &h) {
            std::vector<real_poly_root<real>> ret(5);
            find_real_poly_roots(ret, w, std::span<const real>(cfs), h);
            return ret;
        };

        frpr_workspace<real> ws_ref1(3, prec), ws_ref2(3, prec);
        const auto ref1 = run(ws_ref1, cfs1, h1);
        const auto ref2 = run(ws_ref2, cfs2, h2);
        REQUIRE(ref1.size() == 3u);
        REQUIRE(ref2.size() == 3u);

        frpr_workspace<real> ws(3, prec);
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));
        REQUIRE(roots_equal(run(ws, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));

        auto ws_copy = ws;
        REQUIRE(roots_equal(run(ws_copy, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws, cfs1, h1), ref1));

        std::stringstream ss;
        {
            boost::archive::binary_oarchive oa(ss);
            oa << ws;
        }
        frpr_workspace<real> ws_s11n(3, prec);
        {
            boost::archive::binary_iarchive ia(ss);
            ia >> ws_s11n;
        }
        REQUIRE(roots_equal(run(ws_s11n, cfs2, h2), ref2));
        REQUIRE(roots_equal(run(ws_s11n, cfs1, h1), ref1));
    }
}

// The polynomials in the "undecided interval with adjacent bounds" tests: (t - 3)**2 with h = 5, and its mirror image
// (t + 3)**2 with h = -5, which has the same rescaled polynomial. As in the "double root" tests, all the computations
// in the isolation are exact, and the double root at tau = 3/5 ends up in a single undecided interval [lo, hi) with
// hi - lo = 2**-p. Since 3/5 is in [1/2, 1), lo and hi are adjacent floating-point values, and the computed midpoint
// lo/2 + hi/2 is a tie, rounded to the value with an even significand. For the floating-point types tested here, this
// is hi (the integer floor(3 * 2**p / 5) is odd), which is outside the interval: the root must then be reported at
// t = h*lo. For mppp::real, this is the case for all the precisions tested except 11, for which the tie is rounded to
// lo (the root is again reported at t = h*lo).

TEST_CASE("find_real_poly_roots undecided interval with adjacent bounds")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        const auto check = [](const std::vector<fp_t> &cfs, fp_t h) {
            using std::nextafter;

            // Fetch the undecided interval from the isolation, and check that its bounds are adjacent and that the
            // computed midpoint is hi.
            const auto ivals = run_isolation(cfs, h, fp_t(0));
            REQUIRE(ivals.size() == 1u);
            REQUIRE(ivals[0].type == detail::irr_interval_type::undecided);
            const auto lo = ivals[0].lo, hi = ivals[0].hi;
            REQUIRE(nextafter(lo, hi) == hi);
            REQUIRE(lo / 2 + hi / 2 == hi);

            const auto ret = run_find_roots(cfs, h, fp_t(0));
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t == h * lo);
            REQUIRE(!ret[0].dir);
        };

        check({fp_t(9), fp_t(-6), fp_t(1)}, fp_t(5));
        check({fp_t(9), fp_t(6), fp_t(1)}, fp_t(-5));
    });
}

TEST_CASE("find_real_poly_roots undecided interval with adjacent bounds real")
{
    using mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const auto r = [prec](int v) { return real{v, prec}; };

        // Same checks as in the "find_real_poly_roots undecided interval with adjacent bounds" test (except for the
        // computed midpoint, see above), plus the precision of the root.
        const auto check = [prec](const std::vector<real> &cfs, const real &h) {
            frpr_workspace<real> ws(2, prec);
            std::vector<real_poly_root<real>> ret;

            const auto ivals = detail::isolate_real_roots(ws, std::span<const real>(cfs), h, real{0, prec});
            REQUIRE(ivals.size() == 1u);
            REQUIRE(ivals[0].type == detail::irr_interval_type::undecided);
            const auto lo = ivals[0].lo, hi = ivals[0].hi;
            REQUIRE(nextafter(lo, hi) == hi);

            find_real_poly_roots(ret, ws, std::span<const real>(cfs), h);
            REQUIRE(ret.size() == 1u);
            REQUIRE(ret[0].t.get_prec() == prec);
            REQUIRE(ret[0].t == h * lo);
            REQUIRE(!ret[0].dir);
        };

        check({r(9), r(-6), r(1)}, r(5));
        check({r(9), r(6), r(1)}, r(-5));
    }
}

// Precision of the roots at a working precision lower than the precision mp++ assigns to values created from an int (32
// bits), so that any int value leaking into the computation of a root would show up in its precision. The zero and
// linear polynomial cases are not covered at such low precision by the other real tests, nor is a nonzero lb with a
// precision different from the working one.
TEST_CASE("find_real_poly_roots low precision real")
{
    using mppp::real;

    const auto prec = 10;
    const auto r = [&](int v) { return real{v, prec}; };

    std::vector<real_poly_root<real>> ret;

    // Zero polynomial: a single root at t = 0.
    {
        frpr_workspace<real> ws(2, prec);
        const std::vector cfs = {r(0), r(0), r(0)};
        find_real_poly_roots(ret, ws, std::span<const real>(cfs), r(10));
        REQUIRE(ret.size() == 1u);
        REQUIRE(ret[0].t.get_prec() == prec);
    }

    // Linear polynomial p(t) = 3t - 2, with h = 10: a root at t = 2/3, computed by the linear case.
    {
        frpr_workspace<real> ws(1, prec);
        const std::vector cfs = {r(-2), r(3)};
        find_real_poly_roots(ret, ws, std::span<const real>(cfs), r(10));
        REQUIRE(ret.size() == 1u);
        REQUIRE(ret[0].t.get_prec() == prec);
    }

    // p(t) = (t - 1)(t - 2)(2t - 5) with h = 3, and lb = 1.5 with a precision of 200 bits: the roots at t = 2 and t
    // = 2.5 are reported.
    {
        frpr_workspace<real> ws(3, prec);
        const std::vector cfs = {r(-10), r(19), r(-11), r(2)};
        find_real_poly_roots(ret, ws, std::span<const real>(cfs), r(3), real{1.5, 200});
        REQUIRE(ret.size() == 2u);
        for (const auto &root : ret) {
            REQUIRE(root.t.get_prec() == prec);
        }
    }
}

// A test with the Taylor polynomials of an actual numerical integration. The harmonic oscillator x' = v, v' = -x is
// integrated in double precision from random initial conditions, step by step, until a step crosses x = 0. The
// polynomial of the event equation x = 0 over that step is the Taylor polynomial of x, whose coefficients are the
// first order + 1 Taylor coefficients of the step. Its root is compared with the exact solution
// x(t) = x0*cos(t) + v0*sin(t) = R*cos(t - phi), with R = hypot(x0, v0) and phi = atan2(v0, x0), whose roots are
// t = phi + pi/2 + k*pi, where the derivative -R*sin(t - phi) has the sign of (-1)**(k+1).
TEST_CASE("find_real_poly_roots harmonic oscillator")
{
    auto [x, v] = make_vars("x", "v");

    auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -x}, {0., 0.}};
    const auto order = ta.get_order();

    frpr_workspace<double> ws(order);
    std::vector<real_poly_root<double>> ret;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rdist(-1., 1.);

    for (auto i = 0; i < 100; ++i) {
        // Random initial conditions at t = 0, with an amplitude R of at least 0.5.
        //
        // NOTE: the error in the time of the crossing is of the order of the integration error in x divided by
        // R, the magnitude of the derivative at the crossing. Keeping R away from zero keeps it small.
        double x0 = 0, v0 = 0;
        do {
            x0 = rdist(rng);
            v0 = rdist(rng);
        } while (std::hypot(x0, v0) < .5);

        ta.get_state_data()[0] = x0;
        ta.get_state_data()[1] = v0;
        ta.set_time(0.);

        // Step, recording the Taylor coefficients, until a step crosses x = 0.
        while (true) {
            const auto x_start = ta.get_state()[0];
            REQUIRE(std::get<0>(ta.step(true)) == taylor_outcome::success);
            if ((x_start < 0) != (ta.get_state()[0] < 0)) {
                break;
            }
        }

        // Find the root of the Taylor polynomial of x over the last step, which started at t_start and has size h.
        const auto h = ta.get_last_h();
        const auto t_start = ta.get_time() - h;
        find_real_poly_roots(ret, ws, std::span<const double>(ta.get_tc().data(), order + 1u), h);

        // Consecutive roots of the exact solution are pi apart: if the step is shorter than pi, x cannot come back to
        // zero within the step, and the crossing is the only root. Otherwise, since x changes sign over the step, the
        // number of roots is odd.
        if (h < std::numbers::pi) {
            REQUIRE(ret.size() == 1u);
        } else {
            REQUIRE(ret.size() % 2u == 1u);
        }

        // Compare each root with the nearest root of the exact solution.
        //
        // NOTE: the difference is dominated by the error of the numerical integration, of the order of eps / R after
        // the few steps taken. 1e-12 is a generous bound.
        const auto phi = std::atan2(v0, x0);
        for (const auto &root : ret) {
            const auto t_root = t_start + root.t;
            const auto k = std::round((t_root - phi - std::numbers::pi / 2) / std::numbers::pi);
            const auto t_exact = phi + std::numbers::pi / 2 + k * std::numbers::pi;
            REQUIRE(std::abs(t_root - t_exact) < 1e-12);

            // The direction of the root is the sign of (-1)**(k+1).
            REQUIRE(root.dir == ((static_cast<long long>(k) % 2 == 0) ? -1 : 1));
        }
    }
}
