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

// Check if ivals contains the entry for a root exactly at x.
template <typename T>
bool has_root_at(const ivals_t<T> &ivals, T x)
{
    return std::ranges::any_of(ivals, [x](const auto &iv) {
        return iv.type == detail::irr_interval_type::isolating && iv.lo == x && iv.hi == x && iv.sign == 0;
    });
}

// Check that ivals is a correct output for a polynomial whose real roots in [0, 1) are exactly the distinct values in
// roots, each paired with the sign of the derivative at the root:
//
// - an isolating interval (lo, hi) with lo < hi contains exactly one root, and its sign is the sign of that root;
// - an entry with lo == hi is exactly at a root, and its sign is zero (the sign paired with the root is not used);
// - every root is covered by an isolating interval, by an entry with lo == hi or by an undecided interval [lo, hi).
template <typename T>
void check_correct(const ivals_t<T> &ivals, const std::vector<std::pair<rat_t, int>> &roots)
{
    using detail::irr_interval_type;

    for (const auto &iv : ivals) {
        if (iv.type != irr_interval_type::isolating) {
            continue;
        }

        const auto lo = to_rat(iv.lo), hi = to_rat(iv.hi);

        if (lo == hi) {
            REQUIRE(std::ranges::count_if(roots, [&](const auto &r) { return r.first == lo; }) == 1);
            REQUIRE(iv.sign == 0);
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
    // p(t) = t*(t - 3), roots at tau = 0 and tau = 3/10.
    const std::vector cfs = {0., -3., 1.};

    auto ivals = run_isolation(cfs, 10., 0.);

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0.));
    check_correct(ivals, {{rat_t(0), 0}, {rat_t(3, 10), 1}});

    // With a cooldown, the root at zero is not reported.
    ivals = run_isolation(cfs, 10., 0.1);

    REQUIRE(ivals.size() == 1u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(!has_root_at(ivals, 0.));
    check_correct(ivals, {{rat_t(3, 10), 1}});
}

TEST_CASE("root at split point")
{
    // p(t) = (t - 5)(t - 8), roots at tau = 1/2 (the first bisection point) and tau = 4/5. All the interval
    // computations in the isolation are exact for this polynomial, so that its value at tau = 1/2 is computed as
    // exactly [0, 0] and the root is reported as a root at the lower bound of [1/2, 3/4).
    const auto ivals = run_isolation({40., -13., 1.}, 10., 0.);

    REQUIRE(ivals.size() == 2u);
    REQUIRE(!has_undecided(ivals));
    REQUIRE(has_root_at(ivals, 0.5));
    check_correct(ivals, {{rat_t(1, 2), 0}, {rat_t(4, 5), 1}});
}

TEST_CASE("no roots")
{
    // p(t) = t**2 + 1.
    const auto ivals = run_isolation({1., 0., 1.}, 1., 0.);

    REQUIRE(ivals.empty());
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
    // p(t) = (3t - 1)(3t - 1 - delta) = 9t**2 - (6 + 3*delta)t + 1 + delta, with delta = 2**-50. The coefficients are
    // exactly representable, and the roots 1/3 and (1 + delta)/3 are about 1.3 epsilons apart. The isolation is not
    // expected to separate them: they will likely end up in undecided intervals, because the sign-change count becomes
    // inexact or because the minimum width is reached. We only check that the output is correct.
    const auto delta = std::ldexp(1., -50);
    const auto ivals = run_isolation({1. + delta, -(6. + 3. * delta), 9.}, 1., 0.);

    // The derivative is negative at the smaller root and positive at the larger one.
    check_correct(ivals, {{rat_t(1, 3), -1}, {(1 + to_rat(delta)) / 3, 1}});
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
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, inf), std::invalid_argument,
                           Message("The h value passed to find_real_poly_roots() must be finite, but a value of inf was "
                                   "passed instead"));
    REQUIRE_THROWS_MATCHES(find_real_poly_roots(ret, ws, span, nan), std::invalid_argument,
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
