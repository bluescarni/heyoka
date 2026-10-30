// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <heyoka/find_real_poly_roots.hpp>

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
