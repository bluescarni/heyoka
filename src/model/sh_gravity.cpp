// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include <fmt/core.h>

#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/sh_gravity.hpp>

namespace heyoka::model::detail
{

namespace
{

// Algorithm for the computation of the integer square root via binary search. See:
//
// https://en.wikipedia.org/wiki/Integer_square_root#Algorithm_using_binary_search
//
// NOTE: internally the algorithm uses 64 bit integers in order to avoid overflows.
std::uint32_t isqrt(const std::uint32_t y)
{
    // Lower bound of the square root.
    std::uint64_t L = 0;
    // Upper bound of the square root.
    auto R = static_cast<std::uint64_t>(y) + 1u;

    while (L != R - 1u) {
        // Midpoint to test.
        const auto M = (L + R) / 2u;

        if (M * M <= y) {
            L = M;
        } else {
            R = M;
        }
    }

    return static_cast<std::uint32_t>(L);
}

} // namespace

// Helper to construct an S/C coefficients getter for a custom spherical harmonics gravity model from a list of S/C
// coefficients.
//
// The coefficients are expected to be stored in S/C pairs as a flattened list up to degree n:
//
// [(C00, S00), (C10, S10), (C11, S11), (C20, S20), (C21, S21), (C22, S22), ..., (Cnn, Snn)]
//
// NOTE: the returned std::function captures sc_list by reference, and thus cannot be used after the destruction of
// sc_list.
[[nodiscard]] sh_gravity_sc_getter_t
sh_gravity_sc_getter_from_list(const std::vector<std::array<expression, 2>> &sc_list)
{
    const auto sc_list_size = sc_list.size();

    if (sc_list_size == 0u) [[unlikely]] {
        throw std::invalid_argument(
            "A custom spherical harmonics gravity model cannot be created from an empty list of S/C coefficients");
    }

    // NOTE: the size of sc_list is valid iff it is equal to n*(n+1)/2 for some integer n>0:
    //
    // n*(n+1)/2 = sc_list_size
    //
    // By solving this quadratic equation for n, we obtain the discriminant 8*sc_list_size+1 which must be a perfect
    // square in order for the solution of the quadratic equation to be an integer:
    //
    // n = (sqrt(8*sc_list_size+1) - 1)/2.
    //
    // n is the maximum spherical harmonics degree plus one.

    // Compute the discriminant.
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
    const auto discr = 8 * safe_uint32_t(sc_list_size) + 1;

    // Check if it is a perfect square via isqrt().
    const auto isqrt_discr = isqrt(discr);
    if (isqrt_discr * isqrt_discr != discr) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid custom spherical harmonics gravity model: the list of "
                        "coefficients has a size of {}, which is not equal to n*(n+1)/2 for any natural number n",
                        sc_list_size));
    }

    // Now we can compute max_n - the maximum spherical harmonics degree of sc_list.
    const auto max_n = ((isqrt_discr - 1u) / 2u) - 1u;

    return [&sc_list, max_n](const std::uint32_t n, const std::uint32_t m) {
        using safe_size_t = boost::safe_numerics::safe<decltype(sc_list.size())>;

        // NOTE: this check is already run in the implementation functions.
        assert(m <= n);

        if (n > max_n) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Invalid degree {} specified for a custom spherical harmonics "
                                                    "gravity model: the maximum supported degree is {}",
                                                    n, max_n));
        }

        const auto idx = (n * (safe_size_t(n) + 1) / 2) + m;
        assert(idx < sc_list.size());
        return sc_list[idx];
    };
}

namespace
{

// Common checks for the sh_gravity_*_impl() functions.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void sh_gravity_impl_common_checks(const std::uint32_t n, const std::uint32_t m)
{
    if (n == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::overflow_error("Overflow detected during the construction of a spherical harmonics gravity model");
        // LCOV_EXCL_STOP
    }

    if (m > n) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid combination of degree and order specified for a "
                        "spherical harmonics gravity model: the order {} is greater than the degree {}",
                        m, n));
    }
}

// Kronecker delta.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto sh_gravity_impl_kdelta(const std::uint32_t n, const std::uint32_t m)
{
    return static_cast<double>(n == m);
}

// This helper will generate all the V/W terms necessary for the computation of the spherical harmonics gravity
// potential/acceleration up to V_nn and W_nn as a function of the body-centric Cartesian coordinates. The terms are
// generated via the recursive algorithm explained in Montenbruck, 3.2.4, with one important deviation: since we are
// assuming that normalised C/S coefficients are being employed, we have to adapt the recursive algorithm to produce the
// normalised counterparts of the V/W terms. See the notebook in the tools/ directory for a derivation (but it is just a
// matter of adapting certain numerical factors in the recursion).
//
// NOTE: the numerical factors in the recursion formulae are implemented in hard-coded double-precision arithmetic. As a
// consequence, attempting to instantiate spherical harmonics gravity models with precision higher than double will
// produce inaccurate results. Pragmatically, we expect double precision to be sufficient for typical usages of gravity
// models, but if this ever becomes an issue in the future we can always introduce the numerical coefficients as custom
// nullary constant functions.
auto sh_gravity_impl_make_rec_map(const std::uint32_t max_n, const expression &xa_r2, const expression &ya_r2,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  const expression &za_r2, const expression &a2_r2, const expression &a_r)
{
    if (max_n == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::overflow_error("Overflow detected during the construction of a spherical harmonics gravity model");
        // LCOV_EXCL_STOP
    }

    // Seed the recursion map with the V00/W00 term.
    boost::unordered_flat_map<std::array<std::uint32_t, 2>, std::array<expression, 2>> rec_map;
    rec_map.try_emplace({0, 0}, std::array{a_r, 0_dbl});

    // Run the recursion.
    //
    // NOTE: we use '<' to stop the iteration because at the last iteration we only need the seeding to generate
    // V_nn/W_nn, which is performed as the last step of the second-to-last iteration.
    for (std::uint32_t m = 0; m < max_n; ++m) {
        for (std::uint32_t n = m + 1u; n <= max_n; ++n) {
            // Compute the quantities for the first part of the recursion formula.
            assert(n >= 1u);
            const auto &[V_nm1_m, W_nm1_m] = rec_map.at({n - 1u, m});
            const auto F1 = std::sqrt(((2. * n) + 1) * ((2. * n) - 1)
                                      / ((static_cast<double>(n) - m) * (static_cast<double>(n) + m)));

            // Compute the first part of the recursion formula.
            auto V_nm = F1 * za_r2 * V_nm1_m;
            auto W_nm = F1 * za_r2 * W_nm1_m;

            // NOTE: we have to ignore the second part of the recursion formula if n == m + 1, otherwise we get a
            // nonsensical term. See also eq. (17) in the Cunningham paper.
            if (n != m + 1u) {
                // Compute the quantities for the second part of the recursion formula.
                assert(n >= 2u);
                const auto &[V_nm2_m, W_nm2_m] = rec_map.at({n - 2u, m});
                const auto F2
                    = std::sqrt(((2. * n) + 1) * (static_cast<double>(n) - m - 1) * (static_cast<double>(n) + m - 1)
                                / ((static_cast<double>(n) - m) * (static_cast<double>(n) + m) * ((2. * n) - 3)));

                // Add the second part of the recursion formula.
                V_nm -= F2 * a2_r2 * V_nm2_m;
                W_nm -= F2 * a2_r2 * W_nm2_m;
            }

            // Add V_nm/W_nm.
            assert(!rec_map.contains({n, m}));
            rec_map.try_emplace({n, m}, std::array{V_nm, W_nm});
        }

        // Seed the starting terms for the next iteration of m. These are V_(m+1,m+1)/W_(m+1,m+1).
        const auto &[V_mm, W_mm] = rec_map.at({m, m});
        const auto mp1 = m + 1u;
        const auto F = std::sqrt((2. - sh_gravity_impl_kdelta(0, mp1)) * ((2. * mp1) + 1)
                                 / (2. * mp1 * (2. - sh_gravity_impl_kdelta(0, m))));
        const auto V_mp1_mp1 = F * ((xa_r2 * V_mm) - (ya_r2 * W_mm));
        const auto W_mp1_mp1 = F * ((xa_r2 * W_mm) + (ya_r2 * V_mm));
        assert(!rec_map.contains({mp1, mp1}));
        rec_map.try_emplace({mp1, mp1}, std::array{V_mp1_mp1, W_mp1_mp1});
    }

#if !defined(NDEBUG)

    // Debug checks.
    decltype(rec_map.size()) n_found = 0;
    for (std::uint32_t m = 0; m <= max_n; ++m) {
        for (std::uint32_t n = m; n <= max_n; ++n) {
            assert(rec_map.contains({n, m}));

            if (m == 0u) {
                assert(rec_map.at({n, m})[1] == 0_dbl);
            }

            ++n_found;
        }
    }
    assert(n_found == rec_map.size());

#endif

    return rec_map;
} // LCOV_EXCL_LINE

} // namespace

expression sh_gravity_pot_impl(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m,
                               const expression &mu, const expression &a, const sh_gravity_sc_getter_t &sc_get)
{
    assert(sc_get);

    // Check n/m.
    sh_gravity_impl_common_checks(n, m);

    // Pre-compute several quantities.
    const auto &[x, y, z] = xyz;
    const auto r2 = sum({pow(x, 2.), pow(y, 2.), pow(z, 2.)});
    const auto a_r2 = a / r2;
    const auto xa_r2 = x * a_r2;
    const auto ya_r2 = y * a_r2;
    const auto za_r2 = z * a_r2;
    const auto a2_r2 = a * a_r2;
    const auto a_r = a / sqrt(r2);
    const auto mu_a = mu / a;

    // Generate the recursion for the V/W terms.
    const auto rec_map = sh_gravity_impl_make_rec_map(n, xa_r2, ya_r2, za_r2, a2_r2, a_r);

    // Assemble the terms of the summation.
    std::vector<expression> terms;
    for (std::uint32_t i = 0; i <= n; ++i) {
        // NOTE: in order to generate the full potential, we would iterate j in the [0, i] range here. However, we allow
        // to stop the iteration at a order m < i, hence the iteration range here is [0, min(m, i)].
        for (std::uint32_t j = 0; j <= std::min(m, i); ++j) {
            const auto [C, S] = sc_get(i, j);
            const auto &[V, W] = rec_map.at({i, j});

            terms.push_back((C * V) + (S * W));
        }
    }

    return mu_a * sum(std::move(terms));
}

// NOTE: the computation of the acceleration is adapted from Montenbruck 3.2.5, with modifications due to the use of
// normalised S/C coefficients.
//
// NOTE: the numerical factors in the recursion formulae are implemented in hard-coded double-precision arithmetic. As a
// consequence, attempting to instantiate spherical harmonics gravity models with precision higher than double will
// produce inaccurate results. Pragmatically, we expect double precision to be sufficient for typical usages of gravity
// models, but if this ever becomes an issue in the future we can always introduce the numerical coefficients as custom
// nullary constant functions.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> sh_gravity_acc_impl(const std::array<expression, 3> &xyz, const std::uint32_t n,
                                              // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                              const std::uint32_t m, const expression &mu, const expression &a,
                                              const sh_gravity_sc_getter_t &sc_get)
{
    assert(sc_get);

    // Check n/m.
    sh_gravity_impl_common_checks(n, m);

    // Pre-compute several quantities.
    const auto &[x, y, z] = xyz;
    const auto r2 = sum({pow(x, 2.), pow(y, 2.), pow(z, 2.)});
    const auto a_r2 = a / r2;
    const auto xa_r2 = x * a_r2;
    const auto ya_r2 = y * a_r2;
    const auto za_r2 = z * a_r2;
    const auto a2_r2 = a * a_r2;
    const auto a_r = a / sqrt(r2);
    const auto mu_a2 = mu / pow(a, 2.);

    // Generate the recursion for the V/W terms.
    const auto rec_map = sh_gravity_impl_make_rec_map(n + 1u, xa_r2, ya_r2, za_r2, a2_r2, a_r);

    // Initialise the vectors storing the summation terms for the accelerations.
    std::vector<expression> x_terms, y_terms, z_terms;

    // Assemble the terms of the summation.
    for (std::uint32_t i = 0; i <= n; ++i) {
        // NOTE: in order to generate the full accelerations, we would iterate j in the [0, i] range here. However, we
        // allow to stop the iteration at a order m < i, hence the iteration range here is [0, min(m, i)].
        for (std::uint32_t j = 0; j <= std::min(m, i); ++j) {
            const auto [C, S] = sc_get(i, j);

            // Compute the numerical coefficients.
            //
            // NOTE: these differ from the original formulae due to the use of normalised coefficients.
            auto cxy_0 = std::sqrt((2. - sh_gravity_impl_kdelta(0, j)) * ((2. * i) + 1) * (2. + i + j) * (1. + i + j)
                                   / ((2. - sh_gravity_impl_kdelta(0, j + 1u)) * ((2. * i) + 3)));
            const auto cz = (1. + i - j) * std::sqrt((1. + i + j) * ((2. * i) + 1) / (((2. * i) + 3) * (1. + i - j)));

            // NOTE: the x and y terms have special-casing for m == 0.
            if (j == 0u) {
                const auto &[V, W] = rec_map.at({i + 1u, 1});

                x_terms.push_back(-C * cxy_0 * V);
                y_terms.push_back(-C * cxy_0 * W);
            } else {
                cxy_0 *= 0.5;
                const auto cxy_1 = 0.5 * (2. + i - j) * (1. + i - j)
                                   * std::sqrt((2. - sh_gravity_impl_kdelta(0, j)) * ((2. * i) + 1)
                                               / ((2. - sh_gravity_impl_kdelta(0, j - 1u)) * ((2. * i) + 3)
                                                  * (2. + i - j) * (1. + i - j)));

                const auto &[Vp1, Wp1] = rec_map.at({i + 1u, j + 1u});
                const auto &[Vm1, Wm1] = rec_map.at({i + 1u, j - 1u});

                x_terms.insert(x_terms.end(), {-C * cxy_0 * Vp1, -S * cxy_0 * Wp1, C * cxy_1 * Vm1, S * cxy_1 * Wm1});
                y_terms.insert(y_terms.end(), {-C * cxy_0 * Wp1, S * cxy_0 * Vp1, -C * cxy_1 * Wm1, S * cxy_1 * Vm1});
            }

            const auto &[V, W] = rec_map.at({i + 1u, j});
            z_terms.insert(z_terms.end(), {-C * cz * V, -S * cz * W});
        }
    }

    return {mu_a2 * sum(std::move(x_terms)), mu_a2 * sum(std::move(y_terms)), mu_a2 * sum(std::move(z_terms))};
}

} // namespace heyoka::model::detail
