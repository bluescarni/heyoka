// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <ranges>
#include <stdexcept>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/egm2008.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/egm2008.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

// Helper to fetch the EGM2008 S/C coefficients for a specific
// degree and order.
std::array<double, 2> get_egm2008_sc(std::uint32_t n, std::uint32_t m)
{
    assert(m <= n);

    // NOTE: the coefficients in the dataset begin from n=2.
    if (n == 0u) {
        return {1, 0};
    }

    if (n == 1u) {
        return {0, 0};
    }

    const auto start_n_idx = ((n * (n + 1u)) / 2u) - 3u;
    const auto idx = start_n_idx + m;

    assert(idx < std::ranges::size(egm2008_CS));

    return {egm2008_CS[idx][0], egm2008_CS[idx][1]};
}

// Common checks for the egm2008_*() functions.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void egm2008_common_checks(std::uint32_t n, std::uint32_t m)
{
    if (n > egm2008_max_degree) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid degree specified for the EGM2008 geopotential model: the maximum degree is "
                        "{}, but a degree of {} was specified",
                        egm2008_max_degree, n));
    }

    if (m > n) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid combination of degree and order specified for the EGM2008 "
                                                "geopotential model: the order {} is greater than the degree {}",
                                                m, n));
    }
}

// Kronecker delta.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto egm2008_kdelta(std::uint32_t n, std::uint32_t m)
{
    return static_cast<double>(n == m);
}

// This helper will generate all the V/W terms necessary for the computation of the
// geopotential up to V_nn and W_nn as a function of the geocentric Cartesian coordinates.
// The terms are generated via the recursive algorithm
// explained in Montenbruck, 3.2.4, with one important deviation: since the EGM2008 model
// provides the *normalised* C/S coefficients, we have to adapt the recursive algorithm
// to produce the normalised counterparts of the V/W terms. See the notebook in the tools/
// directory for a derivation (but it is just a matter of adapting certain numerical factors
// in the recursion).
//
// NOTE: apart from the renormalisation bits, there is nothing in this algorithm which
// is specific to the EGM2008 model. That is, this should be usable with small modifications
// in the implementation of other geopotential models.
auto egm2008_make_rec_map(std::uint32_t max_n, const expression &xa_r2, const expression &ya_r2,
                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                          const expression &za_r2, const expression &a2_r2, const expression &a_r)
{
    // Seed the recursion map with the V00/W00 term.
    boost::unordered_flat_map<std::array<std::uint32_t, 2>, std::array<expression, 2>> rec_map;
    rec_map.try_emplace({0, 0}, std::array{a_r, 0_dbl});

    // Run the recursion.
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

            // NOTE: we have to ignore the second part of the recursion formula
            // if n == m + 1, otherwise we get a nonsensical term. See also eq. (17)
            // in the Cunningham paper.
            if (n != m + 1u) {
                // Compute the quantities for the second part of the recursion formula.
                assert(n >= 2u);
                const auto &[V_nm2_m, W_nm2_m] = rec_map.at({n - 2u, m});
                const auto F2
                    = std::sqrt(((2. * n) + 1) * (static_cast<double>(n) - m - 1) * (static_cast<double>(n) + m - 1)
                                / ((static_cast<double>(n) - m) * (static_cast<double>(n) + m) * (2. * n - 3)));

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
        const auto F
            = std::sqrt((2. - egm2008_kdelta(0, mp1)) * (2. * mp1 + 1) / (2. * mp1 * (2. - egm2008_kdelta(0, m))));
        const auto V_mp1_mp1 = F * (xa_r2 * V_mm - ya_r2 * W_mm);
        const auto W_mp1_mp1 = F * (xa_r2 * W_mm + ya_r2 * V_mm);
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
expression egm2008_pot_impl(const std::array<expression, 3> &xyz, std::uint32_t n, std::uint32_t m,
                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                            const expression &mu, const expression &a)
{
    // Check degree/order.
    egm2008_common_checks(n, m);

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
    const auto rec_map = egm2008_make_rec_map(n, xa_r2, ya_r2, za_r2, a2_r2, a_r);

    // Assemble the terms of the summation.
    std::vector<expression> terms;
    for (std::uint32_t i = 0; i <= n; ++i) {
        // NOTE: in order to generate the full potential, we would iterate
        // j in the [0, i] range here. However, we allow to stop the iteration
        // at a order m < i, hence the iteration range here is [0, min(m, i)].
        for (std::uint32_t j = 0; j <= std::min(m, i); ++j) {
            const auto [C, S] = get_egm2008_sc(i, j);
            const auto &[V, W] = rec_map.at({i, j});

            terms.push_back(C * V + S * W);
        }
    }

    return mu_a * sum(std::move(terms));
}

// NOTE: the computation of the acceleration is adapted from Montenbruck 3.2.5, with the
// modifications to the formulae due to the use of normalised coefficients in EGM2008.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> egm2008_acc_impl(const std::array<expression, 3> &xyz, std::uint32_t n, std::uint32_t m,
                                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                           const expression &mu, const expression &a)
{
    // Check degree/order.
    egm2008_common_checks(n, m);

    // NOTE: in the Cunningham recursion formulae for the acceleration, we need V/W terms
    // up to degree/order n+1.
    if (n == egm2008_max_degree) [[unlikely]] {
        throw std::invalid_argument(
            // LCOV_EXCL_START
            fmt::format("Invalid degree specified for the computation of the acceleration in "
                        "the EGM2008 geopotential model: a degree of {} is too high, the maximum allowed value is {}",
                        // LCOV_EXCL_STOP
                        n, egm2008_max_degree - 1u));
    }

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
    const auto rec_map = egm2008_make_rec_map(n + 1u, xa_r2, ya_r2, za_r2, a2_r2, a_r);

    // Initialise the vectors storing the summation terms for the accelerations.
    std::vector<expression> x_terms, y_terms, z_terms;

    // Assemble the terms of the summation.
    for (std::uint32_t i = 0; i <= n; ++i) {
        // NOTE: in order to generate the full accelerations, we would iterate
        // j in the [0, i] range here. However, we allow to stop the iteration
        // at a order m < i, hence the iteration range here is [0, min(m, i)].
        for (std::uint32_t j = 0; j <= std::min(m, i); ++j) {
            const auto [C, S] = get_egm2008_sc(i, j);

            // Compute the numerical coefficients.
            // NOTE: these differ from the original formulae due to the use normalised geopotential coefficients.
            auto cxy_0 = std::sqrt((2. - egm2008_kdelta(0, j)) * (2. * i + 1) * (2. + i + j) * (1. + i + j)
                                   / ((2. - egm2008_kdelta(0, j + 1u)) * (2. * i + 3)));
            const auto cz = (1. + i - j) * std::sqrt((1. + i + j) * (2. * i + 1) / ((2. * i + 3) * (1. + i - j)));

            // NOTE: the x and y terms have special-casing for m == 0.
            if (j == 0u) {
                const auto &[V, W] = rec_map.at({i + 1u, 1});

                x_terms.push_back(-C * cxy_0 * V);
                y_terms.push_back(-C * cxy_0 * W);
            } else {
                cxy_0 *= 0.5;
                const auto cxy_1
                    = 0.5 * (2. + i - j) * (1. + i - j)
                      * std::sqrt((2. - egm2008_kdelta(0, j)) * (2. * i + 1)
                                  / ((2. - egm2008_kdelta(0, j - 1u)) * (2. * i + 3) * (2. + i - j) * (1. + i - j)));

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

} // namespace model::detail

HEYOKA_END_NAMESPACE
