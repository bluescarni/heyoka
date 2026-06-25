// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/egm2008.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/model/sh_gravity.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

// Fetch the array of C/S coefficients for the EGM2008 model.
mdspan<const double, extents<std::size_t, std::dynamic_extent, 2>> get_egm2008_CS() noexcept
{
    assert(std::ranges::size(detail::egm2008_CS) % 2u == 0u);
    return mdspan<const double, extents<std::size_t, std::dynamic_extent, 2>>{
        detail::egm2008_CS, std::ranges::size(detail::egm2008_CS) / 2u};
}

namespace detail
{

namespace
{

// Helper to fetch the EGM2008 C/S coefficients for a specific degree and order.
std::array<expression, 2> get_egm2008_cs(const std::uint32_t n, const std::uint32_t m)
{
    // NOTE: this check is performed in the pot/acc implementation functions.
    assert(m <= n);

    // NOTE: the coefficients in the dataset begin from n=2.
    if (n == 0u) {
        return {1_dbl, 0_dbl};
    }

    if (n == 1u) {
        return {0_dbl, 0_dbl};
    }

    // Fetch the span over the coefficients.
    const auto cs_span = get_egm2008_CS();

    const auto start_n_idx = ((n * (n + 1u)) / 2u) - 3u;
    const auto idx = start_n_idx + m;

    // NOTE: the degree is validated in the pot/acc implementation functions, hence here 'idx' is guaranteed to be in
    // range (and, as a consequence, 'n' is small enough that the computation above cannot overflow).
    assert(idx < cs_span.extent(0));

    return {expression{cs_span[idx, 0]}, expression{cs_span[idx, 1]}};
}

// Common checks for the egm2008_*() functions.
//
// NOTE: these checks are partially redundant with those performed in sh_gravity.cpp, but we want to keep them in order
// to provide improved error messages over the generic ones from sh_gravity.cpp.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void egm2008_common_checks(const std::uint32_t n, const std::uint32_t m)
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

} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
expression egm2008_pot_impl(const std::array<expression, 3> &xyz, const std::uint32_t n, const std::uint32_t m,
                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                            const expression &mu, const expression &a)
{
    // Check degree/order before delegating.
    egm2008_common_checks(n, m);

    return sh_gravity_pot_impl(xyz, n, m, mu, a, &get_egm2008_cs);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<expression, 3> egm2008_acc_impl(const std::array<expression, 3> &xyz, const std::uint32_t n,
                                           const std::uint32_t m,
                                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                           const expression &mu, const expression &a)
{
    // Check degree/order before delegating.
    egm2008_common_checks(n, m);

    return sh_gravity_acc_impl(xyz, n, m, mu, a, &get_egm2008_cs);
}

} // namespace detail

// Default values of the gravitational parameter 'mu' and Earth radius 'a' for the egm2008_*() functions.
//
// NOTE: these are in SI units, taken from the official documentation of EGM2008.
double get_egm2008_mu() noexcept
{
    return 3986004.415e8;
}

double get_egm2008_a() noexcept
{
    return 6378136.3;
}

} // namespace model

HEYOKA_END_NAMESPACE
