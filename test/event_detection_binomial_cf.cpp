// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// Tests for ed_make_ival_binomial_cf_like().
//
// ed_make_ival_binomial_cf_like() computes an interval [lo, hi] enclosing the binomial coefficient C(n, k). The tests
// compare the result with the exact value: if C(n, k) is representable, lo and hi must both be equal to C(n, k),
// otherwise lo and hi must be the two consecutive values enclosing C(n, k). The tests also check that an exception is
// thrown when lo or hi overflows.

using namespace heyoka;
using namespace heyoka_test;

using rat = boost::multiprecision::cpp_rational;
using boost::multiprecision::cpp_int;
using Catch::Matchers::Message;

// The floating-point types tested.
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

// The largest n tested.
constexpr std::uint32_t n_max = 140;

// The rows 0, 1, ..., n_max of Pascal's triangle: the element k of the row n is C(n, k).
std::vector<std::vector<cpp_int>> pascal_triangle()
{
    std::vector<std::vector<cpp_int>> ret{{1}};

    for (std::uint32_t n = 1; n <= n_max; ++n) {
        const auto &prev = ret.back();

        std::vector<cpp_int> row(n + 1u, 1);
        for (std::uint32_t k = 1; k < n; ++k) {
            row[k] = prev[k - 1u] + prev[k];
        }

        ret.push_back(std::move(row));
    }

    return ret;
}

const auto binomials = pascal_triangle();

// The error message of the exception thrown when C(n, k) overflows the target type.
std::string overflow_message(std::uint32_t n, std::uint32_t k)
{
    return fmt::format("The computation of a binomial coefficient with arguments ({}, {}) overflows the target "
                       "floating-point type in ed_make_ival_binomial_cf_like()",
                       n, k);
}

// Check the interval computed by ed_make_ival_binomial_cf_like() for C(n, k). x is passed to
// ed_make_ival_binomial_cf_like(), and p is the number of significand bits of x's type.
template <typename T>
void check_binomial(std::uint32_t n, std::uint32_t k, const T &x, std::uint32_t p)
{
    using std::nextafter;

    const auto &C_int = binomials[n][k];
    const rat C(C_int);
    const auto [lo, hi] = detail::ed_make_ival_binomial_cf_like(n, k, x);

#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::same_as<T, mppp::real>) {
        // The results must have the precision of x.
        REQUIRE(lo.get_prec() == x.get_prec());
        REQUIRE(hi.get_prec() == x.get_prec());
    }
#endif

    // C(n, k) is representable if and only if its odd part fits in p bits.
    if (boost::multiprecision::msb(C_int) - boost::multiprecision::lsb(C_int) + 1u <= p) {
        REQUIRE(lo == hi);
        REQUIRE(to_rat(lo) == C);
    } else {
        REQUIRE(to_rat(lo) < C);
        REQUIRE(C < to_rat(hi));
        REQUIRE(hi == nextafter(lo, static_cast<T>(std::numeric_limits<double>::infinity())));
    }
}

TEST_CASE("fixed precision")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        const auto p = static_cast<std::uint32_t>(std::numeric_limits<fp_t>::digits);
        const auto max_exponent = static_cast<std::size_t>(std::numeric_limits<fp_t>::max_exponent);

        for (std::uint32_t n = 0; n <= n_max; ++n) {
            for (std::uint32_t k = 0; k <= n; ++k) {
                // Within the tested range, the result overflows if and only if C(n, k) has more than max_exponent bits.
                // This happens only for float, from n = 132.
                if (boost::multiprecision::msb(binomials[n][k]) + 1u > max_exponent) {
                    REQUIRE_THROWS_MATCHES(detail::ed_make_ival_binomial_cf_like(n, k, x), std::overflow_error,
                                           Message(overflow_message(n, k)));
                } else {
                    check_binomial(n, k, x, p);
                }
            }
        }
    });

    // C(1100, 550) has 1095 bits, thus it overflows double.
    REQUIRE_THROWS_MATCHES(detail::ed_make_ival_binomial_cf_like(1100u, 550u, 0.), std::overflow_error,
                           Message(overflow_message(1100, 550)));

    // C(132496419, 5) lies strictly between the largest float and 2**128: lo is the largest float, hi overflows.
    const auto C = cpp_int(132496419) * 132496418 * 132496417 * 132496416 * 132496415 / 120;
    REQUIRE(rat(C) > to_rat(std::numeric_limits<float>::max()));
    REQUIRE(C < (cpp_int(1) << 128));

    REQUIRE_THROWS_MATCHES(detail::ed_make_ival_binomial_cf_like(132496419u, 5u, 0.f), std::overflow_error,
                           Message(overflow_message(132496419, 5)));
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("real")
{
    for (const auto prec : {11, 24, 53, 64, 113, 200}) {
        const mppp::real x{0, prec};

        for (std::uint32_t n = 0; n <= n_max; ++n) {
            for (std::uint32_t k = 0; k <= n; ++k) {
                check_binomial(n, k, x, static_cast<std::uint32_t>(prec));
            }
        }
    }
}

#endif
