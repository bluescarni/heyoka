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
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// Tests for ed_count_sign_changes_ival().
//
// The tests are exhaustive: for every sequence of signs from {'+', '-', '0', '?'} up to a maximum length, the output of
// ed_count_sign_changes_ival() is compared with the minimum/maximum number of sign changes computed by brute force over
// all the replacements of each '?' with '+', '-' or '0'.

using namespace heyoka;
using namespace heyoka_test;

// The floating-point types tested.
const auto fp_types = std::tuple<float, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// The integer representation of the '?' sign. The other signs are represented as +1 ('+'), -1 ('-') and 0 ('0').
constexpr int und = 2;

// The number of sign changes in a sequence of signs without '?', skipping the zeros.
std::uint32_t count_point_sign_changes(const std::vector<int> &signs)
{
    std::uint32_t ret = 0;
    int last_nz = 0;

    for (const auto s : signs) {
        if (s == 0) {
            continue;
        }

        if (last_nz != 0 && s != last_nz) {
            ++ret;
        }

        last_nz = s;
    }

    return ret;
}

// The minimum/maximum number of sign changes in a sequence of signs, computed by brute force over all the replacements
// of each '?' with '+', '-' or '0'.
std::pair<std::uint32_t, std::uint32_t> brute_force_sign_changes(std::vector<int> signs)
{
    // Locate the '?' signs.
    std::vector<std::size_t> und_idx;
    for (std::size_t i = 0; i < signs.size(); ++i) {
        if (signs[i] == und) {
            und_idx.push_back(i);
        }
    }

    // Compute the number of replacements, i.e., 3**(number of '?').
    //
    // NOTE: every value from 0 to n_rep - 1 can be interpreted as a base-3 number with k digits, where k is the number
    // of '?' signs. Each digit represents the choice for the corresponding '?', so that each value represents a
    // different filling.
    std::uint64_t n_rep = 1;
    for (std::size_t i = 0; i < und_idx.size(); ++i) {
        n_rep *= 3u;
    }

    auto min_n_sc = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_n_sc = 0;

    for (std::uint64_t rep = 0; rep < n_rep; ++rep) {
        // Decode rep in base 3 into the signs replacing the '?'.
        auto tmp = rep;
        for (const auto idx : und_idx) {
            signs[idx] = static_cast<int>(tmp % 3u) - 1;
            tmp /= 3u;
        }

        const auto n_sc = count_point_sign_changes(signs);
        min_n_sc = std::min(min_n_sc, n_sc);
        max_n_sc = std::max(max_n_sc, n_sc);
    }

    return {min_n_sc, max_n_sc};
}

// A sign, together with the (mlo, hi) pairs used to represent it as an interval coefficient.
struct sign_data {
    int sign;
    std::vector<std::pair<double, double>> ivals;
};

// Run the exhaustive test for the floating-point type T, for all the sequences of signs of length up to max_len.
// make_fp converts a double into a T.
template <typename T>
void run_exhaustive_test(const std::uint32_t max_len, const auto &make_fp)
{
    const auto inf = std::numeric_limits<double>::infinity();

    // The four signs. The interval representations include infinite bounds and zero bounds of either sign.
    const std::array<sign_data, 4> sign_table = {
        sign_data{.sign = 1, .ivals = {{-1., 2.}, {-.5, .5}, {-1., inf}}},
        sign_data{.sign = -1, .ivals = {{2., -1.}, {.5, -.5}, {inf, -1.}}},
        sign_data{.sign = 0, .ivals = {{0., 0.}, {-0., 0.}, {0., -0.}, {-0., -0.}}},
        sign_data{
            .sign = und,
            .ivals = {{1., 1.}, {-0., 1.}, {0., 1.}, {1., 0.}, {1., -0.}, {inf, inf}, {1., inf}, {inf, 1.}},
        },
    };

    // Counter used to cycle through the interval representations of the signs.
    std::size_t counter = 0;

    std::vector<int> signs;
    std::vector<T> cfs;

    for (std::uint32_t len = 1; len <= max_len; ++len) {
        // Compute the number of sequences of length len, i.e., 4**len.
        std::uint64_t n_seq = 1;
        for (std::uint32_t i = 0; i < len; ++i) {
            n_seq *= 4u;
        }

        for (std::uint64_t seq = 0; seq < n_seq; ++seq) {
            signs.clear();
            cfs.clear();

            // Decode seq in base 4 into the sequence of signs, and build the interval coefficients.
            auto tmp = seq;
            for (std::uint32_t i = 0; i < len; ++i) {
                const auto &sd = sign_table[tmp % 4u];
                tmp /= 4u;

                signs.push_back(sd.sign);

                const auto &[mlo, hi] = sd.ivals[counter++ % sd.ivals.size()];
                cfs.push_back(make_fp(mlo));
                cfs.push_back(make_fp(hi));
            }

            REQUIRE(detail::ed_count_sign_changes_ival<T>(std::span<const T>(cfs)) == brute_force_sign_changes(signs));
        }
    }
}

TEST_CASE("exhaustive")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        run_exhaustive_test<fp_t>(8, [](double v) { return static_cast<fp_t>(v); });
    });
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("real")
{
    for (const auto prec : {11, 53, 200}) {
        run_exhaustive_test<mppp::real>(6, [prec](double v) { return mppp::real{v, prec}; });
    }
}

#endif
