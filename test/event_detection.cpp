// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <tuple>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <boost/math/special_functions/binomial.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Minimal interval class supporting a couple
// of elementary operations.
template <typename T>
struct ival {
    T lower;
    T upper;

    ival() : ival(T(0)) {}
    explicit ival(T val) : ival(val, val) {}
    explicit ival(T l, T u) : lower(l), upper(u)
    {
#if !defined(NDEBUG)
        assert(upper >= lower);
#endif
    }
};

// NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
template <typename T>
ival<T> operator+(ival<T> a, ival<T> b)
{
    return ival<T>(a.lower + b.lower, a.upper + b.upper);
}

template <typename T>
ival<T> operator*(ival<T> a, ival<T> b)
{
    const auto tmp1 = a.lower * b.lower;
    const auto tmp2 = a.lower * b.upper;
    const auto tmp3 = a.upper * b.lower;
    const auto tmp4 = a.upper * b.upper;

    const auto l = std::min(std::min(tmp1, tmp2), std::min(tmp3, tmp4));
    const auto u = std::max(std::max(tmp1, tmp2), std::max(tmp3, tmp4));

    return ival<T>(l, u);
}

template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

TEST_CASE("fast exclusion check")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        std::uniform_real_distribution<double> rdist(-10., 10.);
        std::vector<fp_t> poly, h;
        std::vector<std::uint32_t> back_flag, res;

        for (auto batch_size : {1u, 2u, 4u}) {
            for (auto order : {1u, 2u, 13u, 20u}) {
                for (auto use_cs : {false, true}) {
                    llvm_state s{kw::opt_level = opt_level};

                    // Add the function and fetch it.
                    detail::llvm_add_fex_check<fp_t>(s, order, batch_size, use_cs);
                    s.optimise();
                    s.compile();
                    auto fex_check = reinterpret_cast<void (*)(const fp_t *, const fp_t *, const std::uint32_t *,
                                                               std::uint32_t *)>(s.jit_lookup("fex_check"));

                    // Prepare the buffers.
                    poly.resize((order + 1u) * batch_size);
                    h.resize(batch_size);
                    back_flag.resize(batch_size);
                    res.resize(batch_size);

                    // Iterate over a number of trials.
                    for (auto _ = 0; _ < 100; ++_) {
                        // Generate random polys.
                        for (auto &cf : poly) {
                            cf = rdist(rng);
                        }

                        // Generate random hs.
                        for (auto i = 0u; i < batch_size; ++i) {
                            h[i] = rdist(rng);
                            back_flag[i] = h[i] < 0;
                        }

                        // Invoke the fast exclusion check.
                        fex_check(poly.data(), h.data(), back_flag.data(), res.data());

                        if (use_cs) {
                            // Run the manual check with an ad-hoc implementation of
                            // the CS algorithm.
                            for (auto i = 0u; i < batch_size; ++i) {
                                auto cur_bj = poly[i]
                                              * (boost::math::binomial_coefficient<fp_t>(0, 0)
                                                 / boost::math::binomial_coefficient<fp_t>(order, 0));

                                std::pair<fp_t, fp_t> bj_minmax{cur_bj, cur_bj};

                                for (std::uint32_t j = 1; j <= order; ++j) {
                                    cur_bj = 0;
                                    fp_t cur_h_pow = 1;

                                    for (std::uint32_t k = 0; k <= j; ++k) {
                                        cur_bj += poly[k * batch_size + i] * cur_h_pow
                                                  * (boost::math::binomial_coefficient<fp_t>(j, k)
                                                     / boost::math::binomial_coefficient<fp_t>(order, k));

                                        cur_h_pow *= h[i];
                                    }

                                    bj_minmax.first = std::min(bj_minmax.first, cur_bj);
                                    bj_minmax.second = std::max(bj_minmax.second, cur_bj);
                                }

                                REQUIRE(bj_minmax.second >= bj_minmax.first);

                                const auto s_lower = sgn(bj_minmax.first), s_upper = sgn(bj_minmax.second);
                                REQUIRE(res[i] == (s_lower == s_upper && s_lower != 0));
                            }
                        } else {
                            // Run the manual check with the ival class.
                            for (auto i = 0u; i < batch_size; ++i) {
                                const auto h_int = (h[i] >= 0) ? ival<fp_t>(0, h[i]) : ival<fp_t>(h[i], 0);

                                ival<fp_t> acc(poly[order * batch_size + i]);

                                for (std::uint32_t j = 1; j <= order; ++j) {
                                    acc = ival<fp_t>(poly[(order - j) * batch_size + i]) + acc * h_int;
                                }

                                REQUIRE(acc.upper >= acc.lower);

                                const auto s_lower = sgn(acc.lower), s_upper = sgn(acc.upper);
                                REQUIRE(res[i] == (s_lower == s_upper && s_lower != 0));
                            }
                        }
                    }
                }
            }
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tuple_for_each(fp_types, [&tester, opt_level](auto x) { tester(x, opt_level); });
    }
}
