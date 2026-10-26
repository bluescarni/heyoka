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
#include <initializer_list>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// Tests for llvm_ed_ival_de_casteljau_half().
//
// llvm_ed_ival_de_casteljau_half() expresses a Bernstein polynomial on [a, b] in the Bernstein bases of [a, m] and
// [m, b], with m = (a+b)/2. The computation does not depend on [a, b], hence the tests take [a, b] = [0, 1], so that
// the two halves are [0, 1/2] and [1/2, 1].
//
// The tests evaluate the input polynomial and the two halves at a set of points, in rational interval arithmetic and
// using the definition of the Bernstein basis.
//
// In memory, an interval polynomial is stored with the negated lower bound and the upper bound of each coefficient next
// to each other:
//
// [-b0_lo, b0_hi, -b1_lo, b1_hi, -b2_lo, b2_hi, ...]

using namespace heyoka;
using namespace heyoka_test;

using rat = boost::multiprecision::cpp_rational;
using boost::multiprecision::cpp_int;

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

// The polynomial degrees tested.
//
// NOTE: in single and double precision, the degrees up to 23 use the unrolled implementation of
// llvm_ed_ival_de_casteljau_half(), the higher degrees the loop implementation.
const std::vector<std::uint32_t> degrees = {0, 1, 2, 3, 4, 7, 10, 20, 23, 24, 40};

// The value at x of the polynomial with interval Bernstein coefficients b on the interval [lb, ub], computed in
// rational interval arithmetic from the definition of the Bernstein basis:
//
// sum_k b_k * C(n,k) * (x-lb)**k * (ub-x)**(n-k) / (ub-lb)**n.
rat_ival bernstein_eval(const std::vector<rat_ival> &b, const rat &lb, const rat &ub, const rat &x)
{
    const auto n = static_cast<std::uint32_t>(b.size() - 1u);

    // Compute the powers (x-lb)**k and (ub-x)**k for k = 0, ..., n, and (ub-lb)**n.
    std::vector<rat> pow_l = {rat(1)}, pow_r = {rat(1)};
    rat len_pow_n = 1;
    for (std::uint32_t k = 1; k <= n; ++k) {
        pow_l.push_back(pow_l.back() * (x - lb));
        pow_r.push_back(pow_r.back() * (ub - x));
        len_pow_n *= ub - lb;
    }

    rat_ival ret{0, 0};

    // NOTE: binom is C(n,k) at the iteration k.
    cpp_int binom = 1;
    for (std::uint32_t k = 0; k <= n; ++k) {
        ret = ival_add(ret, ival_scale(rat(binom) * pow_l[k] * pow_r[n - k] / len_pow_n, b[k]));
        binom = binom * (n - k) / (k + 1u);
    }

    return ret;
}

// Convert the interval polynomial cfs, stored in memory as described at the top of the file, to rational intervals. The
// bounds must be finite.
template <typename T>
std::vector<rat_ival> to_rat_ivals(const std::vector<T> &cfs)
{
    using std::isfinite;

    const auto op1 = cfs.size() / 2u;

    std::vector<rat_ival> ret;
    for (std::size_t k = 0; k < op1; ++k) {
        REQUIRE(isfinite(cfs[2u * k]));
        REQUIRE(isfinite(cfs[2u * k + 1u]));

        ret.push_back({-to_rat(cfs[2u * k]), to_rat(cfs[2u * k + 1u])});
    }

    return ret;
}

// Pointer to a function computing the split of an interval polynomial. The first and second arguments are the output
// arrays for the halves [0, 1/2] and [1/2, 1], the third argument the input array. All arrays have size 2 * (n + 1).
template <typename T>
using split_t = void (*)(T *, T *, const T *);

// Add to s a function computing the split of an interval polynomial of degree n via llvm_ed_ival_de_casteljau_half(),
// compile s and return a pointer to the function. fp_t is the scalar LLVM type used in the computation, and T is the
// corresponding C++ type.
template <typename T>
split_t<T> add_split(llvm_state &s, llvm::Type *fp_t, std::uint32_t n)
{
    auto &builder = s.builder();
    auto &context = s.context();

    auto *const ext_fp_t = detail::make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    auto *const ft = llvm::FunctionType::get(builder.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
    auto *const f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "split", &s.module());
    auto *const out_left_ptr = f->getArg(0);
    auto *const out_right_ptr = f->getArg(1);
    auto *const in_ptr = f->getArg(2);

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    {
        const auto ivg = detail::llvm_ival_guard(s, f, fp_t);

        // Load the input intervals.
        std::vector<llvm::Value *> cfs;
        for (std::uint32_t i = 0; i <= n; ++i) {
            auto *const mlo_ptr = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.getInt32(2u * i));
            auto *const hi_ptr = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.getInt32(2u * i + 1u));
            auto *const mlo = detail::ext_load_vector_from_memory(s, fp_t, mlo_ptr, 1);
            auto *const hi = detail::ext_load_vector_from_memory(s, fp_t, hi_ptr, 1);
            cfs.push_back(ivg.ival_construct(mlo, hi));
        }

        const auto [left, right] = detail::llvm_ed_ival_de_casteljau_half(ivg, cfs);

        // Store the output intervals ret into out_ptr.
        const auto store = [&](llvm::Value *out_ptr, const std::vector<llvm::Value *> &ret) {
            for (std::uint32_t i = 0; i <= n; ++i) {
                const auto [mlo, hi] = ivg.ival_extract(ret[i]);
                auto *const mlo_ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(2u * i));
                auto *const hi_ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(2u * i + 1u));
                detail::ext_store_vector_to_memory(s, mlo_ptr, mlo);
                detail::ext_store_vector_to_memory(s, hi_ptr, hi);
            }
        };
        store(out_left_ptr, left);
        store(out_right_ptr, right);

        ivg.finalise();
        builder.CreateRetVoid();
    }

    s.compile();

    return reinterpret_cast<split_t<T>>(s.jit_lookup("split"));
}

// Overload of add_split() in which the scalar LLVM type is deduced from T.
template <typename T>
split_t<T> add_split(llvm_state &s, std::uint32_t n)
{
    return add_split<T>(s, detail::to_external_llvm_type<T>(s.context()), n);
}

std::mt19937 rng(42);

// Generate count distinct random test points in [-1, 2]. The points are drawn uniformly from the grid with spacing 1/64
// (x = j / 64 - 1, j = 0, ..., 192), which keeps the denominators in the rational computations small.
std::vector<rat> rand_test_points(std::size_t count)
{
    std::vector<int> js(193);
    std::iota(js.begin(), js.end(), 0);
    std::ranges::shuffle(js, rng);

    REQUIRE(count <= js.size());

    std::vector<rat> ret;
    for (std::size_t i = 0; i < count; ++i) {
        ret.push_back(rat(js[i], 64) - 1);
    }

    return ret;
}

// Check the output of llvm_ed_ival_de_casteljau_half() for the input polynomial in. out_left and out_right are the
// output polynomials on [0, 1/2] and [1/2, 1]. At each test point x, their evaluations must contain the evaluation of
// in.
//
// The 5 test points are generated with rand_test_points().
template <typename T>
void check_split(const std::vector<T> &in, const std::vector<T> &out_left, const std::vector<T> &out_right)
{
    const auto in_rat = to_rat_ivals(in);
    const auto left_rat = to_rat_ivals(out_left);
    const auto right_rat = to_rat_ivals(out_right);

    for (const auto &x : rand_test_points(5)) {
        const auto val = bernstein_eval(in_rat, 0, 1, x);
        const auto val_left = bernstein_eval(left_rat, 0, rat(1, 2), x);
        const auto val_right = bernstein_eval(right_rat, rat(1, 2), 1, x);

        REQUIRE(val_left.lo <= val.lo);
        REQUIRE(val.hi <= val_left.hi);
        REQUIRE(val_right.lo <= val.lo);
        REQUIRE(val.hi <= val_right.hi);
    }
}

// Random interval coefficients.
TEST_CASE("random")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto split = add_split<fp_t>(s, n);

                for (auto trial = 0; trial < 10; ++trial) {
                    // Generate the input polynomial.
                    //
                    // NOTE: every third coefficient is a zero-width interval.
                    std::vector<fp_t> in(2u * (n + 1u));
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        const auto lo = rand_uniform<fp_t>(rng, -10., 10.);
                        const auto w = rand_uniform<fp_t>(rng, -10., 10.);
                        const auto hi = (i % 3u == 0u) ? lo : lo + (w < 0 ? -w : w);

                        in[2u * i] = -lo;
                        in[2u * i + 1u] = hi;
                    }

                    std::vector<fp_t> out_left(in.size()), out_right(in.size());
                    split(out_left.data(), out_right.data(), in.data());

                    check_split(in, out_left, out_right);
                }
            }
        }
    });
}

// Small integer coefficients, for which all the computations are exact: the two halves must be exactly the input
// polynomial.
//
// NOTE: the input coefficients are integers in [-10, 10], and the de Casteljau algorithm repeatedly averages pairs of
// neighbouring numbers, at most n times (n being the polynomial degree). After j averagings, the numbers are integers
// divided by 2**j, and they stay within [-10, 10]. In binary, such a number has at most 4 digits before the point
// (10 = 0b1010) and j after. Hence, no rounding occurs if the floating-point type has at least n + 4 binary digits.
TEST_CASE("exact")
{
    std::uniform_int_distribution<int> b_dist(-10, 10);

    tuple_for_each(fp_types, [&b_dist](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        for (const auto n : degrees) {
            if (static_cast<int>(n) + 4 > std::numeric_limits<fp_t>::digits) {
                continue;
            }

            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto split = add_split<fp_t>(s, n);

                for (auto trial = 0; trial < 10; ++trial) {
                    // Generate the input polynomial with zero-width intervals [b_i, b_i].
                    std::vector<fp_t> in(2u * (n + 1u));
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        const auto b_i = b_dist(rng);

                        in[2u * i] = static_cast<fp_t>(-b_i);
                        in[2u * i + 1u] = static_cast<fp_t>(b_i);
                    }

                    std::vector<fp_t> out_left(in.size()), out_right(in.size());
                    split(out_left.data(), out_right.data(), in.data());

                    // The output intervals must be zero-width.
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        REQUIRE(out_left[2u * k] == -out_left[2u * k + 1u]);
                        REQUIRE(out_right[2u * k] == -out_right[2u * k + 1u]);
                    }

                    // By checking for evaluation equality at n + 2 distinct test points, we are verifying that out_left
                    // and out_right are exactly the same polynomial as 'in' (polynomials of degree at most n are
                    // uniquely determined by their values at n + 1 distinct points).
                    const auto in_rat = to_rat_ivals(in);
                    const auto left_rat = to_rat_ivals(out_left);
                    const auto right_rat = to_rat_ivals(out_right);

                    for (const auto &x : rand_test_points(n + 2u)) {
                        const auto val = bernstein_eval(in_rat, 0, 1, x);
                        const auto val_left = bernstein_eval(left_rat, 0, rat(1, 2), x);
                        const auto val_right = bernstein_eval(right_rat, rat(1, 2), 1, x);

                        REQUIRE(val_left.lo == val.lo);
                        REQUIRE(val_left.hi == val.hi);
                        REQUIRE(val_right.lo == val.lo);
                        REQUIRE(val_right.hi == val.hi);
                    }
                }
            }
        }
    });
}

// An input for which rounding to nearest would produce intervals not enclosing the exact result: the output must be
// rounded outward.
TEST_CASE("rounding direction")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        const fp_t eps = std::numeric_limits<fp_t>::epsilon();
        const fp_t tiny = eps / 64;
        const fp_t half_up = (1 + eps) / 2;

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto split = add_split<fp_t>(s, 1);

            // The input polynomial has the coefficients [-1, 1] and [-tiny, tiny], and the exact average of the two is
            // [-(1 + tiny)/2, (1 + tiny)/2]. Rounding outward must produce [-(1 + eps)/2, (1 + eps)/2], whereas
            // rounding to nearest would produce [-1/2, 1/2], which does not enclose the exact interval.
            const std::vector<fp_t> in = {fp_t(1), fp_t(1), tiny, tiny};
            std::vector<fp_t> out_left(in.size()), out_right(in.size());
            split(out_left.data(), out_right.data(), in.data());

            REQUIRE(out_left == std::vector<fp_t>{fp_t(1), fp_t(1), half_up, half_up});
            REQUIRE(out_right == std::vector<fp_t>{half_up, half_up, tiny, tiny});
        }
    });
}

// Inputs with overflowing sums and infinite bounds: the output must be the expected valid intervals.
TEST_CASE("overflow")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        const auto inf = static_cast<fp_t>(std::numeric_limits<double>::infinity());
        const auto max_val = std::numeric_limits<fp_t>::max();
        const fp_t half_max = max_val / 2;

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto split = add_split<fp_t>(s, 1);

            std::vector<fp_t> out_left(4), out_right(4);

            // Coefficients [max_val, max_val] and [max_val, max_val]: the upper bound of their sum overflows to +inf,
            // whereas its lower bound stops at max_val. The average is thus [max_val/2, +inf].
            const std::vector<fp_t> in_pos = {-max_val, max_val, -max_val, max_val};
            split(out_left.data(), out_right.data(), in_pos.data());
            REQUIRE(out_left == std::vector<fp_t>{-max_val, max_val, -half_max, inf});
            REQUIRE(out_right == std::vector<fp_t>{-half_max, inf, -max_val, max_val});

            // Coefficients [-max_val, -max_val] and [-max_val, -max_val]: the average is [-inf, -max_val/2].
            const std::vector<fp_t> in_neg = {max_val, -max_val, max_val, -max_val};
            split(out_left.data(), out_right.data(), in_neg.data());
            REQUIRE(out_left == std::vector<fp_t>{max_val, -max_val, inf, -half_max});
            REQUIRE(out_right == std::vector<fp_t>{inf, -half_max, max_val, -max_val});

            // Coefficients [-inf, -1] and [1, +inf]: the average is [-inf, +inf].
            const std::vector<fp_t> in_inf = {inf, fp_t(-1), fp_t(-1), inf};
            split(out_left.data(), out_right.data(), in_inf.data());
            REQUIRE(out_left == std::vector<fp_t>{inf, fp_t(-1), inf, inf});
            REQUIRE(out_right == std::vector<fp_t>{inf, inf, fp_t(-1), inf});
        }
    });
}

#if defined(HEYOKA_HAVE_REAL)

// The "random" and "rounding direction" test cases for mppp::real at several precisions. The output must also keep the
// precision of the input.
TEST_CASE("real")
{
    using fp_t = mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto split = add_split<fp_t>(s, detail::internal_llvm_type_like(s, fp_t{0, prec}), n);

                for (auto trial = 0; trial < 10; ++trial) {
                    std::vector<fp_t> in(2u * (n + 1u));
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        const auto lo = rand_uniform_real(rng, -10., 10., prec);
                        const auto w = rand_uniform_real(rng, -10., 10., prec);
                        const auto hi = (i % 3u == 0u) ? lo : lo + (w < 0 ? -w : w);

                        in[2u * i] = -lo;
                        in[2u * i + 1u] = hi;
                    }

                    std::vector<fp_t> out_left(in.size(), fp_t{0, prec}), out_right(in.size(), fp_t{0, prec});
                    split(out_left.data(), out_right.data(), in.data());

                    for (std::size_t k = 0; k < in.size(); ++k) {
                        REQUIRE(out_left[k].get_prec() == prec);
                        REQUIRE(out_right[k].get_prec() == prec);
                    }

                    check_split(in, out_left, out_right);
                }
            }
        }

        // eps = 2**(1 - prec) is the distance between 1 and the next value with precision prec.
        const fp_t one{1, prec};
        const fp_t eps{"0x1p-" + std::to_string(prec - 1), 16, prec};
        const fp_t tiny{"0x1p-" + std::to_string(prec + 5), 16, prec};
        const fp_t half_up = (one + eps) * fp_t{0.5, prec};

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto split = add_split<fp_t>(s, detail::internal_llvm_type_like(s, fp_t{0, prec}), 1);

            const std::vector<fp_t> in = {one, one, tiny, tiny};
            std::vector<fp_t> out_left(in.size(), fp_t{0, prec}), out_right(in.size(), fp_t{0, prec});
            split(out_left.data(), out_right.data(), in.data());

            REQUIRE(out_left == std::vector<fp_t>{one, one, half_up, half_up});
            REQUIRE(out_right == std::vector<fp_t>{half_up, half_up, tiny, tiny});
        }
    }
}

#endif
