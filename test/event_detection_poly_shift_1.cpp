// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cfenv>
#include <cstdint>
#include <initializer_list>
#include <limits>
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
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// Tests for llvm_ed_ival_poly_shift_1().
//
// llvm_ed_ival_poly_shift_1() computes intervals enclosing the coefficients of p(x+1), where p(x) is a polynomial with
// interval coefficients. The tests check that the computed intervals contain the exact ones, computed with interval
// arithmetic on rationals.

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
const std::vector<std::uint32_t> degrees = {0, 1, 2, 3, 4, 7, 10, 20, 40};

// An interval with rational endpoints.
struct rat_ival {
    rat lo, hi;
};

// Convert the finite value x of type T exactly to a rational.
//
// x is split into a sum of doubles, each of which is converted exactly. Each subtraction is exact by Sterbenz's lemma,
// as d (x rounded to double) is within a factor 2 of x. This requires x and the remainders to be within the normal
// range of double.
template <typename T>
rat to_rat(T x)
{
    rat ret = 0;

    while (x != 0) {
        const auto d = static_cast<double>(x);
        ret += rat(d);
        x -= static_cast<T>(d);
    }

    return ret;
}

// The binomial coefficient C(m, k), with k <= m.
cpp_int exact_binomial(std::uint32_t m, std::uint32_t k)
{
    cpp_int ret = 1;

    // NOTE: after the iteration i, ret is C(m - k + i, i).
    for (std::uint32_t i = 1; i <= k; ++i) {
        ret = ret * (m - k + i) / i;
    }

    return ret;
}

// The product of the rational c and the interval x.
rat_ival ival_scale(const rat &c, const rat_ival &x)
{
    if (c >= 0) {
        return {c * x.lo, c * x.hi};
    } else {
        return {c * x.hi, c * x.lo};
    }
}

// The exact shift by 1 of the interval polynomial with coefficients cfs (in ascending order), computed via the binomial
// formula b_k = sum_{m >= k} C(m, k) * c_m in interval arithmetic.
std::vector<rat_ival> exact_shift_1(const std::vector<rat_ival> &cfs)
{
    std::vector<rat_ival> ret;

    for (std::uint32_t k = 0; k < cfs.size(); ++k) {
        rat_ival acc{0, 0};

        for (auto m = k; m < cfs.size(); ++m) {
            const auto term = ival_scale(rat(exact_binomial(m, k)), cfs[m]);
            acc.lo += term.lo;
            acc.hi += term.hi;
        }

        ret.push_back(acc);
    }

    return ret;
}

// Pointer to a function computing the shift by 1 of an interval polynomial. The first argument is the output array, the
// second argument the input array. Both arrays have size 2 * (n + 1) and use the split (-lo, hi) layout.
template <typename T>
using shift_t = void (*)(T *, const T *);

// Add to s a function computing the shift by 1 of an interval polynomial of degree n via llvm_ed_ival_poly_shift_1(),
// compile s and return a pointer to the function. fp_t is the scalar LLVM type used in the computation, and T is the
// corresponding C++ type.
template <typename T>
shift_t<T> add_shift(llvm_state &s, llvm::Type *fp_t, std::uint32_t n)
{
    auto &builder = s.builder();
    auto &context = s.context();

    auto *const ext_fp_t = detail::make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    auto *const ft = llvm::FunctionType::get(builder.getVoidTy(), {ptr_t, ptr_t}, false);
    auto *const f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "shift", &s.module());
    auto *const out_ptr = f->getArg(0);
    auto *const in_ptr = f->getArg(1);

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    {
        const auto ivg = detail::llvm_ival_guard(s, f, fp_t);

        std::vector<llvm::Value *> cfs;
        for (std::uint32_t i = 0; i < 2u * (n + 1u); ++i) {
            auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, in_ptr, builder.getInt32(i));
            cfs.push_back(detail::ext_load_vector_from_memory(s, fp_t, ptr, 1));
        }

        const auto ret = detail::llvm_ed_ival_poly_shift_1(ivg, cfs);

        for (std::uint32_t i = 0; i < 2u * (n + 1u); ++i) {
            auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(i));
            detail::ext_store_vector_to_memory(s, ptr, ret[i]);
        }

        ivg.finalise();
        builder.CreateRetVoid();
    }

    s.compile();

    return reinterpret_cast<shift_t<T>>(s.jit_lookup("shift"));
}

// Overload of add_shift() in which the scalar LLVM type is deduced from T.
template <typename T>
shift_t<T> add_shift(llvm_state &s, std::uint32_t n)
{
    return add_shift<T>(s, detail::to_external_llvm_type<T>(s.context()), n);
}

// Add a and b.
template <typename T>
T test_add(T a, T b)
{
    return a + b;
}

// Check whether addition in the type T honours the rounding mode set via std::fesetround(). Software-emulated
// quadruple-precision arithmetic may not honour it.
template <typename T>
bool add_honours_rounding()
{
    // NOTE: call through a volatile function pointer, so that the compiler can neither evaluate the addition at compile
    // time nor move it across the changes of rounding mode.
    T (*volatile f)(T, T) = &test_add<T>;

    const T eps = std::numeric_limits<T>::epsilon();

    REQUIRE(std::fesetround(FE_UPWARD) == 0);
    const T res = f(T(1), eps / 4);
    REQUIRE(std::fesetround(FE_TONEAREST) == 0);

    // The exact sum is 1 + eps / 4: rounding up gives 1 + eps, rounding to nearest 1.
    return res > 1;
}

// Generate a random value of type T of magnitude up to about 10.
template <typename T>
T rand_val(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> dist(-10., 10.);

    // NOTE: combine two doubles in order to fill the significands of the types wider than double.
    return static_cast<T>(dist(rng)) + static_cast<T>(dist(rng)) * static_cast<T>(0x1p-53);
}

std::mt19937 rng(42);

TEST_CASE("random")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!add_honours_rounding<fp_t>()) {
                WARN("Skipping the test: addition does not honour the rounding mode on this platform");
                return;
            }
        }

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto shift = add_shift<fp_t>(s, n);

                for (auto trial = 0; trial < 20; ++trial) {
                    // Generate the input polynomial, both in the split (-lo, hi) layout and as rational intervals.
                    //
                    // NOTE: every third coefficient is a zero-width interval.
                    std::vector<fp_t> in(2u * (n + 1u));
                    std::vector<rat_ival> in_rat;
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        const auto lo = rand_val<fp_t>(rng);
                        const auto w = rand_val<fp_t>(rng);
                        const auto hi = (i % 3u == 0u) ? lo : lo + (w < 0 ? -w : w);

                        in[i] = -lo;
                        in[n + 1u + i] = hi;
                        in_rat.push_back({to_rat(lo), to_rat(hi)});
                    }

                    std::vector<fp_t> out(in.size());
                    shift(out.data(), in.data());

                    // The computed intervals must contain the exact ones.
                    const auto ref = exact_shift_1(in_rat);
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        const rat out_lo = -to_rat(out[k]);
                        const rat out_hi = to_rat(out[n + 1u + k]);

                        REQUIRE(out_lo <= ref[k].lo);
                        REQUIRE(ref[k].hi <= out_hi);
                    }
                }
            }
        }
    });
}

TEST_CASE("rounding direction")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!add_honours_rounding<fp_t>()) {
                WARN("Skipping the test: addition does not honour the rounding mode on this platform");
                return;
            }
        }

        const fp_t eps = std::numeric_limits<fp_t>::epsilon();
        const fp_t tiny = eps / 64;

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto shift = add_shift<fp_t>(s, 1);

            // The input polynomial is [-1, 1] + [-tiny, tiny]*x, and the exact shifted polynomial is
            // [-1 - tiny, 1 + tiny] + [-tiny, tiny]*x. Rounding the constant coefficient outward must produce
            // [-1 - eps, 1 + eps], whereas rounding to nearest would produce [-1, 1], which does not enclose the exact
            // interval.
            const std::vector<fp_t> in = {fp_t(1), tiny, fp_t(1), tiny};
            std::vector<fp_t> out(in.size());
            shift(out.data(), in.data());

            REQUIRE(out == std::vector<fp_t>{1 + eps, tiny, 1 + eps, tiny});
        }
    });
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("real")
{
    using fp_t = mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        // Generate a random value with precision prec of magnitude up to about 10.
        const auto rand_real = [prec]() {
            std::uniform_real_distribution<double> dist(-10., 10.);

            const auto a = dist(rng);
            const auto b = dist(rng);

            // NOTE: combine two doubles in order to fill the significands of the precisions higher than double's.
            return fp_t{a, prec} + fp_t{b, prec} * fp_t{0x1p-53, prec};
        };

        // The same checks as in the "random" test case.
        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto shift = add_shift<fp_t>(s, detail::internal_llvm_type_like(s, fp_t{0, prec}), n);

                for (auto trial = 0; trial < 20; ++trial) {
                    std::vector<fp_t> in(2u * (n + 1u));
                    std::vector<rat_ival> in_rat;
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        const auto lo = rand_real();
                        const auto w = rand_real();
                        const auto hi = (i % 3u == 0u) ? lo : lo + (w < 0 ? -w : w);

                        in[i] = -lo;
                        in[n + 1u + i] = hi;
                        in_rat.push_back({to_rat(lo), to_rat(hi)});
                    }

                    std::vector<fp_t> out(in.size(), fp_t{0, prec});
                    shift(out.data(), in.data());

                    const auto ref = exact_shift_1(in_rat);
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        // The function must not alter the precision of the output values.
                        REQUIRE(out[k].get_prec() == prec);
                        REQUIRE(out[n + 1u + k].get_prec() == prec);

                        const rat out_lo = -to_rat(out[k]);
                        const rat out_hi = to_rat(out[n + 1u + k]);

                        REQUIRE(out_lo <= ref[k].lo);
                        REQUIRE(ref[k].hi <= out_hi);
                    }
                }
            }
        }

        // The same check as in the "rounding direction" test case. eps = 2**(1 - prec) is the distance between 1 and
        // the next value with precision prec.
        const fp_t one{1, prec};
        const fp_t eps{"0x1p-" + std::to_string(prec - 1), 16, prec};
        const fp_t tiny{"0x1p-" + std::to_string(prec + 5), 16, prec};

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto shift = add_shift<fp_t>(s, detail::internal_llvm_type_like(s, fp_t{0, prec}), 1);

            const std::vector<fp_t> in = {one, tiny, one, tiny};
            std::vector<fp_t> out(in.size(), fp_t{0, prec});
            shift(out.data(), in.data());

            REQUIRE(out == std::vector<fp_t>{one + eps, tiny, one + eps, tiny});
        }
    }
}

#endif
