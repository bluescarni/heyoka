// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
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

// Tests for llvm_ed_ival_poly_rescale().
//
// llvm_ed_ival_poly_rescale() computes intervals enclosing the coefficients c_i * h**i of p(h*x), where p(x) is a
// polynomial with coefficients c_i. The tests check that the computed intervals contain the exact coefficients, computed
// with rationals, and that the output components are never NaN or -inf, and +inf only in case of overflow.

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

// The largest finite value of the type T, as a rational.
//
// NOTE: the largest finite value is (2**p - 1) * 2**(e - p), where p is the number of significand bits and e is the
// maximum exponent, as defined by std::numeric_limits.
template <typename T>
rat max_rat()
{
    constexpr auto p = static_cast<unsigned>(std::numeric_limits<T>::digits);
    constexpr auto e = static_cast<unsigned>(std::numeric_limits<T>::max_exponent);

    return rat(((cpp_int(1) << p) - 1) << (e - p));
}

// Pointer to a function computing the rescaling of a polynomial of degree n. The first argument is the output array,
// which has size 2 * (n + 1) and uses the split (-lo, hi) layout. The second argument is the array of the n + 1 input
// coefficients, the third argument a pointer to the scale factor h.
template <typename T>
using rescale_t = void (*)(T *, const T *, const T *);

// Add to s a function computing the rescaling of a polynomial of degree n via llvm_ed_ival_poly_rescale(), compile s
// and return a pointer to the function. fp_t is the scalar LLVM type used in the computation, and T is the
// corresponding C++ type.
template <typename T>
rescale_t<T> add_rescale(llvm_state &s, llvm::Type *fp_t, std::uint32_t n)
{
    auto &builder = s.builder();
    auto &context = s.context();

    auto *const ext_fp_t = detail::make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    auto *const ft = llvm::FunctionType::get(builder.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
    auto *const f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "rescale", &s.module());
    auto *const out_ptr = f->getArg(0);
    auto *const cfs_ptr = f->getArg(1);
    auto *const h_ptr = f->getArg(2);

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    {
        const auto ivg = detail::llvm_ival_guard(s, f, fp_t);

        std::vector<llvm::Value *> cfs;
        for (std::uint32_t i = 0; i <= n; ++i) {
            auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, cfs_ptr, builder.getInt32(i));
            cfs.push_back(detail::ext_load_vector_from_memory(s, fp_t, ptr, 1));
        }

        auto *const h = detail::ext_load_vector_from_memory(s, fp_t, h_ptr, 1);

        const auto ret = detail::llvm_ed_ival_poly_rescale(ivg, cfs, h);

        for (std::uint32_t i = 0; i < 2u * (n + 1u); ++i) {
            auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(i));
            detail::ext_store_vector_to_memory(s, ptr, ret[i]);
        }

        ivg.finalise();
        builder.CreateRetVoid();
    }

    s.compile();

    return reinterpret_cast<rescale_t<T>>(s.jit_lookup("rescale"));
}

// Overload of add_rescale() in which the scalar LLVM type is deduced from T.
template <typename T>
rescale_t<T> add_rescale(llvm_state &s, std::uint32_t n)
{
    return add_rescale<T>(s, detail::to_external_llvm_type<T>(s.context()), n);
}

// Check the output out of the rescaling by h of the polynomial with coefficients cfs.
//
// The output components must be valid bounds of the exact coefficients c_i * h**i, and they must never be NaN or -inf.
// They may be +inf only if c_i is nonzero and either |h|**i or |c_i * h**i| is larger than max_val / 2, where max_val
// is the largest finite value. An empty max_val means that overflow is not possible.
template <typename T>
void check_rescale(const std::vector<T> &out, const std::vector<T> &cfs, const T &h, const std::optional<rat> &max_val)
{
    using boost::multiprecision::abs;
    using std::isinf;
    using std::isnan;

    const auto op1 = cfs.size();
    const auto h_rat = to_rat(h);

    // The exact value of h**i.
    rat h_pow = 1;

    for (std::size_t i = 0; i < op1; ++i) {
        const auto c_rat = to_rat(cfs[i]);
        const rat exact = c_rat * h_pow;

        const auto &mlo = out[i];
        const auto &hi = out[op1 + i];

        REQUIRE(!isnan(mlo));
        REQUIRE(!isnan(hi));
        REQUIRE(!(isinf(mlo) && mlo < 0));
        REQUIRE(!(isinf(hi) && hi < 0));

        if (c_rat == 0 || !max_val || (abs(h_pow) <= *max_val / 2 && abs(exact) <= *max_val / 2)) {
            REQUIRE(!isinf(mlo));
            REQUIRE(!isinf(hi));
        }

        if (!isinf(mlo)) {
            REQUIRE(-to_rat(mlo) <= exact);
        }
        if (!isinf(hi)) {
            REQUIRE(exact <= to_rat(hi));
        }

        h_pow *= h_rat;
    }
}

// Check that the output out of the rescaling by h of the polynomial with coefficients cfs consists of zero-width
// intervals equal to the exact coefficients c_i * h**i.
template <typename T>
void check_exact(const std::vector<T> &out, const std::vector<T> &cfs, const T &h)
{
    const auto op1 = cfs.size();
    const auto h_rat = to_rat(h);

    // The exact value of h**i.
    rat h_pow = 1;

    for (std::size_t i = 0; i < op1; ++i) {
        const rat exact = to_rat(cfs[i]) * h_pow;

        REQUIRE(-to_rat(out[i]) == exact);
        REQUIRE(to_rat(out[op1 + i]) == exact);

        h_pow *= h_rat;
    }
}

// Check that rescale (a function for polynomials of degree 3) rounds outward, using the polynomial
// 1 + x + x**2 - x**3 and h = +-(1 + eps), where eps is the distance between 1 and the next representable value. For
// instance, (1 + eps)**2 = 1 + 2*eps + eps**2 must be enclosed in [1 + 2*eps, 1 + 3*eps], whereas rounding to nearest
// would give [1 + 2*eps, 1 + 2*eps]. The bounds of (1 + eps)**3 are those of (1 + eps)**2 multiplied by 1 + eps and
// rounded outward, i.e., [1 + 3*eps, 1 + 5*eps].
template <typename T>
void check_rounding_direction(rescale_t<T> rescale, const T &one, const T &eps)
{
    const std::vector<T> cfs = {one, one, one, -one};
    std::vector<T> out(8, one);

    // For h = 1 + eps, the rescaled coefficients are 1, |h|, |h|**2 and -|h|**3.
    const T h = one + eps;
    rescale(out.data(), cfs.data(), &h);
    REQUIRE(out
            == std::vector<T>{-one, -(one + eps), -(one + 2 * eps), one + 5 * eps, one, one + eps, one + 3 * eps,
                              -(one + 3 * eps)});

    // For h = -(1 + eps), the rescaled coefficients are 1, -|h|, |h|**2 and |h|**3.
    const T mh = -h;
    rescale(out.data(), cfs.data(), &mh);
    REQUIRE(out
            == std::vector<T>{-one, one + eps, -(one + 2 * eps), -(one + 3 * eps), one, -(one + eps), one + 3 * eps,
                              one + 5 * eps});
}

// Multiply a and b.
template <typename T>
T test_mul(T a, T b)
{
    return a * b;
}

// Check whether multiplication in the type T honours the rounding mode set via std::fesetround(). This is not the case
// for the software-emulated quadruple precision of LLVM's compiler-rt (used, e.g., on FreeBSD), whose multiplication
// always rounds to nearest.
template <typename T>
bool mul_honours_rounding()
{
    // NOTE: call through a volatile function pointer, so that the compiler can neither evaluate the multiplication at
    // compile time nor move it across the changes of rounding mode.
    T (*volatile f)(T, T) = &test_mul<T>;

    const T eps = std::numeric_limits<T>::epsilon();
    const T x = 1 + eps;

    REQUIRE(std::fesetround(FE_UPWARD) == 0);
    const T res = f(x, x);
    REQUIRE(std::fesetround(FE_TONEAREST) == 0);

    // The exact product is 1 + 2 * eps + eps * eps: rounding up gives 1 + 3 * eps, rounding to nearest 1 + 2 * eps.
    return res > 1 + 2 * eps;
}

// Generate a random value of type T of magnitude up to about 10.
template <typename T>
T rand_val(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> dist(-10., 10.);

    // NOTE: combine two doubles in order to fill the significands of the types wider than double.
    return static_cast<T>(dist(rng)) + static_cast<T>(dist(rng)) * static_cast<T>(0x1p-53);
}

// Generate a random value of type T of magnitude between about 0.1 and 10, with random sign.
template <typename T>
T rand_h(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> dist(0.1, 10.);
    std::bernoulli_distribution sign;

    // NOTE: combine two doubles in order to fill the significands of the types wider than double.
    const auto ret = static_cast<T>(dist(rng)) + static_cast<T>(dist(rng)) * static_cast<T>(0x1p-53);

    return sign(rng) ? -ret : ret;
}

std::mt19937 rng(42);

TEST_CASE("random")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!mul_honours_rounding<fp_t>()) {
                WARN("Skipping the test: multiplication does not honour the rounding mode on this platform");
                return;
            }
        }

        const std::optional<rat> max_val = max_rat<fp_t>();

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto rescale = add_rescale<fp_t>(s, n);

                for (auto trial = 0; trial < 20; ++trial) {
                    // Generate the input polynomial.
                    //
                    // NOTE: every third coefficient is zero.
                    std::vector<fp_t> cfs(n + 1u);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        cfs[i] = (i % 3u == 2u) ? fp_t(0) : rand_val<fp_t>(rng);
                    }

                    // Generate h. The first two trials use h = 0 and h = -0.
                    const auto h = (trial == 0) ? fp_t(0) : ((trial == 1) ? -fp_t(0) : rand_h<fp_t>(rng));

                    std::vector<fp_t> out(2u * (n + 1u));
                    rescale(out.data(), cfs.data(), &h);

                    check_rescale(out, cfs, h, max_val);
                }
            }
        }
    });
}

TEST_CASE("exact")
{
    // NOTE: with h = +-1, +-2 and +-0.5, all the products are exact. Hence, this test does not depend on the rounding
    // mode and it is run also when multiplication does not honour it.
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto rescale = add_rescale<fp_t>(s, n);

                for (const auto h : {fp_t(1), fp_t(-1), fp_t(2), fp_t(-2), fp_t(.5), fp_t(-.5)}) {
                    // Generate the input polynomial.
                    //
                    // NOTE: every third coefficient is zero.
                    std::vector<fp_t> cfs(n + 1u);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        cfs[i] = (i % 3u == 2u) ? fp_t(0) : rand_val<fp_t>(rng);
                    }

                    std::vector<fp_t> out(2u * (n + 1u));
                    rescale(out.data(), cfs.data(), &h);

                    check_exact(out, cfs, h);
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
            if (!mul_honours_rounding<fp_t>()) {
                WARN("Skipping the test: multiplication does not honour the rounding mode on this platform");
                return;
            }
        }

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto rescale = add_rescale<fp_t>(s, 3);

            check_rounding_direction(rescale, fp_t(1), std::numeric_limits<fp_t>::epsilon());
        }
    });
}

TEST_CASE("overflow")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!mul_honours_rounding<fp_t>()) {
                WARN("Skipping the test: multiplication does not honour the rounding mode on this platform");
                return;
            }
        }

        const auto inf = static_cast<fp_t>(std::numeric_limits<double>::infinity());
        const auto max_val = std::numeric_limits<fp_t>::max();

        // |h| = 2**e is finite, while |h|**2 = 2**(max_exponent + 2) overflows.
        const auto e = std::numeric_limits<fp_t>::max_exponent / 2 + 1;
        fp_t abs_h(1);
        for (auto i = 0; i < e; ++i) {
            abs_h *= 2;
        }

        // The input polynomial is 1 + x + x**3 - x**4, with zero coefficients at degrees 2 and 5.
        const std::vector<fp_t> cfs = {fp_t(1), fp_t(1), fp_t(0), fp_t(1), fp_t(-1), fp_t(0)};

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto rescale = add_rescale<fp_t>(s, 5);

            std::vector<fp_t> out(12);

            // For i >= 2, the upper bound of |h|**i overflows to +inf, whereas its lower bound stops at max_val.
            // Hence:
            //
            // - the coefficients of degree 0 and 1 are exact,
            // - the zero coefficients of degree 2 and 5 are exactly zero,
            // - the coefficients of degree 3 and 4 have one bound computed from the upper bound of |h|**i, which is
            //   infinite, and the other bound computed from the lower bound of |h|**i, which is max_val in absolute
            //   value.
            const auto h = abs_h;
            rescale(out.data(), cfs.data(), &h);
            REQUIRE(out
                    == std::vector<fp_t>{fp_t(-1), -abs_h, fp_t(0), -max_val, inf, fp_t(0), fp_t(1), abs_h, fp_t(0), inf,
                                         -max_val, fp_t(0)});

            // For negative h, the odd-degree coefficients are negated.
            const auto mh = -abs_h;
            rescale(out.data(), cfs.data(), &mh);
            REQUIRE(out
                    == std::vector<fp_t>{fp_t(-1), abs_h, fp_t(0), inf, inf, fp_t(0), fp_t(1), -abs_h, fp_t(0), -max_val,
                                         -max_val, fp_t(0)});
        }
    });
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("real")
{
    using fp_t = mppp::real;

    for (const auto prec : {11, 53, 113, 200}) {
        const fp_t zero{0, prec};
        const fp_t one{1, prec};

        // Generate a random value with precision prec of magnitude up to about 10.
        const auto rand_real = [prec]() {
            std::uniform_real_distribution<double> dist(-10., 10.);

            const auto a = dist(rng);
            const auto b = dist(rng);

            // NOTE: combine two doubles in order to fill the significands of the precisions higher than double's.
            return fp_t{a, prec} + fp_t{b, prec} * fp_t{0x1p-53, prec};
        };

        // Generate a random value with precision prec of magnitude between about 0.1 and 10, with random sign.
        const auto rand_real_h = [prec]() {
            std::uniform_real_distribution<double> dist(0.1, 10.);
            std::bernoulli_distribution sign;

            const auto a = dist(rng);
            const auto b = dist(rng);

            // NOTE: combine two doubles in order to fill the significands of the precisions higher than double's.
            const auto ret = fp_t{a, prec} + fp_t{b, prec} * fp_t{0x1p-53, prec};

            return sign(rng) ? -ret : ret;
        };

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto rescale = add_rescale<fp_t>(s, detail::internal_llvm_type_like(s, zero), n);

                // The same checks as in the "random" test case. Overflow is not possible, hence the output must be
                // finite.
                for (auto trial = 0; trial < 20; ++trial) {
                    std::vector<fp_t> cfs(n + 1u, zero);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        if (i % 3u != 2u) {
                            cfs[i] = rand_real();
                        }
                    }

                    const auto h = (trial == 0) ? zero : ((trial == 1) ? -zero : rand_real_h());

                    std::vector<fp_t> out(2u * (n + 1u), zero);
                    rescale(out.data(), cfs.data(), &h);

                    // The function must not alter the precision of the output values.
                    for (const auto &val : out) {
                        REQUIRE(val.get_prec() == prec);
                    }

                    check_rescale(out, cfs, h, std::nullopt);
                }

                // The same checks as in the "exact" test case.
                for (const auto h_dbl : {1., -1., 2., -2., .5, -.5}) {
                    const fp_t h{h_dbl, prec};

                    std::vector<fp_t> cfs(n + 1u, zero);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        if (i % 3u != 2u) {
                            cfs[i] = rand_real();
                        }
                    }

                    std::vector<fp_t> out(2u * (n + 1u), zero);
                    rescale(out.data(), cfs.data(), &h);

                    check_exact(out, cfs, h);
                }
            }
        }

        // The same check as in the "rounding direction" test case. eps = 2**(1 - prec) is the distance between 1 and
        // the next value with precision prec.
        const fp_t eps{"0x1p-" + std::to_string(prec - 1), 16, prec};

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto rescale = add_rescale<fp_t>(s, detail::internal_llvm_type_like(s, zero), 3);

            check_rounding_direction(rescale, one, eps);
        }
    }
}

#endif
