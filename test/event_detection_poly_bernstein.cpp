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
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
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

// Tests for llvm_ed_ival_poly_bernstein().
//
// llvm_ed_ival_poly_bernstein() computes intervals enclosing the Bernstein coefficients b_k on [0, 1] of q(x) = p(h*x),
// where p(t) is a polynomial with coefficients a_i. The tests compare the output with the exact coefficients, computed
// with rationals: the intervals must contain them, and their bounds must never be NaN or infinite on the inner side.

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
// NOTE: in double precision, the binomial coefficients C(n,k) are all exactly representable only up to n = 56. The
// degree 60 thus exercises the division by inexact binomial coefficients also in double precision.
const std::vector<std::uint32_t> degrees = {0, 1, 2, 3, 4, 7, 10, 20, 40, 60};

// The binomial coefficient C(m, j), with j <= m.
cpp_int exact_binomial(std::uint32_t m, std::uint32_t j)
{
    cpp_int num = 1, den = 1;

    for (std::uint32_t i = 0; i < j; ++i) {
        num *= m - i;
        den *= i + 1u;
    }

    return num / den;
}

// The product of the polynomials with integer coefficients a and b (stored in ascending order).
std::vector<std::int64_t> poly_mul(const std::vector<std::int64_t> &a, const std::vector<std::int64_t> &b)
{
    std::vector<std::int64_t> ret(a.size() + b.size() - 1u, 0);

    for (decltype(a.size()) i = 0; i < a.size(); ++i) {
        for (decltype(b.size()) j = 0; j < b.size(); ++j) {
            ret[i + j] += a[i] * b[j];
        }
    }

    return ret;
}

// The exact Bernstein coefficients b_k = sum_{i<=k} C(k,i)/C(n,i) * c_i of q(x) = p(h*x), where c_i = a_i * h**i are
// the coefficients of q and the a_i (stored in ascending order in cfs) are the coefficients of p.
template <typename T>
std::vector<rat> exact_bernstein(const std::vector<T> &cfs, const T &h)
{
    const auto n = static_cast<std::uint32_t>(cfs.size() - 1u);
    const auto h_rat = to_rat(h);

    // Compute the c_i.
    std::vector<rat> c;
    rat h_pow = 1;
    for (const auto &a : cfs) {
        c.push_back(to_rat(a) * h_pow);
        h_pow *= h_rat;
    }

    // Compute the b_k.
    std::vector<rat> b;
    for (std::uint32_t k = 0; k <= n; ++k) {
        rat b_k = 0;
        for (std::uint32_t i = 0; i <= k; ++i) {
            b_k += rat(exact_binomial(k, i), exact_binomial(n, i)) * c[i];
        }
        b.push_back(b_k);
    }

    return b;
}

// Pointer to a function computing the Bernstein forms of batch_size polynomials of degree n. The first argument is the
// output array, of size 2 * (n + 1) * batch_size, the second argument the array of the (n + 1) * batch_size input
// coefficients, the third argument the array of the batch_size values of h. For the j-th polynomial, the i-th input
// coefficient is at index i * batch_size + j, and the k-th output interval uses the split (-lo, hi) layout, with -lo at
// index k * batch_size + j and hi at index (n + 1 + k) * batch_size + j.
template <typename T>
using bernstein_t = void (*)(T *, const T *, const T *);

// Add to s a function computing the Bernstein forms of batch_size polynomials of degree n via
// llvm_ed_ival_poly_bernstein(), compile s and return a pointer to the function. fp_t is the scalar LLVM type used in
// the computation, and T is the corresponding C++ type.
template <typename T>
bernstein_t<T> add_bernstein(llvm_state &s, llvm::Type *fp_t, std::uint32_t n, std::uint32_t batch_size = 1)
{
    auto &builder = s.builder();
    auto &context = s.context();

    auto *const ext_fp_t = detail::make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    auto *const ft = llvm::FunctionType::get(builder.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);
    auto *const f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "bernstein", &s.module());
    auto *const out_ptr = f->getArg(0);
    auto *const cfs_ptr = f->getArg(1);
    auto *const h_ptr = f->getArg(2);

    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    {
        const auto ivg = detail::llvm_ival_guard(s, f, detail::make_vector_type(fp_t, batch_size));

        std::vector<llvm::Value *> cfs;
        for (std::uint32_t i = 0; i <= n; ++i) {
            auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, cfs_ptr, builder.getInt32(i * batch_size));
            cfs.push_back(detail::ext_load_vector_from_memory(s, fp_t, ptr, batch_size));
        }

        auto *const h = detail::ext_load_vector_from_memory(s, fp_t, h_ptr, batch_size);

        const auto ret = detail::llvm_ed_ival_poly_bernstein(ivg, cfs, h);

        // Store the output intervals in the split layout.
        for (std::uint32_t i = 0; i <= n; ++i) {
            const auto [mlo, hi] = ivg.ival_extract(ret[i]);
            auto *const mlo_ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32(i * batch_size));
            auto *const hi_ptr
                = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.getInt32((n + 1u + i) * batch_size));
            detail::ext_store_vector_to_memory(s, mlo_ptr, mlo);
            detail::ext_store_vector_to_memory(s, hi_ptr, hi);
        }

        ivg.finalise();
        builder.CreateRetVoid();
    }

    s.compile();

    return reinterpret_cast<bernstein_t<T>>(s.jit_lookup("bernstein"));
}

// Overload of add_bernstein() in which the scalar LLVM type is deduced from T.
template <typename T>
bernstein_t<T> add_bernstein(llvm_state &s, std::uint32_t n, std::uint32_t batch_size = 1)
{
    return add_bernstein<T>(s, detail::to_external_llvm_type<T>(s.context()), n, batch_size);
}

// Check the output out of the conversion to the Bernstein form of the polynomial with coefficients cfs rescaled by h.
//
// The output components must never be NaN or -inf, and the output intervals must enclose the exact Bernstein
// coefficients. If finite_required is true, the output components must also be finite.
template <typename T>
void check_bernstein(const std::vector<T> &out, const std::vector<T> &cfs, const T &h, bool finite_required)
{
    using std::isinf;
    using std::isnan;

    const auto op1 = cfs.size();
    const auto b = exact_bernstein(cfs, h);

    for (decltype(cfs.size()) k = 0; k < op1; ++k) {
        const auto &mlo = out[k];
        const auto &hi = out[op1 + k];

        REQUIRE(!isnan(mlo));
        REQUIRE(!isnan(hi));
        REQUIRE(!(isinf(mlo) && mlo < 0));
        REQUIRE(!(isinf(hi) && hi < 0));

        if (finite_required) {
            REQUIRE(!isinf(mlo));
            REQUIRE(!isinf(hi));
        }

        if (!isinf(mlo)) {
            REQUIRE(-to_rat(mlo) <= b[k]);
        }
        if (!isinf(hi)) {
            REQUIRE(b[k] <= to_rat(hi));
        }
    }
}

std::mt19937 rng(42);

TEST_CASE("random")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!add_honours_rounding<fp_t>() || !mul_honours_rounding<fp_t>()) {
                WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
                return;
            }
        }

        std::bernoulli_distribution sign;

        for (const auto n : degrees) {
            // NOTE: with coefficients and h of magnitude up to about 10, overflow is possible only in single precision
            // at the degrees n > 20, where |h|**n can exceed the largest float. In all the other cases, the output must
            // be finite.
            const auto finite_required = !std::same_as<fp_t, float> || n <= 20u;

            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto bernstein = add_bernstein<fp_t>(s, n);

                for (auto trial = 0; trial < 20; ++trial) {
                    // Generate the input polynomial.
                    //
                    // NOTE: every third coefficient is zero.
                    std::vector<fp_t> cfs(n + 1u);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        cfs[i] = (i % 3u == 2u) ? fp_t(0) : rand_uniform<fp_t>(rng, -10., 10.);
                    }

                    // Generate h with magnitude between about 0.1 and 10 and random sign. The first two trials use
                    // h = 0 and h = -0.
                    auto h = rand_uniform<fp_t>(rng, 0.1, 10.);
                    if (sign(rng)) {
                        h = -h;
                    }
                    if (trial == 0) {
                        h = fp_t(0);
                    } else if (trial == 1) {
                        h = -fp_t(0);
                    }

                    std::vector<fp_t> out(2u * (n + 1u));
                    bernstein(out.data(), cfs.data(), &h);

                    check_bernstein(out, cfs, h, finite_required);
                }
            }
        }
    });
}

TEST_CASE("overflow")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if constexpr (detail::is_ieee754_binary128<fp_t>) {
            if (!add_honours_rounding<fp_t>() || !mul_honours_rounding<fp_t>()) {
                WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
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

        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto bernstein = add_bernstein<fp_t>(s, 2);

            std::vector<fp_t> out(6);

            for (const auto h : {abs_h, -abs_h}) {
                // p(t) = 1 + t**2: the exact Bernstein coefficients of p(h*x) are 1, 1 and 1 + h**2. The upper bound of
                // h**2 overflows to +inf, whereas its lower bound stops at max_val, and so does the lower bound of the
                // sum 1 + h**2 computed in the shift.
                const std::vector<fp_t> cfs_pos = {fp_t(1), fp_t(0), fp_t(1)};
                bernstein(out.data(), cfs_pos.data(), &h);
                REQUIRE(out == std::vector<fp_t>{fp_t(-1), fp_t(-1), -max_val, fp_t(1), fp_t(1), inf});

                // p(t) = -1 - t**2: the exact Bernstein coefficients are -1, -1 and -1 - h**2.
                const std::vector<fp_t> cfs_neg = {fp_t(-1), fp_t(0), fp_t(-1)};
                bernstein(out.data(), cfs_neg.data(), &h);
                REQUIRE(out == std::vector<fp_t>{fp_t(1), fp_t(1), inf, fp_t(-1), fp_t(-1), -max_val});
            }
        }
    });
}

// In this test, the input polynomial is constructed from integer Bernstein coefficients b_k, so that all the operations
// of the conversion are exact and the output intervals must be exactly [b_k, b_k].
//
// NOTE: the test is limited to double precision. The degrees and the b_k are small, so that all the values in the
// computation are integers (or integers divided by powers of 2) exactly representable in double precision.
TEST_CASE("exact")
{
    std::uniform_int_distribution<std::int64_t> b_dist(-10, 10);

    for (const auto n : {0u, 1u, 2u, 3u, 4u, 7u, 10u}) {
        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto bernstein = add_bernstein<double>(s, n);

            for (const auto h : {1., -1., 2., -2., .5, -.5}) {
                for (auto trial = 0; trial < 10; ++trial) {
                    // Generate the Bernstein coefficients.
                    std::vector<std::int64_t> b(n + 1u);
                    for (auto &b_k : b) {
                        b_k = b_dist(rng);
                    }

                    // Expand q(x) = sum_k b_k * C(n,k) * x**k * (1-x)**(n-k) into the monomial basis.
                    std::vector<std::int64_t> c(n + 1u, 0);
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        // Compute C(n,k) * x**k * (1-x)**(n-k).
                        std::vector<std::int64_t> basis = {static_cast<std::int64_t>(exact_binomial(n, k))};
                        for (std::uint32_t j = 0; j < k; ++j) {
                            basis = poly_mul(basis, {0, 1});
                        }
                        for (std::uint32_t j = k; j < n; ++j) {
                            basis = poly_mul(basis, {1, -1});
                        }

                        for (std::uint32_t i = 0; i <= n; ++i) {
                            c[i] += b[k] * basis[i];
                        }
                    }

                    // Compute the input coefficients a_i = c_i / h**i, so that p(h*x) = q(x).
                    std::vector<double> cfs(n + 1u);
                    double h_pow = 1;
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        cfs[i] = static_cast<double>(c[i]) / h_pow;
                        h_pow *= h;
                    }

                    // Check the construction against the exact Bernstein coefficients.
                    const auto b_exact = exact_bernstein(cfs, h);
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        REQUIRE(b_exact[k] == b[k]);
                    }

                    std::vector<double> out(2u * (n + 1u));
                    bernstein(out.data(), cfs.data(), &h);

                    // The output intervals must be exactly [b_k, b_k].
                    for (std::uint32_t k = 0; k <= n; ++k) {
                        REQUIRE(out[k] == -static_cast<double>(b[k]));
                        REQUIRE(out[n + 1u + k] == static_cast<double>(b[k]));
                    }
                }
            }
        }
    }
}

// Test the batch mode in double precision.
TEST_CASE("batch")
{
    const std::uint32_t batch_size = 4;

    std::bernoulli_distribution sign;

    for (const auto n : degrees) {
        for (const auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};
            const auto bernstein = add_bernstein<double>(s, n, batch_size);

            for (auto trial = 0; trial < 10; ++trial) {
                // Generate the input polynomials as in the "random" test case.
                std::vector<double> cfs((n + 1u) * batch_size);
                for (std::uint32_t i = 0; i <= n; ++i) {
                    for (std::uint32_t j = 0; j < batch_size; ++j) {
                        cfs[i * batch_size + j] = (i % 3u == 2u) ? 0. : rand_uniform<double>(rng, -10., 10.);
                    }
                }

                // Generate the values of h as in the "random" test case. The first trial uses h = 0 and h = -0 for the
                // first two polynomials.
                std::vector<double> h(batch_size);
                for (auto &h_j : h) {
                    h_j = rand_uniform<double>(rng, 0.1, 10.);
                    if (sign(rng)) {
                        h_j = -h_j;
                    }
                }
                if (trial == 0) {
                    h[0] = 0.;
                    h[1] = -0.;
                }

                std::vector<double> out(2u * (n + 1u) * batch_size);
                bernstein(out.data(), cfs.data(), h.data());

                // Check the output for each polynomial.
                for (std::uint32_t j = 0; j < batch_size; ++j) {
                    std::vector<double> cfs_j(n + 1u), out_j(2u * (n + 1u));
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        cfs_j[i] = cfs[i * batch_size + j];
                    }
                    for (std::uint32_t k = 0; k < 2u * (n + 1u); ++k) {
                        out_j[k] = out[k * batch_size + j];
                    }

                    check_bernstein(out_j, cfs_j, h[j], true);
                }
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("real")
{
    using fp_t = mppp::real;

    std::bernoulli_distribution sign;

    for (const auto prec : {11, 53, 113, 200}) {
        const fp_t zero{0, prec};

        // Generate a random value with precision prec approximately uniformly distributed in [lo, hi].
        const auto rand_real = [prec](double lo, double hi) {
            std::uniform_real_distribution<double> dist(lo, hi);

            const auto a = dist(rng);
            const auto b = dist(rng);

            // NOTE: adding a second double scaled by 2**-53 gives up to about 106 significant bits for the precisions
            // higher than double's.
            return fp_t{a, prec} + fp_t{b, prec} * fp_t{0x1p-53, prec};
        };

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto bernstein = add_bernstein<fp_t>(s, detail::internal_llvm_type_like(s, zero), n);

                // The same checks as in the "random" test case. Overflow is not possible, hence the output must be
                // finite.
                for (auto trial = 0; trial < 20; ++trial) {
                    std::vector<fp_t> cfs(n + 1u, zero);
                    for (std::uint32_t i = 0; i <= n; ++i) {
                        if (i % 3u != 2u) {
                            cfs[i] = rand_real(-10., 10.);
                        }
                    }

                    auto h = rand_real(0.1, 10.);
                    if (sign(rng)) {
                        h = -h;
                    }
                    if (trial == 0) {
                        h = zero;
                    } else if (trial == 1) {
                        h = -zero;
                    }

                    std::vector<fp_t> out(2u * (n + 1u), zero);
                    bernstein(out.data(), cfs.data(), &h);

                    // The function must not alter the precision of the output values.
                    for (const auto &val : out) {
                        REQUIRE(val.get_prec() == prec);
                    }

                    check_bernstein(out, cfs, h, true);
                }
            }
        }
    }
}

#endif
