// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <utility>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Floor.
llvm::Value *llvm_floor(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.floor",
#if defined(HEYOKA_HAVE_REAL128)
                          "floorq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_floor",
#endif
                          {x});
}

// Trunc.
llvm::Value *llvm_trunc(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.trunc",
#if defined(HEYOKA_HAVE_REAL128)
                          "truncq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_trunc",
#endif
                          {x});
}

// Helper to compute sin and cos simultaneously.
// NOTE: although there exists a SLEEF function for computing sin/cos
// at the same time, we cannot use it directly because it returns a pair
// of SIMD vectors rather than a single one and that does not play
// well with the calling conventions. In theory we could write a wrapper
// for these sincos functions using pointers for output values,
// but compiling such a wrapper requires correctly
// setting up the SIMD compilation flags. Perhaps we can consider this in the
// future to improve performance.
// NOTE: for the vfabi machinery, I think we would need to create internal scalar
// and vector functions that implement the sincos() primitive. Then we would call
// the scalar primitive attaching the vfabi info about the vector variants. For this
// to work it looks like we would need a list of SIMD widths supported on the
// CPU, possibly implemented in target_features.
// NOTE: another possible improvement is an optimisation pass that automatically detects
// sin/cos usages that can be compressed in a single sincos call. If this were to work,
// we could just implement this a sin + cos and let the optimisation pass do
// the heavy lifting.
std::pair<llvm::Value *, llvm::Value *> llvm_sincos(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    [[maybe_unused]] auto *x_t = x->getType();
    [[maybe_unused]] auto *scal_t = x_t->getScalarType();

    // NOTE: real128 has a specialised primitive for this.
#if defined(HEYOKA_HAVE_REAL128)

    auto &context = s.context();

    if (scal_t == to_external_llvm_type<mppp::real128>(context, false)) {
        auto &builder = s.builder();

        // Convert the vector argument to scalars.
        auto x_scalars = vector_to_scalars(builder, x);

        // Execute the sincosq() function on the scalar values and store
        // the results in res_scalars.
        // NOTE: need temp storage because sincosq uses pointers
        // for output values.
        auto *s_all = builder.CreateAlloca(scal_t);
        auto *c_all = builder.CreateAlloca(scal_t);
        std::vector<llvm::Value *> res_sin, res_cos;
        for (const auto &x_scal : x_scalars) {
            llvm_invoke_external(s, "sincosq", builder.getVoidTy(), {x_scal, s_all, c_all},
                                 llvm::AttributeList::get(context, llvm::AttributeList::FunctionIndex,
                                                          {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));

            res_sin.emplace_back(builder.CreateLoad(scal_t, s_all));
            res_cos.emplace_back(builder.CreateLoad(scal_t, c_all));
        }

        // Reconstruct the return value as a vector.
        return {scalars_to_vector(builder, res_sin), scalars_to_vector(builder, res_cos)};
    }

#endif

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(x_t) != 0) {
        return llvm_real_sincos(s, x);
    }

#endif

    return {llvm_sin(s, x), llvm_cos(s, x)};
}

// Two-argument arctan.
llvm::Value *llvm_atan2(llvm_state &s, llvm::Value *y, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan2", {y, x});
}

// Exponential.
llvm::Value *llvm_exp(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.exp",
#if defined(HEYOKA_HAVE_REAL128)
                          "expq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_exp",
#endif
                          {x});
}

// Cosine.
llvm::Value *llvm_cos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.cos",
#if defined(HEYOKA_HAVE_REAL128)
                          "cosq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_cos",
#endif
                          {x});
}

// Sine.
llvm::Value *llvm_sin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sin",
#if defined(HEYOKA_HAVE_REAL128)
                          "sinq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sin",
#endif
                          {x});
}

// Hyperbolic cosine.
llvm::Value *llvm_cosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "cosh", {x});
}

// Error function.
llvm::Value *llvm_erf(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "erf", {x});
}

// Natural logarithm.
llvm::Value *llvm_log(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.log",
#if defined(HEYOKA_HAVE_REAL128)
                          "logq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_log",
#endif
                          {x});
}

// Inverse sine.
llvm::Value *llvm_asin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asin", {x});
}

// Inverse hyperbolic sine.
llvm::Value *llvm_asinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asinh", {x});
}

// Inverse tangent.
llvm::Value *llvm_atan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan", {x});
}

// Inverse hyperbolic tangent.
llvm::Value *llvm_atanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atanh", {x});
}

// Sigmoid.
llvm::Value *llvm_sigmoid(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    // Create the 1 constant.
    auto *one_fp = llvm_constantfp(s, x->getType(), 1.);

    // Compute -x.
    auto *m_x = llvm_fneg(s, x);

    // Compute e^(-x).
    auto *e_m_x = llvm_exp(s, m_x);

    // Return 1 / (1 + e_m_arg).
    return llvm_fdiv(s, one_fp, llvm_fadd(s, one_fp, e_m_x));
}

// Inverse cosine.
llvm::Value *llvm_acos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acos", {x});
}

// Inverse hyperbolic cosine.
llvm::Value *llvm_acosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acosh", {x});
}

// Hyperbolic sine.
llvm::Value *llvm_sinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "sinh", {x});
}

// Square root.
llvm::Value *llvm_sqrt(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sqrt",
#if defined(HEYOKA_HAVE_REAL128)
                          "sqrtq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sqrt",
#endif
                          {x});
}

// Tangent.
llvm::Value *llvm_tan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tan", {x});
}

// Hyperbolic tangent.
llvm::Value *llvm_tanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tanh", {x});
}

// Exponentiation.
llvm::Value *llvm_pow(llvm_state &s, llvm::Value *x, llvm::Value *y)
{
    return llvm_math_intr(s, "llvm.pow",
#if defined(HEYOKA_HAVE_REAL128)
                          "powq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_pow",
#endif
                          {x, y});
}

} // namespace detail

HEYOKA_END_NAMESPACE
