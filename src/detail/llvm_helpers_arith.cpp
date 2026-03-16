// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <stdexcept>

#include <fmt/core.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL)

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

llvm::Value *llvm_fadd(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFAdd(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_add", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fadd values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fsub(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFSub(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_sub", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fsub values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fmul(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFMul(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_mul", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fmul values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fdiv(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFDiv(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        auto *f = real_nary_op(s, fp_t, "mpfr_div", 2u);

        return builder.CreateCall(f, {a, b});
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fdiv values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fneg(llvm_state &s, llvm::Value *a)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFNeg(a);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fneg(s, a);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fneg values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Squaring.
llvm::Value *llvm_square(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    if (x->getType()->getScalarType()->isFloatingPointTy()) {
        return s.builder().CreateFMul(x, x);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x->getType()) != 0) {
        auto *f = real_nary_op(s, x->getType(), "mpfr_sqr", 1u);
        return s.builder().CreateCall(f, {x});
#endif
    } else {
        throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of square()",
                                                llvm_type_name(x->getType())));
    }
}

// Fused multiply-add.
llvm::Value *llvm_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    return llvm_math_intr(s, "llvm.fma",
#if defined(HEYOKA_HAVE_REAL128)
                          "fmaq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_fma",
#endif
                          {x, y, z});
}

// Helper to compute abs(x_v).
llvm::Value *llvm_abs(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.fabs",
#if defined(HEYOKA_HAVE_REAL128)
                          "fabsq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_abs",
#endif
                          {x});
}

} // namespace detail

HEYOKA_END_NAMESPACE
