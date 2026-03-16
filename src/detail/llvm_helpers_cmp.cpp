// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <limits>
#include <stdexcept>

#include <fmt/core.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL)

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

llvm::Value *llvm_fcmp_ult(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpULT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ult(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ult values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_uge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpUGE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_uge(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_uge values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ule(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpULE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ule(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ule values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_oge(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOGE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_oge(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_oge values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ole(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOLE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ole(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ole values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_olt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOLT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_olt(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_olt values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ogt(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOGT(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ogt(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ogt values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_oeq(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpOEQ(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_oeq(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_oeq values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_one(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpONE(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_one(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_one values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

llvm::Value *llvm_fcmp_ord(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = a->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpORD(a, b);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fcmp_ord(s, a, b);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Unable to fcmp_ord values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Check if the input floating-point value(s) x is anything other
// than zero (including NaN).
llvm::Value *llvm_fnz(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = x->getType();

    if (fp_t->getScalarType()->isFloatingPointTy()) {
        return builder.CreateFCmpUNE(x, llvm::ConstantFP::get(x->getType(), 0.));
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(fp_t) != 0) {
        return llvm_real_fnz(s, x);
#endif
    } else {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Unable to invoke llvm_fnz() on values of type '{}'", llvm_type_name(fp_t)));
        // LCOV_EXCL_STOP
    }
}

// Minimum value, floating-point arguments. Implemented as std::min():
// return (b < a) ? b : a;
llvm::Value *llvm_min(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    return builder.CreateSelect(llvm_fcmp_olt(s, b, a), b, a);
}

// Maximum value, floating-point arguments. Implemented as std::max():
// return (a < b) ? b : a;
llvm::Value *llvm_max(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    return builder.CreateSelect(llvm_fcmp_olt(s, a, b), b, a);
}

// Same as llvm_min(), but returns NaN if any operand is NaN:
// return (b == b) ? ((b < a) ? b : a) : b;
llvm::Value *llvm_min_nan(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto *b_not_nan = llvm_fcmp_oeq(s, b, b);
    auto *b_lt_a = llvm_fcmp_olt(s, b, a);

    return builder.CreateSelect(b_not_nan, builder.CreateSelect(b_lt_a, b, a), b);
}

// Same as llvm_max(), but returns NaN if any operand is NaN:
// return (b == b) ? ((a < b) ? b : a) : b;
llvm::Value *llvm_max_nan(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto *b_not_nan = llvm_fcmp_oeq(s, b, b);
    auto *a_lt_b = llvm_fcmp_olt(s, a, b);

    return builder.CreateSelect(b_not_nan, builder.CreateSelect(a_lt_b, b, a), b);
}

// Branchless sign function.
// NOTE: requires FP value.
// NOTE: this will return 0 if val is NaN.
llvm::Value *llvm_sgn(llvm_state &s, llvm::Value *val)
{
    assert(val != nullptr);

    auto &builder = s.builder();

    auto *x_t = val->getType()->getScalarType();

    if (x_t->isFloatingPointTy()) {
        // Build the zero constant.
        auto *zero = llvm_constantfp(s, val->getType(), 0.);

        // Run the comparisons.
        auto *cmp0 = llvm_fcmp_olt(s, zero, val);
        auto *cmp1 = llvm_fcmp_olt(s, val, zero);

        // Convert to int32.
        llvm::Type *int_type = make_vector_type(builder.getInt32Ty(), get_vector_size(val));
        auto *icmp0 = builder.CreateZExt(cmp0, int_type);
        auto *icmp1 = builder.CreateZExt(cmp1, int_type);

        // Compute and return the result.
        return builder.CreateSub(icmp0, icmp1);
    }

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(val->getType()) != 0) {
        return llvm_real_sgn(s, val);
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(fmt::format("Invalid type '{}' encountered in the LLVM implementation of sgn()",
                                            llvm_type_name(val->getType())));
    // LCOV_EXCL_STOP
}

// is_finite().
llvm::Value *llvm_is_finite(llvm_state &s, llvm::Value *x)
{
    assert(x != nullptr);

    auto *x_t = x->getType();

    if (x_t->getScalarType()->isFloatingPointTy()) {
        // Codegen +- inf.
        auto *pinf = llvm_codegen(s, x_t, number{std::numeric_limits<double>::infinity()});
        auto *minf = llvm_codegen(s, x_t, number{-std::numeric_limits<double>::infinity()});

        // Check that if x is not +- inf or NaN.
        auto *x_not_pinf = llvm_fcmp_one(s, x, pinf);
        auto *x_not_minf = llvm_fcmp_one(s, x, minf);

        // Put the conditions together and return.
        return s.builder().CreateLogicalAnd(x_not_pinf, x_not_minf);
#if defined(HEYOKA_HAVE_REAL)
    } else if (llvm_is_real(x_t) != 0) {
        return llvm_real_isfinite(s, x);
#endif
        // LCOV_EXCL_START
        //
        // NOLINTNEXTLINE(readability-inconsistent-ifelse-braces)
    } else [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid type '{}' encountered in the LLVM implementation of is_finite()", llvm_type_name(x_t)));
    }
    // LCOV_EXCL_STOP
}

// Check if the input floating-point value is a natural number.
llvm::Value *llvm_is_natural(llvm_state &s, llvm::Value *x)
{
    // Is x finite?
    auto *x_finite = llvm_is_finite(s, x);

    // Is x>=0?
    auto *x_ge_0 = llvm_fcmp_oge(s, x, llvm_codegen(s, x->getType(), number{0.}));

    // Is x an integral value?
    auto *x_int = llvm_fcmp_oeq(s, x, llvm_trunc(s, x));

    // Put the conditions together and return.
    auto &bld = s.builder();
    auto *ret = bld.CreateLogicalAnd(x_finite, x_ge_0);
    ret = bld.CreateLogicalAnd(ret, x_int);

    return ret;
}

} // namespace detail

HEYOKA_END_NAMESPACE
