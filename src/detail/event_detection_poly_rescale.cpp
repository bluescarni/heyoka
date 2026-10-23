// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <vector>

#include <llvm/ADT/APInt.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to flip the sign bit of x if the sign bit of y is set (i.e., to XOR the sign bit of y into x). For non-NaN
// values, this corresponds to changing the sign of x if y<0, except that the sign change will also take place if y is
// -0.
//
// x and y must be scalar float or double values of the same type.
llvm::Value *xor_signbit(llvm_state &s, llvm::Value *const x, llvm::Value *const y)
{
    assert(x != nullptr);
    assert(y != nullptr);
    assert(x->getType() == y->getType());
    assert(x->getType()->isFloatTy() || x->getType()->isDoubleTy());

    auto &bld = s.builder();

    // Fetch the integer type with the same size as x.
    auto *const int_t = bld.getIntNTy(x->getType()->getScalarSizeInBits());

    // Isolate the sign bit of y.
    auto *const sign_mask = bld.getInt(llvm::APInt::getSignMask(int_t->getBitWidth()));
    auto *const y_sign = bld.CreateAnd(bld.CreateBitCast(y, int_t), sign_mask);

    // XOR it into x and convert back.
    return bld.CreateBitCast(bld.CreateXor(bld.CreateBitCast(x, int_t), y_sign), x->getType());
}

} // namespace

// Helper implementing polynomial rescaling in interval arithmetic.
//
// cfs contains the scalar coefficients in the monomial basis of an input polynomial p(x), stored as usual in ascending
// order. h is the scale factor, also a scalar. This function will compute the coefficients, in the monomial basis, of
// the rescaled polynomial p(h*x). The computation is rigorous, returning interval coefficients in the (-lo, hi) format
// (that is, the lower bound of the interval is carried around negated), stored in ascending order:
//
// [(-c0_lo, c0_hi), (-c1_lo, c1_hi), (-c2_lo, c2_hi), ...]
//
// (so that the rescaled polynomial is p(h*x) = [c0_lo, c0_hi] + [c1_lo, c1_hi] * x + [c2_lo, c2_hi] * x**2 + ...). The
// calculations will be performed using the arithmetic primitives from the input guard ivg. The IR is emitted at the
// current insertion point, which must be within the function guarded by ivg.
//
// cfs is assumed non-empty. h and all elements of cfs are assumed to have the same type as ivg.get_val_t(). All input
// values are assumed to be finite. The output intervals enclose the exact coefficients, and they have no NaN bounds.
// Infinite bounds may be returned in case of overflow, but never in the invalid forms [-inf, -inf] and [+inf, +inf].
std::vector<llvm::Value *> llvm_ed_ival_poly_rescale(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs,
                                                     llvm::Value *const h)
{
    assert(!cfs.empty());
    assert(h != nullptr);
    assert(h->getType() == ivg.get_val_t());
    assert(std::ranges::all_of(cfs, [h](auto *const x) { return x != nullptr && x->getType() == h->getType(); }));

    auto &s = ivg.get_state();

    // Fetch the poly degree+1.
    const auto op1 = cfs.size();

    // Determine whether the odd-degree coefficients are negated via sign bit manipulation (see below).
    auto *const val_t = ivg.get_val_t();
    const auto use_signbit = val_t->isFloatTy() || val_t->isDoubleTy();

    // Determine if h is negative (not needed if use_signbit is true).
    auto *const h_neg = use_signbit ? nullptr : ivg.val_lt(h, llvm_constantfp(s, val_t, 0.));

    // Init the return value.
    std::vector<llvm::Value *> retval;
    retval.resize(op1);

    // c0 is unaltered.
    retval[0] = ivg.ival_construct(ivg.val_neg(cfs[0]), cfs[0]);

    // Compute |h| and init its running power as (-|h|, |h|).
    auto *const abs_h = ivg.val_abs(h);
    auto *pow_abs_h = ivg.ival_construct(ivg.val_neg(abs_h), abs_h);

    // Compute the remaining coefficients.
    auto &bld = s.builder();
    for (decltype(cfs.size()) i = 1; i < op1; ++i) {
        // Fetch the current scalar coefficient.
        auto *cf = cfs[i];

        // Negate odd-degree coefficients if h < 0.
        //
        // NOTE: for scalar float and double, on x86-64 LLVM turns the select into a branch, which results in heavy
        // register spilling. Thus, in this case we XOR the sign bit of h into cf instead. This also negates cf when h
        // is -0, which is harmless since cf * |h|**i is zero in this case.
        if (i % 2u != 0u) {
            cf = use_signbit ? xor_signbit(s, cf, h) : bld.CreateSelect(h_neg, ivg.val_neg(cf), cf);
        }

        // Compute the rescaled coefficient.
        auto *const new_cf = ivg.val_mul_nn_ival(cf, pow_abs_h);

        // Write it into the return value.
        retval[i] = new_cf;

        // Update pow_abs_h for the next iteration, if necessary.
        if (i != op1 - 1u) {
            pow_abs_h = ivg.ival_mul_nn_val(pow_abs_h, abs_h);
        }
    }

    return retval;
}

} // namespace detail

HEYOKA_END_NAMESPACE
