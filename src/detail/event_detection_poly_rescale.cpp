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

#include <llvm/IR/Constant.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper implementing polynomial rescaling in interval arithmetic.
//
// cfs contains the scalar coefficients in the monomial basis of an input polynomial p(x), stored as usual in ascending
// order. h is the scale factor. This function will compute the coefficients, in the monomial basis, of the rescaled
// polynomial p(h*x). The computation is rigorous, returning interval coefficients in the (-lo, hi) format (that is, the
// lower bound of the interval is carried around negated), stored in ascending order in split format:
//
// [-c0_lo, -c1_lo, -c2_lo, ..., c0_hi, c1_hi, c2_hi, ...]
//
// (so that the rescaled polynomial is p(h*x) = [c0_lo, c0_hi] + [c1_lo, c1_hi] * x + [c2_lo, c2_hi] * x**2 + ...). The
// calculations will be performed using the arithmetic primitives from the input guard ivg. The IR is emitted at the
// current insertion point, which must be within the function guarded by ivg.
//
// cfs is assumed non-empty. h and all elements of cfs are assumed to have the same type as the input guard ivg. All
// input values are assumed to be finite. The output components are valid bounds, which may be +inf in case of overflow
// but never NaN.
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

    // Determine if h is negative.
    auto *const zero = llvm_constantfp(s, ivg.get_val_t(), 0.);
    auto *const h_neg = ivg.lt(h, zero);

    // Init the return value.
    std::vector<llvm::Value *> retval;
    retval.resize(op1 * boost::safe_numerics::safe<decltype(cfs.size())>(2));

    // c0 is unaltered.
    retval[0] = ivg.neg(cfs[0]);
    retval[op1] = cfs[0];

    // Compute |h| and init its running power as (-|h|, |h|).
    auto *const abs_h = ivg.abs(h);
    auto *pow_abs_h_mlo = ivg.neg(abs_h);
    auto *pow_abs_h_hi = abs_h;

    // Compute the remaining coefficients.
    auto &bld = s.builder();
    for (decltype(cfs.size()) i = 1; i < op1; ++i) {
        // Fetch the current scalar coefficient.
        auto *cf = cfs[i];

        // Negate odd-degree coefficients if h < 0.
        if (i % 2u != 0u) {
            cf = bld.CreateSelect(h_neg, ivg.neg(cf), cf);
        }

        // Compute the rescaled coefficient.
        //
        // NOTE: here the computation is [c, c] * [lo, hi], with [lo, hi] guaranteed to be a non-negative interval. The
        // standard interval arithmetic formula for the product then simplifies to [c*lo, c*hi] if c >= 0, [c*hi, c*lo]
        // otherwise.
        //
        // Because we are storing [lo, hi] as (mlo, hi) (where mlo = -lo), the computations actually become:
        //
        // - (c*mlo, c*hi) if c>=0,
        // - (c*-hi, c*-mlo) if c<0,
        //
        // with all the multiplications rounding up. In order to reduce operations, we can rewrite the result as (|c| *
        // select(c<0, hi, mlo), |c| * select(0<c, hi, mlo)). Note that the second select excludes c==0 in the
        // condition. This is done on purpose, as hi may become +inf in case of overflow (see below). If this happens
        // and c==0, we want to avoid a 0 * inf multiplication which would (wrongly) produce NaN: since the inf was the
        // result of an overflow (and not a pole like 1/0), in this case we are ok with the 0 result that is produced by
        // the other branch.
        auto *const abs_cf = ivg.abs(cf);
        auto *const new_cf_mlo = ivg.mul_up(abs_cf, bld.CreateSelect(ivg.lt(cf, zero), pow_abs_h_hi, pow_abs_h_mlo));
        auto *const new_cf_hi = ivg.mul_up(abs_cf, bld.CreateSelect(ivg.lt(zero, cf), pow_abs_h_hi, pow_abs_h_mlo));

        // Write the coefficients in the return value.
        retval[i] = new_cf_mlo;
        retval[i + op1] = new_cf_hi;

        // Update pow_abs_h for the next iteration, if necessary.
        if (i != op1 - 1u) {
            // NOTE: here we are computing [lo, hi] * |h|, with [lo, hi] guaranteed to be a non-negative interval.
            // Hence, the simplified formula for interval/scalar multiplication.
            //
            // NOTE: here pow_abs_h_hi can overflow to +inf, while pow_abs_h_mlo can never overflow to -inf: it is
            // non-positive and rounding up stops it at -max.
            pow_abs_h_mlo = ivg.mul_up(pow_abs_h_mlo, abs_h);
            pow_abs_h_hi = ivg.mul_up(pow_abs_h_hi, abs_h);
        }
    }

    return retval;
}

} // namespace detail

HEYOKA_END_NAMESPACE
