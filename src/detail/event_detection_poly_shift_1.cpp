// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Loop version of llvm_ed_ival_poly_shift_1().
std::vector<llvm::Value *> llvm_ed_ival_poly_shift_1_loop(const llvm_ival_guard &ivg,
                                                          const std::vector<llvm::Value *> &cfs,
                                                          const std::uint32_t op1)
{
    // Get the state and builder from ivg.
    auto &s = ivg.get_state();
    auto &bld = s.builder();

    // Codegen a few constants.
    auto *const val_0 = bld.getInt32(0);
    auto *const val_1 = bld.getInt32(1);
    auto *const val_op1 = bld.getInt32(op1);

    // NOTE: the loops below index the coefficients at runtime, thus the coefficients must be stored in memory. Since
    // the algorithm operates in-place, a single alloca array suffices.
    auto *const ival_t = ivg.get_ival_t();
    auto *const array_t = llvm::ArrayType::get(ival_t, op1);
    auto *const cfs_arr_alloca = llvm_entry_alloca(s, array_t);
    // Copy the original coefficients into the alloca.
    for (std::uint32_t idx = 0; idx < op1; ++idx) {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.getInt32(idx)});
        bld.CreateStore(cfs[idx], ptr);
    }

    // Now we can run the Taylor shift loop.
    //
    // NOTE: the original reference stores the coefficients in descending order, here they are in ascending order. Thus,
    // the index i in the reference corresponds to the index op1 - 1 - i here.
    llvm_loop_u32(s, val_1, val_op1, [&](llvm::Value *j) {
        // Compute the inner loop end: op1 - j + 1.
        auto *inner_loop_end = bld.CreateSub(val_op1, j);
        inner_loop_end = bld.CreateAdd(inner_loop_end, val_1);

        // Inner loop.
        llvm_loop_u32(s, val_1, inner_loop_end, [&](llvm::Value *i) {
            // Compute indices and pointers in cfs_arr_alloca of a[i] and a[i-1] from the reference.

            // ai_idx = op1 - 1 - i.
            auto *ai_idx = bld.CreateSub(val_op1, val_1);
            ai_idx = bld.CreateSub(ai_idx, i);
            auto *const ai_ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, ai_idx});

            // aim1_idx = op1 - i.
            auto *const aim1_idx = bld.CreateSub(val_op1, i);
            auto *const aim1_ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, aim1_idx});

            // Compute the result: a[i] + a[i-1].
            auto *const ai = bld.CreateLoad(ival_t, ai_ptr);
            auto *const aim1 = bld.CreateLoad(ival_t, aim1_ptr);
            auto *const ret = ivg.ival_add(ai, aim1);

            // Update the a[i] value in cfs_arr_alloca.
            bld.CreateStore(ret, ai_ptr);
        });
    });

    // Init the return value.
    std::vector<llvm::Value *> ret;
    ret.reserve(cfs.size());

    // Load the values from cfs_arr_alloca to fill in the return value.
    for (std::uint32_t idx = 0; idx < op1; ++idx) {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.getInt32(idx)});
        ret.push_back(bld.CreateLoad(ival_t, ptr));
    }

    return ret;
}

// Manually-unrolled version of llvm_ed_ival_poly_shift_1().
std::vector<llvm::Value *> llvm_ed_ival_poly_shift_1_unrolled(const llvm_ival_guard &ivg,
                                                              const std::vector<llvm::Value *> &cfs,
                                                              const std::uint32_t op1)
{
    // Init the return value from cfs.
    auto ret = cfs;

    // Run the Taylor shift loop.
    //
    // NOTE: the original reference stores the coefficients in descending order, here they are in ascending order. Thus,
    // the index i in the reference corresponds to the index op1 - 1 - i here.
    for (std::uint32_t j = 1; j < op1; ++j) {
        for (std::uint32_t i = 1; i < op1 - j + 1u; ++i) {
            const auto ai_idx = op1 - 1u - i;
            const auto aim1_idx = op1 - i;

            // Fetch a[i] and a[i-1].
            auto *const ai = ret[ai_idx];
            auto *const aim1 = ret[aim1_idx];

            // Compute the result: a[i] + a[i-1].
            auto *const res = ivg.ival_add(ai, aim1);

            // Update the a[i] value in ret.
            ret[ai_idx] = res;
        }
    }

    return ret;
}

} // namespace

// Helper implementing polynomial Taylor shifting by 1 in interval arithmetic.
//
// cfs is an input polynomial p(x) = c0 + c1*x + c2*x**2 + ... with the coefficients represented as intervals [lo, hi]
// in the (-lo, hi) format (that is, the lower bound of the interval is carried around negated). The coefficients are
// stored in the cfs vector in ascending order, that is:
//
// [(-c0_lo, c0_hi), (-c1_lo, c1_hi), (-c2_lo, c2_hi), ...]
//
// This function will compute and return in the same format intervals enclosing the coefficients of the polynomial
// p(x+1), using the arithmetic primitives from the input guard ivg. It is assumed that all values in cfs are of type
// ivg.get_ival_t(). cfs must not be empty. The IR is emitted at the current insertion point, which must be within the
// function guarded by ivg.
//
// The input intervals are assumed to have no NaN bounds. Infinite bounds may be present, but never in the invalid forms
// [-inf, -inf] and [+inf, +inf]. Similarly, the output intervals have no NaN bounds and are never in the invalid forms
// [-inf, -inf] and [+inf, +inf].
//
// We use the "Optimized Algorithm for Horner Polynomial Taylor shift with real coefficients" algorithm from here
//
// https://www.researchgate.net/publication/368987410_Polynomial_Taylor_Shifting
//
// In our case, shift==1.
std::vector<llvm::Value *> llvm_ed_ival_poly_shift_1(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs)
{
    assert(!cfs.empty());
    assert(
        std::ranges::all_of(cfs, [&ivg](auto *const x) { return x != nullptr && x->getType() == ivg.get_ival_t(); }));

    // Make sure we can use 32-bit ints for all indexing.
    const auto op1 = boost::numeric_cast<std::uint32_t>(cfs.size());

    // Fetch the scalar type of the interval bounds from ivg.
    const auto *const scal_t = ivg.get_val_t()->getScalarType();

    // NOTE: if we are working with float/double scalars/vectors and the poly degree is small enough, we want to
    // manually unroll the computation in order to improve performance. For the other floating-point types, we expect
    // the unrolling not to improve performance.
    if ((scal_t->isFloatTy() || scal_t->isDoubleTy()) && op1 < 25u) {
        return llvm_ed_ival_poly_shift_1_unrolled(ivg, cfs, op1);
    } else {
        return llvm_ed_ival_poly_shift_1_loop(ivg, cfs, op1);
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
