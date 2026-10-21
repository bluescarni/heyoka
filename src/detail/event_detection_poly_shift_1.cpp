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

// Helper implementing polynomial Taylor shifting by 1 in interval arithmetic.
//
// cfs is an input polynomial p(x) = c0 + c1*x + c2*x**2 + ... with the coefficients represented as intervals [lo, hi]
// in the (-lo, hi) format (that is, the lower bound of the interval is carried around negated). The coefficients are
// stored in the cfs vector in ascending order in split format, that is:
//
// [-c0_lo, -c1_lo, -c2_lo, ..., c0_hi, c1_hi, c2_hi, ...]
//
// This function will compute and return in the same format intervals enclosing the coefficients of the polynomial
// p(x+1), using the arithmetic primitives from the input guard ivg. It is assumed that all values in cfs are of type
// ivg.get_val_t(). cfs must not be empty and its size must be an even number. The IR is emitted at the current
// insertion point, which must be within the function guarded by ivg.
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
    assert(cfs.size() % 2u == 0u);
    assert(std::ranges::all_of(cfs, [&ivg](auto *const x) { return x != nullptr && x->getType() == ivg.get_val_t(); }));

    // Make sure we can use 32-bit ints for all indexing.
    const auto cfs_size = boost::numeric_cast<std::uint32_t>(cfs.size());

    // Fetch the poly degree + 1.
    const auto op1 = cfs_size / 2u;

    // Fetch the val_t from ivg.
    auto *const val_t = ivg.get_val_t();

    // Get the state and builder from ivg.
    auto &s = ivg.get_state();
    auto &bld = s.builder();

    // Codegen a few constants.
    auto *val_0 = bld.getInt32(0);
    auto *val_1 = bld.getInt32(1);
    auto *val_op1 = bld.getInt32(op1);

    // NOTE: the loops below index the coefficients at runtime, thus the coefficients must be stored in memory. Since
    // the algorithm operates in-place, a single alloca array suffices.
    auto *array_t = llvm::ArrayType::get(val_t, cfs_size);
    auto *cfs_arr_alloca = llvm_entry_alloca(s, array_t);
    for (std::uint32_t idx = 0; idx < cfs_size; ++idx) {
        auto *ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.getInt32(idx)});
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
            // Compute indices and pointers in cfs_arr_alloca of a[i] and a[i - 1] from the reference.

            // ai_idx = op1 - 1 - i.
            auto *ai_idx = bld.CreateSub(val_op1, val_1);
            ai_idx = bld.CreateSub(ai_idx, i);
            auto *ai_mlo_ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, ai_idx});
            auto *ai_hi_ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.CreateAdd(ai_idx, val_op1)});

            // aim1_idx = op1 - i.
            auto *aim1_idx = bld.CreateSub(val_op1, i);
            auto *aim1_mlo_ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, aim1_idx});
            auto *aim1_hi_ptr
                = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.CreateAdd(aim1_idx, val_op1)});

            // Compute the negated lo part of the result.
            auto *ai_mlo = bld.CreateLoad(val_t, ai_mlo_ptr);
            auto *aim1_mlo = bld.CreateLoad(val_t, aim1_mlo_ptr);
            auto *ret_mlo = ivg.add_up(ai_mlo, aim1_mlo);

            // Compute the hi part of the result.
            auto *ai_hi = bld.CreateLoad(val_t, ai_hi_ptr);
            auto *aim1_hi = bld.CreateLoad(val_t, aim1_hi_ptr);
            auto *ret_hi = ivg.add_up(ai_hi, aim1_hi);

            // Update the a[i] values in cfs_arr_alloca.
            bld.CreateStore(ret_mlo, ai_mlo_ptr);
            bld.CreateStore(ret_hi, ai_hi_ptr);
        });
    });

    // Init the return value.
    std::vector<llvm::Value *> ret;
    ret.reserve(cfs.size());

    // Load the values from cfs_arr_alloca to fill in the return value.
    for (std::uint32_t idx = 0; idx < cfs_size; ++idx) {
        auto *ptr = bld.CreateInBoundsGEP(array_t, cfs_arr_alloca, {val_0, bld.getInt32(idx)});
        ret.push_back(bld.CreateLoad(val_t, ptr));
    }

    return ret;
}

} // namespace detail

HEYOKA_END_NAMESPACE
