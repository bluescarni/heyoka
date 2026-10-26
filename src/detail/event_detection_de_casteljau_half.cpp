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
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/Constant.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
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

// Manually-unrolled version of llvm_ed_ival_de_casteljau_half().
//
// NOTE: the de Casteljau algorithm constructs the outputs by building a triangle of averages and then reading the new
// coefficients off the sides of the triangle.
//
// Example: the initial Bernstein coefficients are (0, 0, 0, 8), the base of the triangle:
//
// row 0:  0   0   0   8
// row 1:    0   0   4
// row 2:      0   2
// row 3:        1
//
// Each term in the next row is constructed by averaging the two terms above it from the previous row.
//
// Left result, from the left side: (0, 0, 0, 1).
// Right result, from the reversed right side: (1, 2, 4, 8).
std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ed_ival_de_casteljau_half_unrolled(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs,
                                        const std::uint32_t op1)
{
    // Prepare the return values.
    std::vector<llvm::Value *> ret_first, ret_second;
    ret_first.reserve(op1);
    ret_second.reserve(op1);

    // Generate the 0.5 constant for the value type of ivg.
    auto *const half_const = llvm_constantfp(ivg.get_state(), ivg.get_val_t(), 0.5);

    // Init the running row of the triangle from cfs. This is originally the base row, it will be updated in-place to
    // produce the next rows.
    auto row = cfs;

    // Add the initial results.
    ret_first.push_back(row.front());
    // NOTE: results are added in reversed form to ret_second, we will need to reverse this at the end.
    ret_second.push_back(row.back());

    // Process the remaining rows of the triangle.
    for (std::uint32_t i = 1; i < op1; ++i) {
        assert(row.size() > 1u);

        for (decltype(row.size()) j = 1; j < row.size(); ++j) {
            // Compute the new row element.
            auto *const sum = ivg.ival_add(row[j - 1u], row[j]);
            auto *const res = ivg.ival_mul_nn_val(sum, half_const);

            // Assign it.
            row[j - 1u] = res;
        }

        // Pop back the last row element (leftover from the previous row) in order to finish constructing the current
        // row.
        row.pop_back();

        // Append the results.
        ret_first.push_back(row.front());
        ret_second.push_back(row.back());
    }

    assert(row.size() == 1u);

    // Reverse ret_second before returning.
    std::ranges::reverse(ret_second);

    return std::make_pair(std::move(ret_first), std::move(ret_second));
}

// Loop version of llvm_ed_ival_de_casteljau_half().
std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ed_ival_de_casteljau_half_loop(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs,
                                    const std::uint32_t op1)
{
    // Get the state and builder from ivg.
    auto &s = ivg.get_state();
    auto &bld = s.builder();

    // Codegen a few constants.
    auto *const val_0 = bld.getInt32(0);
    auto *const val_1 = bld.getInt32(1);
    auto *const val_op1 = bld.getInt32(op1);
    auto *const half_const = llvm_constantfp(s, ivg.get_val_t(), 0.5);

    // NOTE: we need two arrays, one for the running row and the other for the left result. The right result will be
    // contained in the running row at the end of the algorithm.
    auto *const ival_t = ivg.get_ival_t();
    auto *const array_t = llvm::ArrayType::get(ival_t, op1);
    auto *const row_arr_alloca = llvm_entry_alloca(s, array_t);
    auto *const left_res_alloca = llvm_entry_alloca(s, array_t);

    // Helper to set a value into the left result: left_res[idx] = x.
    const auto set_left = [&bld, array_t, left_res_alloca, val_0](llvm::Value *const idx, llvm::Value *const x) {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, left_res_alloca, {val_0, idx});
        bld.CreateStore(x, ptr);
    };

    // Helper to get a value from the left result: return left_res[idx].
    const auto get_left = [&bld, array_t, left_res_alloca, val_0, ival_t](llvm::Value *const idx) -> llvm::Value * {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, left_res_alloca, {val_0, idx});
        return bld.CreateLoad(ival_t, ptr);
    };

    // Helper to set a value into the running row: row[idx] = x.
    const auto set_row = [&bld, array_t, row_arr_alloca, val_0](llvm::Value *const idx, llvm::Value *const x) {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, row_arr_alloca, {val_0, idx});
        bld.CreateStore(x, ptr);
    };

    // Helper to get a value from the running row: return row[idx].
    const auto get_row = [&bld, array_t, row_arr_alloca, val_0, ival_t](llvm::Value *const idx) -> llvm::Value * {
        auto *const ptr = bld.CreateInBoundsGEP(array_t, row_arr_alloca, {val_0, idx});
        return bld.CreateLoad(ival_t, ptr);
    };

    // Copy the original coefficients into the row to initialise it.
    for (std::uint32_t idx = 0; idx < op1; ++idx) {
        set_row(bld.getInt32(idx), cfs[idx]);
    }

    // Set the first value in the left result.
    set_left(val_0, cfs[0]);

    // Process the remaining rows of the triangle.
    llvm_loop_u32(
        s, val_1, val_op1,
        [&bld, val_op1, val_1, &s, &ivg, &get_row, half_const, &set_row, &set_left, val_0](llvm::Value *const i) {
            // NOTE: the current size of the running row is op1 - i + 1.
            auto *cur_row_size = bld.CreateSub(val_op1, i);
            cur_row_size = bld.CreateAdd(cur_row_size, val_1);

            // Inner loop.
            llvm_loop_u32(s, val_1, cur_row_size,
                          [&ivg, &get_row, &bld, val_1, half_const, &set_row](llvm::Value *const j) {
                              // Compute the new row element.
                              auto *const jm1 = bld.CreateSub(j, val_1);
                              auto *const sum = ivg.ival_add(get_row(jm1), get_row(j));
                              auto *const res = ivg.ival_mul_nn_val(sum, half_const);

                              // Assign it.
                              set_row(jm1, res);
                          });

            // Store the first element of the new row into the left result.
            set_left(i, get_row(val_0));
        });

    // Produce the results, reading out the values from the arrays.
    std::vector<llvm::Value *> ret_first, ret_second;
    ret_first.reserve(op1);
    ret_second.reserve(op1);

    for (std::uint32_t i = 0; i < op1; ++i) {
        auto *const idx = bld.getInt32(i);
        ret_first.push_back(get_left(idx));
        ret_second.push_back(get_row(idx));
    }

    return std::make_pair(std::move(ret_first), std::move(ret_second));
}

} // namespace

// Helper to split the input Bernstein polynomial at the midpoint of its interval.
//
// cfs contains the coefficients of a polynomial expressed in the Bernstein basis of the [a, b] interval, with interval
// coefficients stored in the (-lo, hi) format (that is, the lower bound of the interval is carried around negated) in
// ascending order:
//
// [(-b0_lo, b0_hi), (-b1_lo, b1_hi), (-b2_lo, b2_hi), ...]
//
// This function will produce the two Bernstein polynomials resulting from expressing the input polynomial in the
// Bernstein bases of the intervals [a, m] and [m, b], with m = (a+b)/2. The two polynomials are returned as a pair, in
// this order. The calculations will be performed using the interval arithmetic primitives from the input guard ivg, and
// the resulting polynomials will be represented in the same format explained above. The IR is emitted at the current
// insertion point, which must be within the function guarded by ivg.
//
// cfs is assumed non-empty. All elements of cfs are assumed to have the same type as ivg.get_ival_t() and to represent
// valid intervals. The output intervals enclose the exact coefficients, and they are guaranteed to be valid.
std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ed_ival_de_casteljau_half(const llvm_ival_guard &ivg, const std::vector<llvm::Value *> &cfs)
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
        return llvm_ed_ival_de_casteljau_half_unrolled(ivg, cfs, op1);
    } else {
        return llvm_ed_ival_de_casteljau_half_loop(ivg, cfs, op1);
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
