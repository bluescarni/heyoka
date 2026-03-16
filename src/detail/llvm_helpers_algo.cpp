// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Implementation of the std::upper_bound() algorithm for floating-point types.
//
// Given an array of scalar values sorted in ascending order beginning at ptr and of size arr_size (a 32-bit int), this
// function will return the index of the first element in the array that is *greater than* v. If no such element exists,
// arr_size will be returned. v can be a scalar or a vector. No element in the array can be NaN, but v can be NaN. If v
// is NaN, it is considered greater than any value in the array.
//
// The algorithm is short enough to be reproduced here:
//
// template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type, class Compare>
// ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
// {
//     ForwardIt it;
//     typename std::iterator_traits<ForwardIt>::difference_type count, step;
//     count = std::distance(first, last);
//
//     while (count > 0)
//     {
//         it = first;
//         step = count / 2;
//         std::advance(it, step);
//
//         if (!comp(value, *it))
//         {
//             first = ++it;
//             count -= step + 1;
//         }
//         else
//             count = step;
//     }
//
//     return first;
// }
//
// Particular care must be taken for the vector implementation: while in a scalar implementation the bisection loop is
// never entered if count == 0, in the vector implementation we will be entering the bisection loop with count == 0
// whenever a SIMD lane has finished but the other SIMD lanes have not. In a loop iteration with count == 0, the
// following happens:
//
// - step is set to 0 and 'it' remains inited to 'first';
// - 'it' may be pointing one past the end of the array, and thus we must take care
//   of *not* dereferencing it if that is the case. We thus have two possibilities:
//   - 'it' points somewhere in the array: in this case, we know that 'it' points to
//     the first array element greater than 'v', and thus '!comp(value, *it)' evaluates
//     to false, we end up in the 'count = step' branch, where count is set again
//     to 0. Thus, neither 'first' nor 'count' are altered;
//   - 'it' points one past the end: in this case, we must avoid reading from it and we must
//     replace the condition '!comp(value, *it)' with 'false', so that we end up in the
//     'count = step' branch.
llvm::Value *llvm_upper_bound(llvm_state &s, llvm::Value *ptr, llvm::Value *arr_size, llvm::Value *v)
{
    assert(ptr != nullptr);
    // NOTE: this will also check that ptr is not a vector of pointers.
    assert(ptr->getType()->isPointerTy());
    assert(arr_size != nullptr);
    assert(v != nullptr);

    auto &bld = s.builder();

    // NOTE: infer the scalar type from v.
    auto *scal_t = v->getType()->getScalarType();

    // Fetch the 32-bit int type.
    auto *int32_tp = bld.getInt32Ty();
    // NOTE: this will also check that arr_size is not a vector of values.
    assert(arr_size->getType() == int32_tp);

#if !defined(NDEBUG)

    // Validate the input array.
    llvm_loop_u32(s, bld.getInt32(0), arr_size, [&bld, &s, scal_t, ptr, arr_size](llvm::Value *cur_idx) {
        // Load the current value.
        auto *cur_val = bld.CreateLoad(scal_t, bld.CreateInBoundsGEP(scal_t, ptr, {cur_idx}));

        // Check that it is not NaN.
        llvm_assert(s, llvm_fcmp_ord(s, cur_val, cur_val));

        // Check that it is less than or equal to the next value (if available).
        auto *next_idx = bld.CreateAdd(cur_idx, bld.getInt32(1));
        llvm_if_then_else(
            s, bld.CreateICmpEQ(next_idx, arr_size), []() {},
            [&bld, &s, cur_val, scal_t, ptr, next_idx]() {
                auto *next_val = bld.CreateLoad(scal_t, bld.CreateInBoundsGEP(scal_t, ptr, {next_idx}));

                llvm_assert(s, llvm_fcmp_ole(s, cur_val, next_val));
            });
    });

#endif

    // Determine the batch size.
    std::uint32_t batch_size = 1;
    if (auto *vec_t = llvm::dyn_cast<llvm::FixedVectorType>(v->getType())) {
        batch_size = boost::numeric_cast<std::uint32_t>(vec_t->getNumElements());
        assert(batch_size != 0u);
    }

    // Splat ptr to the batch size.
    auto *arr_ptr = vector_splat(bld, ptr, batch_size);

    // The type to be used for indexing into the scalar array.
    auto *idx_vec_t = make_vector_type(int32_tp, batch_size);

    // The type that will be loaded from the scalar array.
    auto *fp_vec_t = make_vector_type(scal_t, batch_size);

    // Create arrays of 32-bit constants for use below.
    auto *zero_vec_i32 = vector_splat(bld, bld.getInt32(0), batch_size);
    auto *one_vec_i32 = vector_splat(bld, bld.getInt32(1), batch_size);
    auto *two_vec_i32 = vector_splat(bld, bld.getInt32(2), batch_size);

    // Create the 'first' and 'count' variables.
    // NOTE: as "iterator" type we will be using a 32-bit int.
    auto *first = bld.CreateAlloca(idx_vec_t);
    auto *count = bld.CreateAlloca(idx_vec_t);

    // The 'first' iterator is inited with zeros (as it points to the beginning of the array).
    bld.CreateStore(zero_vec_i32, first);

    // 'count' is inited with the size of the array.
    auto *arr_size_splat = vector_splat(bld, arr_size, batch_size);
    bld.CreateStore(arr_size_splat, count);

    // Iterate as long as all elements of 'count' are > 0.
    llvm_while_loop(
        s,
        [&bld, count, idx_vec_t, zero_vec_i32, batch_size]() -> llvm::Value * {
            auto *cmp = bld.CreateICmpUGT(bld.CreateLoad(idx_vec_t, count), zero_vec_i32);

            // NOTE: in scalar mode, no reduction is needed.
            return (batch_size == 1u) ? cmp : bld.CreateOrReduce(cmp);
        },
        [&bld, &s, first, count, idx_vec_t, batch_size, scal_t, fp_vec_t, v, one_vec_i32, arr_ptr, arr_size_splat,
         two_vec_i32]() {
            // Load the value stored in 'first' - this will be the iterator we will
            // be using in the current iteration of the loop.
            llvm::Value *cur_first = bld.CreateLoad(idx_vec_t, first);
            auto *it = cur_first;

            // Compute the step value for the current iteration: step = count / 2.
            llvm::Value *cur_count = bld.CreateLoad(idx_vec_t, count);
            auto *step = bld.CreateUDiv(cur_count, two_vec_i32);

            // Advance 'it' by step.
            it = bld.CreateAdd(it, step);

            // Load the value(s) from 'it' into 'cur_value'. 'mask' is used only in vector mode, otherwise it
            // remains null.
            llvm::Value *cur_value{}, *mask{};
            if (batch_size == 1u) {
                // Normal scalar load.
                HEYOKA_LLVM_ASSERT(s, bld.CreateICmpULT(it, arr_size_splat));
                cur_value = bld.CreateLoad(scal_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}));
            } else {
                // NOTE: as explained above, in vector mode we must take care to avoid loading from 'it'
                // if it points one past the end. We accomplish this with a masked gather.

                // Fetch the alignment of the scalar type.
                const auto align = get_alignment(s.module(), scal_t);

                // Identify the SIMD lane(s) which are *not* reading past the end of the array, storing the
                // result in 'mask'.
                mask = bld.CreateICmpNE(it, arr_size_splat);

                // As a passthru value for the masked gather, use the v value itself.
                auto *passthru = v;

                // Masked gather with passthru.
                cur_value = bld.CreateMaskedGather(fp_vec_t, bld.CreateInBoundsGEP(scal_t, arr_ptr, {it}),
                                                   llvm::Align(align), mask, passthru);
            }

            // Run the comparison.
            // NOTE: the original comparison would be '!comp(value, *it)', which translates to '!(v < *it)' in
            // the current code. *it can never be NaN (apart from a corner case in vector mode when the passthru
            // value is NaN, but we take care of this later ANDing the mask). v could be NaN, in which case we want
            // '!(v < *it)' to evaluate to true because we want to consider NaN greater than non-NaN. In order to do
            // this, we flip the comparison around to '*it <= v', and we implement it via the ULE predicate, which
            // returns true if either v is NaN or '*it <= v'.
            auto *cmp = llvm_fcmp_ule(s, cur_value, v);
            if (batch_size != 1u) {
                // NOTE: in vector mode, we must take care that cmp for the masked-out lanes evaluates to false.
                assert(mask != nullptr);
                cmp = bld.CreateAnd(cmp, mask);
            }

            // We now need to update 'first' and 'count'. Branch on the batch size for efficiency.
            if (batch_size == 1u) {
                // Scalar implementation.
                llvm_if_then_else(
                    s, cmp,
                    [&bld, one_vec_i32, it, first, count, step, cur_count]() {
                        // Assign it + 1 to first.
                        auto *itp1 = bld.CreateAdd(it, one_vec_i32);
                        bld.CreateStore(itp1, first);

                        // Compute new_count = count - (step + 1).
                        auto *stepp1 = bld.CreateAdd(step, one_vec_i32);
                        auto *new_count = bld.CreateSub(cur_count, stepp1);

                        // Assign new_count to count.
                        bld.CreateStore(new_count, count);
                    },
                    [&bld, step, count]() {
                        // NOTE: no update of 'first' needed here.
                        // Assign step to count.
                        bld.CreateStore(step, count);
                    });
            } else {
                // Vector implementation.

                // Compute it + 1.
                auto *itp1 = bld.CreateAdd(it, one_vec_i32);

                // Compute step + 1.
                auto *stepp1 = bld.CreateAdd(step, one_vec_i32);

                // Compute count - (step + 1).
                auto *count_m_stepp1 = bld.CreateSub(cur_count, stepp1);

                // Compute the new first = cmp ? (it + 1) : cur_first.
                auto *new_first = bld.CreateSelect(cmp, itp1, cur_first);

                // Compute the new count = cmp ? (count - (step + 1)) : step.
                auto *new_count = bld.CreateSelect(cmp, count_m_stepp1, step);

                // Store the new first and the new count.
                bld.CreateStore(new_count, count);
                bld.CreateStore(new_first, first);
            }
        });

    // The return value is the value stored in 'first'.
    return bld.CreateLoad(idx_vec_t, first);
}

// Pairwise reduction of a vector of LLVM values.
llvm::Value *pairwise_reduce(std::vector<llvm::Value *> &vals,
                             const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &f)
{
    assert(!vals.empty());
    assert(f);

    // LCOV_EXCL_START
    if (vals.size() == std::numeric_limits<decltype(vals.size())>::max()) {
        throw std::overflow_error("Overflow detected in pairwise_reduce()");
    }
    // LCOV_EXCL_STOP

    while (vals.size() != 1u) {
        std::vector<llvm::Value *> new_vals;

        for (decltype(vals.size()) i = 0; i < vals.size(); i += 2u) {
            if (i + 1u == vals.size()) {
                // We are at the last element of the vector
                // and the size of the vector is odd. Just append
                // the existing value.
                new_vals.push_back(vals[i]);
            } else {
                new_vals.push_back(f(vals[i], vals[i + 1u]));
            }
        }

        new_vals.swap(vals);
    }

    return vals[0];
}

// Pairwise summation of a vector of LLVM values.
// https://en.wikipedia.org/wiki/Pairwise_summation
llvm::Value *pairwise_sum(llvm_state &s, std::vector<llvm::Value *> &sum)
{
    return pairwise_reduce(sum, [&s](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return llvm_fadd(s, a, b); });
}

// Pairwise product of a vector of LLVM values.
llvm::Value *pairwise_prod(llvm_state &s, std::vector<llvm::Value *> &prod)
{
    return pairwise_reduce(prod, [&s](llvm::Value *a, llvm::Value *b) -> llvm::Value * { return llvm_fmul(s, a, b); });
}

} // namespace detail

HEYOKA_END_NAMESPACE
