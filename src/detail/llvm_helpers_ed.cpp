// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Add a function to count the number of sign changes in the coefficients
// of a polynomial of degree n. The coefficients are SIMD vectors of size batch_size
// and scalar type scal_t.
llvm::Function *llvm_add_csc(llvm_state &s, llvm::Type *scal_t, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding a sign changes counter function");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(scal_t);

    // Fetch the vector floating-point type.
    auto *tp = make_vector_type(scal_t, batch_size);

    // Fetch the function name.
    const auto fname = fmt::format("heyoka_csc_degree_{}_{}", n, llvm_mangle_type(tp));

    // The function arguments:
    //
    // - pointer to the return value,
    // - pointer to the array of coefficients.
    //
    // NOTE: both pointers are to the scalar counterparts of the vector types, so that we can call this from regular C++
    // code. The second pointer is to an external type.
    const std::vector<llvm::Type *> fargs(2, llvm::PointerType::getUnqual(context));

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is void.
        auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *out_ptr = f->args().begin();
        out_ptr->setName("out_ptr");
        llvm_add_no_capture_argattr(s, out_ptr);
        out_ptr->addAttr(llvm::Attribute::NoAlias);
        out_ptr->addAttr(llvm::Attribute::WriteOnly);

        auto *cf_ptr = f->args().begin() + 1;
        cf_ptr->setName("cf_ptr");
        llvm_add_no_capture_argattr(s, cf_ptr);
        cf_ptr->addAttr(llvm::Attribute::NoAlias);
        cf_ptr->addAttr(llvm::Attribute::ReadOnly);

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Fetch the type for storing the last_nz_idx variable.
        auto *last_nz_idx_t = make_vector_type(builder.getInt32Ty(), batch_size);

        // The initial last nz idx is zero for all batch elements.
        auto *last_nz_idx = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::ConstantInt::get(last_nz_idx_t, 0u), last_nz_idx);

        // NOTE: last_nz_idx is an index into the poly coefficient vector. Thus, in batch
        // mode, when loading from a vector of indices, we will have to apply an offset.
        // For instance, for batch_size = 4 and last_nz_idx = [0, 1, 1, 2], the actual
        // memory indices to load the scalar coefficients from are:
        // - 0 * 4 + 0 = 0
        // - 1 * 4 + 1 = 5
        // - 1 * 4 + 2 = 6
        // - 2 * 4 + 3 = 11.
        // That is, last_nz_idx * batch_size + offset, where offset is [0, 1, 2, 3].
        llvm::Value *offset = nullptr;
        if (batch_size == 1u) {
            // In scalar mode the offset is simply zero.
            offset = builder.getInt32(0);
        } else {
            offset = llvm::UndefValue::get(make_vector_type(builder.getInt32Ty(), batch_size));
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                offset = builder.CreateInsertElement(offset, builder.getInt32(i), i);
            }
        }

        // Init the vector of coefficient pointers with the base pointer value.
        auto *cf_ptr_v = vector_splat(builder, cf_ptr, batch_size);

        // Init the return value with zero.
        auto *retval = builder.CreateAlloca(last_nz_idx_t);
        builder.CreateStore(llvm::ConstantInt::get(last_nz_idx_t, 0u), retval);

        // The iteration range is [1, n].
        llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n + 1u), [&](llvm::Value *cur_n) {
            // Load the current poly coefficient(s).
            auto *cur_cf = ext_load_vector_from_memory(
                s, scal_t,
                builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.CreateMul(cur_n, builder.getInt32(batch_size))),
                batch_size);

            // Load the last nonzero coefficient(s).
            auto *last_nz_ptr_idx = builder.CreateAdd(
                offset, builder.CreateMul(builder.CreateLoad(last_nz_idx_t, last_nz_idx),
                                          vector_splat(builder, builder.getInt32(batch_size), batch_size)));
            auto *last_nz_ptr = builder.CreateInBoundsGEP(ext_fp_t, cf_ptr_v, last_nz_ptr_idx);
            auto *last_nz_cf = ext_gather_vector_from_memory(s, cur_cf->getType(), last_nz_ptr);

            // Compute the sign of the current coefficient(s).
            auto *cur_sgn = llvm_sgn(s, cur_cf);

            // Compute the sign of the last nonzero coefficient(s).
            auto *last_nz_sgn = llvm_sgn(s, last_nz_cf);

            // Add them and check if the result is zero (this indicates a sign change).
            auto *cmp = builder.CreateICmpEQ(builder.CreateAdd(cur_sgn, last_nz_sgn),
                                             llvm::ConstantInt::get(cur_sgn->getType(), 0u));

            // We also need to check if last_nz_sgn is zero. If that is the case, it means
            // we haven't found any nonzero coefficient yet for the polynomial and we must
            // not modify retval yet.
            auto *zero_cmp = builder.CreateICmpEQ(last_nz_sgn, llvm::ConstantInt::get(last_nz_sgn->getType(), 0u));
            cmp = builder.CreateSelect(zero_cmp, llvm::ConstantInt::get(cmp->getType(), 0u), cmp);

            // Update retval.
            builder.CreateStore(
                builder.CreateAdd(builder.CreateLoad(last_nz_idx_t, retval), builder.CreateZExt(cmp, last_nz_idx_t)),
                retval);

            // Update last_nz_idx.
            builder.CreateStore(
                builder.CreateSelect(builder.CreateICmpEQ(cur_sgn, llvm::ConstantInt::get(cur_sgn->getType(), 0u)),
                                     builder.CreateLoad(last_nz_idx_t, last_nz_idx),
                                     vector_splat(builder, cur_n, batch_size)),
                last_nz_idx);
        });

        // Store the result.
        store_vector_to_memory(builder, out_ptr, builder.CreateLoad(last_nz_idx_t, retval));

        // Return.
        builder.CreateRetVoid();

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Compute the enclosure of the polynomial of order n with coefficients stored in cf_ptr
// over the interval [h_lo, h_hi] using interval arithmetics. The polynomial coefficients
// are vectors of size batch_size and scalar type fp_t. cf_ptr is an external pointer.
// NOTE: the interval arithmetic implementation here is not 100% correct, because
// we do not account for floating-point truncation. In order to be mathematically
// correct, we would need to adjust the results of interval arithmetic add/mul via
// a std::nextafter()-like function. See here for an example:
// https://stackoverflow.com/questions/10420848/how-do-you-get-the-next-value-in-the-floating-point-sequence
// http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node46.html
// Perhaps another alternative would be to employ FP primitives with explicit rounding modes,
// which are available in LLVM. For mppp::real, we could employ the MPFR primitives
// with specific rounding modes.
std::pair<llvm::Value *, llvm::Value *> llvm_penc_interval(llvm_state &s, llvm::Type *fp_t, llvm::Value *cf_ptr,
                                                           std::uint32_t n, llvm::Value *h_lo, llvm::Value *h_hi,
                                                           std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(fp_t != nullptr);
    assert(batch_size > 0u);
    assert(cf_ptr != nullptr);
    assert(h_lo != nullptr);
    assert(h_hi != nullptr);
    assert(llvm::isa<llvm::PointerType>(cf_ptr->getType()));

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while implementing the computation of the enclosure of a "
                                  "polynomial via interval arithmetic");
    }
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Helper to implement the sum of two intervals.
    // NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
    auto ival_sum = [&s](llvm::Value *a_lo, llvm::Value *a_hi, llvm::Value *b_lo, llvm::Value *b_hi) {
        return std::make_pair(llvm_fadd(s, a_lo, b_lo), llvm_fadd(s, a_hi, b_hi));
    };

    // Helper to implement the product of two intervals.
    auto ival_prod = [&s](llvm::Value *a_lo, llvm::Value *a_hi, llvm::Value *b_lo, llvm::Value *b_hi) {
        auto *tmp1 = llvm_fmul(s, a_lo, b_lo);
        auto *tmp2 = llvm_fmul(s, a_lo, b_hi);
        auto *tmp3 = llvm_fmul(s, a_hi, b_lo);
        auto *tmp4 = llvm_fmul(s, a_hi, b_hi);

        // NOTE: here we are not correctly propagating NaNs,
        // for which we would need to use llvm_min/max_nan(),
        // which however incur in a noticeable performance
        // penalty. Thus, even in presence of all finite
        // Taylor coefficients and integration timestep, it could
        // conceivably happen that NaNs are generated in the
        // multiplications above and they are not correctly propagated
        // in these min/max functions, thus ultimately leading to an
        // incorrect result. This however looks like a very unlikely
        // occurrence.
        auto *cmp1 = llvm_min(s, tmp1, tmp2);
        auto *cmp2 = llvm_min(s, tmp3, tmp4);
        auto *cmp3 = llvm_max(s, tmp1, tmp2);
        auto *cmp4 = llvm_max(s, tmp3, tmp4);

        return std::make_pair(llvm_min(s, cmp1, cmp2), llvm_max(s, cmp3, cmp4));
    };

    // Fetch the vector type.
    auto *fp_vec_t = make_vector_type(fp_t, batch_size);

    // Create the lo/hi components of the accumulator.
    auto *acc_lo = builder.CreateAlloca(fp_vec_t);
    auto *acc_hi = builder.CreateAlloca(fp_vec_t);

    // Init the accumulator's lo/hi components with the highest-order coefficient.
    auto *ho_cf = ext_load_vector_from_memory(
        s, fp_t,
        builder.CreateInBoundsGEP(ext_fp_t, cf_ptr,
                                  builder.CreateMul(builder.getInt32(n), builder.getInt32(batch_size))),
        batch_size);
    builder.CreateStore(ho_cf, acc_lo);
    builder.CreateStore(ho_cf, acc_hi);

    // Run the Horner scheme (starting from 1 because we already consumed the
    // highest-order coefficient).
    llvm_loop_u32(s, builder.getInt32(1), builder.getInt32(n + 1u), [&](llvm::Value *i) {
        // Load the current coefficient.
        // NOTE: we are iterating backwards from the high-order coefficients
        // to the low-order ones.
        auto *ptr = builder.CreateInBoundsGEP(
            ext_fp_t, cf_ptr,
            builder.CreateMul(builder.CreateSub(builder.getInt32(n), i), builder.getInt32(batch_size)));
        auto *cur_cf = ext_load_vector_from_memory(s, fp_t, ptr, batch_size);

        // Multiply the accumulator by h.
        auto [acc_h_lo, acc_h_hi]
            = ival_prod(builder.CreateLoad(fp_vec_t, acc_lo), builder.CreateLoad(fp_vec_t, acc_hi), h_lo, h_hi);

        // Update the value of the accumulator.
        auto [new_acc_lo, new_acc_hi] = ival_sum(cur_cf, cur_cf, acc_h_lo, acc_h_hi);
        builder.CreateStore(new_acc_lo, acc_lo);
        builder.CreateStore(new_acc_hi, acc_hi);
    });

    // Return the lo/hi components of the accumulator.
    return {builder.CreateLoad(fp_vec_t, acc_lo), builder.CreateLoad(fp_vec_t, acc_hi)};
}

// Compute the enclosure of the polynomial of order n with coefficients stored in cf_ptr
// over an interval using the Cargo-Shisha algorithm. The polynomial coefficients
// are vectors of size batch_size and scalar type fp_t. The interval of the independent variable
// is [0, h] if h >= 0, [h, 0] otherwise. cf_ptr is an external pointer.
// NOTE: the Cargo-Shisha algorithm produces tighter bounds, but it has quadratic complexity
// and it seems to be less well-behaved numerically in corner cases. It might still be worth it up to double-precision
// computations, where the practical slowdown wrt interval arithmetics is smaller.
std::pair<llvm::Value *, llvm::Value *> llvm_penc_cargo_shisha(llvm_state &s, llvm::Type *fp_t, llvm::Value *cf_ptr,
                                                               std::uint32_t n, llvm::Value *h,
                                                               std::uint32_t batch_size)
{
    // LCOV_EXCL_START
    assert(fp_t != nullptr);
    assert(batch_size > 0u);
    assert(cf_ptr != nullptr);
    assert(h != nullptr);
    assert(llvm::isa<llvm::PointerType>(cf_ptr->getType()));

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while implementing the computation of the enclosure of a "
                                  "polynomial via the Cargo-Shisha algorithm");
    }
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // bj_series will contain the terms of the series
    // for the computation of bj. old_bj_series will be
    // used to deal with the fact that the pairwise sum
    // consumes the input vector.
    std::vector<llvm::Value *> bj_series, old_bj_series;

    // Init the current power of h with h itself.
    auto *cur_h_pow = h;

    // Compute the first value, b0, and add it to bj_series.
    auto *b0 = ext_load_vector_from_memory(s, fp_t, cf_ptr, batch_size);
    bj_series.push_back(b0);

    // Init min/max bj with b0.
    auto *min_bj = b0, *max_bj = b0;

    // Main iteration.
    for (std::uint32_t j = 1u; j <= n; ++j) {
        // Compute the new term of the series.
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.getInt32(j * batch_size));
        auto *cur_cf = ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
        auto *new_term = llvm_fmul(s, cur_cf, cur_h_pow);
        new_term = llvm_fdiv(s, new_term,
                             vector_splat(builder,
                                          llvm_codegen(s, fp_t,
                                                       binomial(number_like(s, fp_t, static_cast<double>(n)),
                                                                number_like(s, fp_t, static_cast<double>(j)))),
                                          batch_size));

        // Add it to bj_series.
        bj_series.push_back(new_term);

        // Update all elements of bj_series (apart from the last one).
        for (std::uint32_t i = 0; i < j; ++i) {
            bj_series[i] = llvm_fmul(s, bj_series[i],
                                     vector_splat(builder,
                                                  llvm_codegen(s, fp_t,
                                                               number_like(s, fp_t, static_cast<double>(j))
                                                                   / number_like(s, fp_t, static_cast<double>(j - i))),
                                                  batch_size));
        }

        // Compute the new bj.
        old_bj_series = bj_series;
        auto *cur_bj = pairwise_sum(s, bj_series);
        old_bj_series.swap(bj_series);

        // Update min/max_bj.
        min_bj = llvm_min(s, min_bj, cur_bj);
        max_bj = llvm_max(s, max_bj, cur_bj);

        // Update cur_h_pow, if we are not at the last iteration.
        if (j != n) {
            cur_h_pow = llvm_fmul(s, cur_h_pow, h);
        }
    }

    return {min_bj, max_bj};
}

// Helper to create a global const array containing
// all binomial coefficients up to (n, n). The coefficients are stored
// as scalars and the return value is a pointer to the first coefficient.
// The array has shape (n + 1, n + 1) and it is stored in row-major format.
llvm::Value *llvm_add_bc_array(llvm_state &s, llvm::Type *fp_t, std::uint32_t n)
{
    // Overflow check.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || (n + 1u) > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding an array of binomial coefficients");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();

    // Fetch the array type.
    auto *arr_type = llvm::ArrayType::get(fp_t, boost::numeric_cast<std::uint64_t>((n + 1u) * (n + 1u)));

    // Generate the binomials as constants.
    std::vector<llvm::Constant *> bc_const;
    for (std::uint32_t i = 0; i <= n; ++i) {
        for (std::uint32_t j = 0; j <= n; ++j) {
            // NOTE: the Boost implementation requires j <= i. We don't care about
            // j > i anyway.
            const auto val = (j <= i) ? binomial(number_like(s, fp_t, static_cast<double>(i)),
                                                 number_like(s, fp_t, static_cast<double>(j)))
                                      : number_like(s, fp_t, 0.);
            bc_const.push_back(llvm::cast<llvm::Constant>(llvm_codegen(s, fp_t, val)));
        }
    }

    // Create the global array.
    auto *bc_const_arr = llvm::ConstantArray::get(arr_type, bc_const);
    auto *g_bc_const_arr = new llvm::GlobalVariable(md, bc_const_arr->getType(), true,
                                                    llvm::GlobalVariable::PrivateLinkage, bc_const_arr);

    // Get out a pointer to the beginning of the array.
    return builder.CreateInBoundsGEP(bc_const_arr->getType(), g_bc_const_arr,
                                     {builder.getInt32(0), builder.getInt32(0)});
}

} // namespace detail

HEYOKA_END_NAMESPACE
