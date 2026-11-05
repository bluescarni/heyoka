// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to add a fast event exclusion check function to an llvm_state.
//
// The function, to be invoked from C++, will compute a rigorous enclosure of the event function's Taylor polynomial
// over the [0, h] range (resp., [h, 0] when integrating backwards in time, i.e., when h < 0) using interval arithmetic.
// The signature of the function is:
//
// void fname(const ext_fp_t *cf_ptr, const ext_fp_t *h_ptr, std::uint32_t *out_ptr, ext_fp_t *enc_ptr)
//
// fname is the name of the LLVM function that will be added. fp_t is the scalar floating-point LLVM type used in the
// computation, and ext_fp_t is the corresponding C++ type. The coefficients of the Taylor polynomial of an event
// function are stored in cf_ptr in a 2D array of shape (n + 1, batch_size). n is the degree of the Taylor polynomial.
// batch_size is the SIMD vector size. h_ptr points to an array of size batch_size containing the value(s) of h. out_ptr
// points to an output array of size batch_size, containing 1 if the enclosure does not contain zero (i.e., the
// polynomial certainly has no roots in the range), 0 otherwise. enc_ptr points to an output 2D array of shape (2,
// batch_size) into which the [lo, hi] enclosure will be written (row 0 holds lo and row 1 holds hi). No overlap is
// allowed between the arrays.
//
// If a global value with the same name exists already in the module, an error will be raised.
//
// NOTE: although integration steps nominally consist of half-open time ranges, here we will be checking for roots in
// closed intervals in order to be consistent with the usual formulation of interval arithmetic. This does not affect
// the correctness of the function, and it will at most slightly degrade the quality of its output.
//
// NOTE: this implementation of the exclusion check assumes that all input values (i.e., the polynomial coefficients and
// the timestep h) are finite. Invoking it with non-finite values may produce incorrect results. In batch mode, this
// applies to each lane separately: non-finite values in a lane may produce incorrect results only in that lane.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Function *llvm_ed_add_fex_check(llvm_state &s, const std::string &fname, llvm::Type *const fp_t,
                                      const std::uint32_t n, const std::uint32_t batch_size)
{
    assert(fp_t != nullptr);
    assert(!fp_t->isVectorTy());
    assert(batch_size > 0u);

    // Overflow check: we need to be able to index into cf_ptr via std::uint32_t.
    try {
        using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
        static_cast<void>(static_cast<std::uint32_t>((safe_uint32_t(n) + 1u) * batch_size));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in llvm_ed_add_fex_check()");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point value type.
    auto *const val_t = make_vector_type(fp_t, batch_size);

    // Remember and restore the current insertion point.
    const llvm::IRBuilderBase::InsertPointGuard ipg(builder);

    // The function arguments:
    //
    // - pointer to the array of poly coefficients (read-only),
    // - pointer to the timestep value(s) (read-only),
    // - output pointer (write-only),
    // - enclosure pointer (write-only).
    //
    // No overlap is allowed. All floating-point pointers are to external types.
    const std::vector<llvm::Type *> fargs(4, llvm::PointerType::getUnqual(context));
    // The function does not return anything.
    auto *const ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *const f = llvm_func_create(ft, llvm::Function::ExternalLinkage, fname, &md);

    // Set the names/attributes of the function arguments.
    auto *const cf_ptr = f->args().begin();
    cf_ptr->setName("cf_ptr");
    llvm_add_no_capture_argattr(s, cf_ptr);
    cf_ptr->addAttr(llvm::Attribute::NoAlias);
    cf_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const h_ptr = f->args().begin() + 1;
    h_ptr->setName("h_ptr");
    llvm_add_no_capture_argattr(s, h_ptr);
    h_ptr->addAttr(llvm::Attribute::NoAlias);
    h_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const out_ptr = f->args().begin() + 2;
    out_ptr->setName("out_ptr");
    llvm_add_no_capture_argattr(s, out_ptr);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *const enc_ptr = f->args().begin() + 3;
    enc_ptr->setName("enc_ptr");
    llvm_add_no_capture_argattr(s, enc_ptr);
    enc_ptr->addAttr(llvm::Attribute::NoAlias);
    enc_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *const bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr);
    builder.SetInsertPoint(bb);

    // Setup the interval arithmetic guard.
    const auto ivg = llvm_ival_guard(s, f, val_t);

    // Load the timestep.
    auto *const h = ext_load_vector_from_memory(s, fp_t, h_ptr, batch_size);

    // The zero constant.
    auto *const zero = llvm_constantfp(s, val_t, 0.);

    // Determine the back flag. True if h < 0, false otherwise.
    auto *const back_flag = ivg.val_lt(h, zero);

    // Helper to load the coefficient of order i of the Taylor polynomial.
    //
    // NOTE: i * batch_size cannot overflow, as checked at the beginning of the function.
    auto *const ext_fp_t = make_external_llvm_type(fp_t);
    const auto load_cf = [&](const std::uint32_t i) {
        auto *const ptr = builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.getInt32(i * batch_size));
        return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
    };

    // Run the Horner scheme.
    //
    // At each iteration, the accumulator is multiplied by the time range and then the current coefficient c is added to
    // it. In order to accommodate backward integrations, for which h<0, we run the evaluation with the time range [0,
    // |h|], making sure to negate the odd-degree coefficients of the input polynomial. This produces a result
    // equivalent, for h<0, to evaluation with the time range [h, 0].
    //
    // NOTE: the input values are required to be finite. Non-finite values in the accumulator can still arise for h!=0
    // (for h==0, the enclosure is trivially the [c0, c0] interval, where c0 is the constant coefficient of the
    // polynomial): the multiplications and additions in the accumulation are rounded up, and they can thus overflow to
    // +inf, but never to -inf. Hence, mlo and hi may become +inf, but never -inf. Once a component of the accumulator
    // is +inf, the only operations it goes through are the multiplication by |h| (which is finite and nonzero, because
    // an overflow requires h!=0), the max with zero and the addition/subtraction of finite coefficients, all of which
    // leave +inf unchanged. Thus, the accumulation never produces NaN. Moreover, because mlo and hi are never -inf, the
    // lower bound -mlo is never +inf and the upper bound hi is never -inf, and the enclosure can never be in the
    // invalid forms [-inf, -inf] and [+inf, +inf].

    // Compute |h|.
    auto *const abs_h = ivg.val_abs(h);

    // Init the accumulator (mlo, hi) with the highest-order coefficient.
    auto *cf_n = load_cf(n);
    // If the poly degree is odd, negate the coefficient for backward integration.
    if (n % 2u != 0u) {
        cf_n = builder.CreateSelect(back_flag, ivg.val_neg(cf_n), cf_n);
    }
    auto *acc = ivg.ival_construct(ivg.val_neg(cf_n), cf_n);

    for (auto i = n; i > 0u; --i) {
        // Load the current coefficient.
        //
        // NOTE: we are iterating backwards from the high-order coefficients to the low-order ones.
        const auto cur_degree = i - 1u;
        auto *cf = load_cf(cur_degree);

        // Negate odd-degree coefficients for backward integration.
        if (cur_degree % 2u != 0u) {
            cf = builder.CreateSelect(back_flag, ivg.val_neg(cf), cf);
        }

        // Update the accumulator.
        auto *const tmp = ivg.ival_mul_zlb(acc, abs_h);
        acc = ivg.ival_add_val(tmp, cf);
    }

    // Check if the enclosure does not contain zero.
    auto *const excl = ivg.ival_nz(acc);

    // Store the result, as std::uint32_t.
    ext_store_vector_to_memory(s, out_ptr,
                               builder.CreateZExt(excl, make_vector_type(builder.getInt32Ty(), batch_size)));

    // Store the enclosure.
    const auto [acc_mlo, acc_hi] = ivg.ival_extract(acc);
    ext_store_vector_to_memory(s, enc_ptr, ivg.val_neg(acc_mlo));
    auto *const enc_hi_ptr = builder.CreateInBoundsGEP(ext_fp_t, enc_ptr, builder.getInt32(batch_size));
    ext_store_vector_to_memory(s, enc_hi_ptr, acc_hi);

    // Finalise the guard before emitting the return statement.
    ivg.finalise();

    // Exit.
    builder.CreateRetVoid();

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE
