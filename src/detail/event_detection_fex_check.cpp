// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <boost/predef/architecture.h>

#include <fmt/core.h>

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/llvm_state.hpp>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Possible implementations for the fast event exclusion check function.
enum class fex_check_impl : std::uint8_t {
    // The portable implementation, based on the global floating-point rounding mode. Always available for all builtin
    // FP types (including vector variants).
    portable,
    // The AVX512-based implementation, using intrinsics with explicit rounding. Requires AVX512F, the fp type to be
    // float or double, and a batch size of either 1 or the size of a 512-bit vector (i.e., 8 for double, 16 for float).
    avx512,
    // The implementation for mppp::real, using MPFR primitives with explicit rounding.
    multiprecision,
};

// LCOV_EXCL_START

// Helper to compute 'a op b', rounded toward +inf, via the AVX512 intrinsics with static rounding. op must be one of
// "add", "sub" or "mul". a and b must be of the same type, which must be either float/double or a 512-bit vector of
// float/double (i.e., of size 16/8).
//
// NOTE: static rounding is available only for scalar operations and for 512-bit vectors. The scalar intrinsics operate
// on the first element of 128-bit vectors, thus scalar operands are inserted into (and the result is extracted from)
// the first element of 128-bit vectors.
llvm::Value *avx512_binop_up(llvm_state &s, const std::string_view op, llvm::Value *const a, llvm::Value *const b)
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    assert(op == "add" || op == "sub" || op == "mul");

    auto &builder = s.builder();

    auto *const x_t = a->getType();
    auto *const scal_t = x_t->getScalarType();
    assert(scal_t->isFloatTy() || scal_t->isDoubleTy());
    const auto is_double = scal_t->isDoubleTy();

    // The rounding argument: rounding toward +inf (_MM_FROUND_TO_POS_INF == 2) with suppression of all floating-point
    // exceptions (_MM_FROUND_NO_EXC == 8).
    auto *const rnd = builder.getInt32(10);

    if (llvm::isa<llvm::FixedVectorType>(x_t)) {
        // 512-bit vectors: use the packed intrinsics.
        assert(llvm::cast<llvm::FixedVectorType>(x_t)->getNumElements() == (is_double ? 8u : 16u));

        return llvm_invoke_intrinsic(builder, fmt::format("llvm.x86.avx512.{}.{}.512", op, is_double ? "pd" : "ps"), {},
                                     {a, b, rnd});
    }

    // Scalars: use the scalar intrinsics on the first element of 128-bit vectors.
    //
    // NOTE: the scalar intrinsics compute the first element of the result only from the first elements of the operands,
    // the other elements of the result being copied from the first operand. Since we use only the first element of the
    // result, we can leave the other elements of the operands as poison. This allows the backend to operate directly on
    // the registers holding the scalar values, without zeroing out the other elements.
    auto *const vec_t = make_vector_type(scal_t, is_double ? 2u : 4u);
    auto *const poison_vec = llvm::PoisonValue::get(vec_t);
    auto *const a_vec = builder.CreateInsertElement(poison_vec, a, static_cast<std::uint64_t>(0));
    auto *const b_vec = builder.CreateInsertElement(poison_vec, b, static_cast<std::uint64_t>(0));

    // NOTE: the last three arguments of the scalar intrinsics are the passthrough vector, the mask and the rounding
    // argument. We set all bits in the mask, so that the passthrough vector is never used.
    auto *const res_vec
        = llvm_invoke_intrinsic(builder, fmt::format("llvm.x86.avx512.mask.{}.{}.round", op, is_double ? "sd" : "ss"),
                                {}, {a_vec, b_vec, poison_vec, builder.getInt8(0xff), rnd});

    return builder.CreateExtractElement(res_vec, static_cast<std::uint64_t>(0));
}

// Wrapper to set the rounding mode to UP for a scalar floating-point type fp_t.
//
// Normally, we use llvm.get/set.rounding() to manipulate the rounding mode, but on some platforms this leads to
// suboptimal code that can be improved with a custom implementation.
//
// The return value is the original rounding mode (i.e., before the change), a platform-specific 32-bit int value that
// can be passed to llvm_restore_rounding() to restore the original rounding mode.
llvm::Value *llvm_set_rounding_up(llvm_state &s, [[maybe_unused]] const llvm::Type *const fp_t)
{
    assert(!fp_t->isVectorTy());

    auto &builder = s.builder();

    // NOTE: on x86-64 we assume that all float/double operations use the SSE-family instructions, rather than the
    // x87 instructions. Under this assumption, we can ignore the x87 rounding mode (which would be read/set by
    // llvm.get/set.rounding()) and operate only on the MXCSR rounding mode.
    //
    // NOTE: this is a compile-time check, which happens to work correctly because currently we are always jitting for
    // the host. If we ever enable jitting for non-host architectures, we should re-write these checks in terms of
    // runtime-derived properties (e.g., via the target triple).
#if BOOST_ARCH_X86_64

    if (fp_t->isDoubleTy() || fp_t->isFloatTy()) {
        // Mask of the rounding control field (bits 13 and 14) in the MXCSR register.
        constexpr std::uint32_t mxcsr_rc_mask = 0x6000;

        // Mask of all the bits of the MXCSR register except the rounding control field.
        constexpr std::uint32_t mxcsr_non_rc_mask = 0xFFFF9FFF;

        // Value of the rounding control field for rounding up.
        constexpr std::uint32_t mxcsr_rc_up = 0x4000;

        // Load the current value of the MXCSR register via a stack slot.
        auto *const mxcsr_ptr = builder.CreateAlloca(builder.getInt32Ty());
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.stmxcsr", {}, {mxcsr_ptr});
        auto *const mxcsr = builder.CreateLoad(builder.getInt32Ty(), mxcsr_ptr);

        // Replace the rounding control field with the value for rounding up, leaving the other bits unchanged, and
        // write the result back into the MXCSR register via the same stack slot.
        builder.CreateStore(builder.CreateOr(builder.CreateAnd(mxcsr, mxcsr_non_rc_mask), mxcsr_rc_up), mxcsr_ptr);
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.ldmxcsr", {}, {mxcsr_ptr});

        // Return the original rounding control field.
        return builder.CreateAnd(mxcsr, mxcsr_rc_mask);
    }

#endif

    // Fetch the current rounding mode.
    auto *const ret = llvm_invoke_intrinsic(builder, "llvm.get.rounding", {}, {});

    // Switch to rounding toward +inf.
    llvm_invoke_intrinsic(builder, "llvm.set.rounding", {},
                          {builder.getInt32(static_cast<std::uint32_t>(llvm::RoundingMode::TowardPositive))});

    return ret;
}

// LCOV_EXCL_STOP

// Wrapper to restore the original rounding mode returned by llvm_set_rounding_up() for a scalar floating-point type
// fp_t.
void llvm_restore_rounding(llvm_state &s, llvm::Value *const rounding, [[maybe_unused]] const llvm::Type *const fp_t)
{
    assert(!fp_t->isVectorTy());

    auto &builder = s.builder();

    assert(rounding->getType() == builder.getInt32Ty());

    // NOTE: see the explanation in llvm_set_rounding_up().
#if BOOST_ARCH_X86_64

    if (fp_t->isDoubleTy() || fp_t->isFloatTy()) {
        // Mask of all the bits of the MXCSR register except the rounding control field (bits 13 and 14).
        constexpr std::uint32_t mxcsr_non_rc_mask = 0xFFFF9FFF;

        // Load the current value of the MXCSR register via a stack slot.
        auto *const mxcsr_ptr = builder.CreateAlloca(builder.getInt32Ty());
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.stmxcsr", {}, {mxcsr_ptr});
        auto *const mxcsr = builder.CreateLoad(builder.getInt32Ty(), mxcsr_ptr);

        // Replace the rounding control field with the original one, leaving the other bits unchanged, and write the
        // result back into the MXCSR register via the same stack slot.
        builder.CreateStore(builder.CreateOr(builder.CreateAnd(mxcsr, mxcsr_non_rc_mask), rounding), mxcsr_ptr);
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.ldmxcsr", {}, {mxcsr_ptr});

        return;
    }

#endif

    // Restore the original rounding mode.
    llvm_invoke_intrinsic(builder, "llvm.set.rounding", {}, {rounding});
}

} // namespace

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

    // Establish the implementation to use.
    auto impl = fex_check_impl::portable;

    // Check if we can use avx512.
    const auto &target_features = get_target_features();
    if (target_features.avx512f) {
        // LCOV_EXCL_START
        const auto fp_is_float = fp_t->isFloatTy();
        const auto fp_is_double = fp_t->isDoubleTy();

        // We can use the AVX512 implementation for float/double if the batch size is either 1 or the size of a 512-bit
        // vector.
        if ((fp_is_float && (batch_size == 1u || batch_size == 16u))
            || (fp_is_double && (batch_size == 1u || batch_size == 8u))) {
            impl = fex_check_impl::avx512;
        }
        // LCOV_EXCL_STOP
    }

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(fp_t) != 0) {
        impl = fex_check_impl::multiprecision;
    }

#endif

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

    // NOTE: the portable implementation requires the strictfp flag to be present on the function definition.
    if (impl == fex_check_impl::portable) {
        f->addFnAttr(llvm::Attribute::StrictFP);
    }

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

    // Value storing the original FP rounding mode. It will be set and used only in the portable codepath.
    llvm::Value *orig_rounding = nullptr;

    // Temporarily disable the fast math flags and the fpmath metadata. We want to ensure that the generated code is a
    // faithful transcription of the algorithm, without any reassociation, contraction, reduced accuracy, etc.
    const llvm::IRBuilderBase::FastMathFlagGuard fmfg(builder);
    builder.setFastMathFlags(llvm::FastMathFlags{});
    builder.setDefaultFPMathTag(nullptr);

    if (impl == fex_check_impl::portable) {
        // NOTE: in the portable implementation, we must make sure that:
        //
        // - the constrained FP intrinsics are being used,
        // - the rounding mode is declared dynamic, i.e., LLVM must assume that it is set at runtime (which we do
        //   explicitly in a separate step below) and must not perform any transformation depending on it,
        // - floating-point exceptions are ignored.
        //
        // All these settings will be reset to their original values when fmfg is destroyed.
        builder.setIsFPConstrained(true);
        builder.setDefaultConstrainedRounding(llvm::RoundingMode::Dynamic);
        builder.setDefaultConstrainedExcept(llvm::fp::ebIgnore);

        // Set the rounding to +inf and save the original rounding mode.
        orig_rounding = llvm_set_rounding_up(s, fp_t);
    }

    // Load the timestep.
    auto *const h = ext_load_vector_from_memory(s, fp_t, h_ptr, batch_size);

    // The zero constant.
    auto *const zero = llvm_constantfp(s, val_t, 0.);

    // Helper to compute a < b.
    //
    // NOTE: LLVM cannot lower constrained comparisons of vectors of fp128 on some platforms (e.g., x86-64 and aarch64
    // as of LLVM 23), nor constrained comparisons of vectors of x86_fp80 on x86-64 with AVX-512 (LLVM 19-21). Thus, for
    // these types we compare element by element. This has no runtime cost, as the backend operates on these vectors
    // element by element anyway.
    const auto lt = [&s, &builder, fp_t](llvm::Value *const a, llvm::Value *const b) -> llvm::Value * {
        if (fp_t->isFP128Ty() || fp_t->isX86_FP80Ty()) {
            const auto a_scalars = vector_to_scalars(builder, a);
            const auto b_scalars = vector_to_scalars(builder, b);

            std::vector<llvm::Value *> res;
            res.reserve(a_scalars.size());
            for (decltype(a_scalars.size()) i = 0; i < a_scalars.size(); ++i) {
                res.push_back(llvm_fcmp_olt(s, a_scalars[i], b_scalars[i]));
            }

            return scalars_to_vector(builder, res);
        }

        return llvm_fcmp_olt(s, a, b);
    };

    // Helper to compute max(x, 0).
    //
    // NOTE: for float and double, we use the maximumnum LLVM intrinsic, which maps to hardware max instructions. With a
    // comparison + select, the scalar code of the portable implementation would instead contain many branches.
    // maximumnum is equivalent to the comparison + select when NaNs are not involved (which is guaranteed under the
    // assumption that the function is being invoked only with finite inputs, see below). For the other types (x87 long
    // double, fp128, mppp::real) we use the comparison + select, which for x87 and fp128 results in cheaper code than
    // maximumnum.
    //
    // The only potential issue with maximumnum is that apparently it currently does not have a constrained counterpart,
    // and this seemingly breaks the LLVM rule that "if any FP operation in a function is constrained then they all must
    // be constrained". There are exceptions to this rule (e.g., abs()) but it is not clear if maximumnum also falls
    // within the exception.
    //
    // In any case, in case of issues, we can always revert to using the comparison + select.
    //
    // NOTE: maximumnum is available since LLVM 20, but LLVM 20 and 21 miscompile it on x86-64. With earlier versions,
    // we always use the comparison + select.
    const auto max0 = [zero, &builder, fp_t, &lt](llvm::Value *const x) -> llvm::Value * {
#if LLVM_VERSION_MAJOR >= 22
        if (fp_t->isFloatTy() || fp_t->isDoubleTy()) {
            return builder.CreateMaximumNum(x, zero);
        }
#else
        // Avoid warning for unused lambda capture.
        static_cast<void>(fp_t);
#endif
        return builder.CreateSelect(lt(x, zero), zero, x);
    };

    // Determine the back flag. True if h < 0, false otherwise.
    auto *const back_flag = lt(h, zero);

    // We now define the basic operations we will be using in the interval Horner implementation.
    //
    // We will be using only rounding up mode throughout the code, taking advantage of the fact that rounding down can
    // be implemented by negating the rounding up of the negation: down(x) = -up(-x) (see also the explanation below).

    // Common helper to construct a rounding-up primitive.
    //
    // 'name' is the name of the operation ("add", "sub" or "mul"), used to select the AVX512 intrinsic and the MPFR
    // primitive. portable_op is the function implementing the operation in the portable implementation.
    const auto make_op_up = [impl, &builder, &s, fp_t](const char *name, auto *const portable_op) {
        return
            [impl, &builder, &s, fp_t, name, portable_op](llvm::Value *const a, llvm::Value *const b) -> llvm::Value * {
                // Avoid warning for unused lambda captures when mppp::real is not available.
                static_cast<void>(builder);
                static_cast<void>(fp_t);

                switch (impl) {
                    // LCOV_EXCL_START
                    case fex_check_impl::avx512:
                        return avx512_binop_up(s, name, a, b);
                    // LCOV_EXCL_STOP
#if defined(HEYOKA_HAVE_REAL)
                    case fex_check_impl::multiprecision:
                        // NOTE: explicit rounding mode passed to the MPFR primitive.
                        return builder.CreateCall(real_nary_op(s, fp_t, fmt::format("mpfr_{}", name), 2, MPFR_RNDU),
                                                  {a, b});
#endif
                    default:
                        assert(impl == fex_check_impl::portable);
                        // NOTE: in portable mode the rounding mode is set to up already, and the builder setup will
                        // ensure that a constrained intrinsic is emitted.
                        return portable_op(s, a, b);
                }
            };
    };

    // Addition with rounding-up.
    const auto add_up = make_op_up("add", llvm_fadd);

    // Subtraction with rounding-up.
    const auto sub_up = make_op_up("sub", llvm_fsub);

    // Multiplication with rounding-up.
    const auto mul_up = make_op_up("mul", llvm_fmul);

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
    // it.
    //
    // For the multiplication part, the standard interval arithmetic formula is (for forward integration):
    //
    // [lo, hi] · [0, h] = [min(lo·0, lo·h, hi·0, hi·h), max(lo·0, lo·h, hi·0, hi·h)],
    //
    // which simplifies to [min(lo·h, 0), max(hi·h, 0)], because h>=0 and lo<=hi. Note that the simplification relies on
    // hi·0 and lo·0 going to zero, which looks a bit iffy if hi/lo are infinite. However, as explained below,
    // infinities in the accumulation mean that the values have become too large and they are not indicative of
    // poles/singularities, so multiplying these large values by zero can still be regarded as producing zero as a
    // result and the simplification still holds.
    //
    // In FP arithmetic, hi·h must be rounded up and lo·h rounded down in order to obtain a certified result. Instead of
    // rounding lo·h down, we can round -lo·h up and then negate the result. Putting it all together, the result of the
    // multiplication is [-max(-lo·h, 0), max(hi·h, 0)], where now all multiplications are rounded up.
    //
    // For the addition part, the result is [c - max(-lo·h, 0), c + max(hi·h, 0)], where, again, the subtraction must be
    // rounded down and the addition rounded up. Like before, the lower bound can be computed with a rounded-up
    // subtraction with swapped operands, followed by a negation, resulting in [-(max(-lo·h, 0) - c), c + max(hi·h, 0)],
    // where now both addition and subtraction are rounded up.
    //
    // For backward integration, we have that h<0 and the evaluation interval is [h, 0]. For any x<=0, the evaluation of
    // a polynomial p in x, p(x), is equivalent to p'(|x|), where p' is p with negated odd-degree coefficients. It then
    // follows that for every number x in the [h, 0] range, p(x)=p'(|x|) and we can thus replace the evaluation of p in
    // the [h, 0] interval with the evaluation of p' in the [0, |h|] interval. In other words, we can handle backward
    // integration with the forward integration formula, provided that h is replaced with |h| and we take care of
    // flipping around the sign of the odd-degree coefficients during the evaluation. The formula for an accumulation
    // step then becomes:
    //
    // [-(max(-lo·|h|, 0) - c), c + max(hi·|h|, 0)]
    //
    // Finally, we note how if we carry around the accumulator not as the [lo, hi] interval, but as the (mlo, hi) pair
    // instead (where mlo = -lo), the formula for an accumulation step becomes:
    //
    // (max(mlo·|h|, 0) - c, c + max(hi·|h|, 0)),
    //
    // which allows us to remove the double negations in the computation of the lo part of the interval.
    //
    // NOTE: throughout this derivation we have assumed that absolute values, negations and min/max computations are
    // always exact and unaffected by the rounding mode.
    //
    // NOTE: assuming that all input values are finite, non-finite values for the enclosure may still be generated by
    // this algorithm if h!=0 (for h==0, the enclosure will trivially produce the [c0, c0] interval, where c0 is the
    // constant coefficient of the polynomial). Specifically, the multiplications -lo·|h| and hi·|h| may result in +inf
    // (but never in -inf due to the upward rounding). If +inf is generated, then we will be in a situation where a
    // finite value (c or its negation) is added to/subtracted from +inf which always results (correctly) in +inf.
    // Similarly, the addition/subtraction may also generate a +inf (but again, never a -inf) from finite input values.
    // In any case, when one of the bounds flips to infinity, it stays there throughout the accumulation ultimately
    // producing a mathematically-correct (albeit perhaps not terribly useful) result.

    // Compute |h|.
    auto *const abs_h = llvm_abs(s, h);

    // Init the accumulator (mlo, hi) with the highest-order coefficient.
    auto *hi = load_cf(n);
    // If the poly degree is odd, negate the coefficient for backward integration.
    if (n % 2u != 0u) {
        hi = builder.CreateSelect(back_flag, llvm_fneg(s, hi), hi);
    }
    auto *mlo = llvm_fneg(s, hi);

    for (auto i = n; i > 0u; --i) {
        // Load the current coefficient.
        //
        // NOTE: we are iterating backwards from the high-order coefficients to the low-order ones.
        const auto cur_degree = i - 1u;
        auto *cf = load_cf(cur_degree);

        // Negate odd-degree coefficients for backward integration.
        if (cur_degree % 2u != 0u) {
            cf = builder.CreateSelect(back_flag, llvm_fneg(s, cf), cf);
        }

        // Compute the new components.
        mlo = sub_up(max0(mul_up(mlo, abs_h)), cf);
        hi = add_up(max0(mul_up(hi, abs_h)), cf);
    }

    // The enclosure does not contain zero if lo > 0 (i.e., mlo < 0) or hi < 0.
    auto *const excl = builder.CreateOr(lt(mlo, zero), lt(hi, zero));

    // Store the result, as std::uint32_t.
    ext_store_vector_to_memory(s, out_ptr,
                               builder.CreateZExt(excl, make_vector_type(builder.getInt32Ty(), batch_size)));

    // Store the enclosure.
    ext_store_vector_to_memory(s, enc_ptr, llvm_fneg(s, mlo));
    auto *const enc_hi_ptr = builder.CreateInBoundsGEP(ext_fp_t, enc_ptr, builder.getInt32(batch_size));
    ext_store_vector_to_memory(s, enc_hi_ptr, hi);

    // Restore the original rounding mode, if necessary.
    if (impl == fex_check_impl::portable) {
        assert(orig_rounding != nullptr);
        llvm_restore_rounding(s, orig_rounding, fp_t);
    }

    // Exit.
    builder.CreateRetVoid();

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE
