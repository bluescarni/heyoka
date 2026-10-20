// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <fmt/core.h>

#include <boost/predef/architecture.h>

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
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

#if BOOST_ARCH_X86_64

// Mask of all the bits of the x86 MXCSR register except the rounding control field (bits 13 and 14).
constexpr std::uint32_t mxcsr_non_rc_mask = 0xFFFF9FFF;

#endif

// Wrapper to set the rounding mode to UP for a builtin scalar/vector floating-point type val_t.
//
// Normally, we use llvm.get/set.rounding() to manipulate the rounding mode, but on some platforms this leads to
// suboptimal code that can be improved with a custom implementation.
//
// The return value is the original rounding mode (i.e., before the change), a platform-specific 32-bit int value that
// can be passed to llvm_restore_rounding() to restore the original rounding mode.
llvm::Value *llvm_set_rounding_up(llvm_state &s, [[maybe_unused]] const llvm::Type *const val_t)
{
    assert(val_t != nullptr);
    assert(val_t->getScalarType()->isFloatingPointTy());

    auto &builder = s.builder();

    // NOTE: on x86-64 we assume that all float/double operations use the SSE-family instructions, rather than the x87
    // instructions. Under this assumption, we can ignore the x87 rounding mode (which would be read/set by
    // llvm.get/set.rounding()) and operate only on the MXCSR rounding mode.
    //
    // NOTE: this is a compile-time check, which happens to work correctly because currently we are always jitting for
    // the host. If we ever enable jitting for non-host architectures, we should re-write these checks in terms of
    // runtime-derived properties (e.g., via the target triple).
#if BOOST_ARCH_X86_64

    // Fetch the scalar floating-point type.
    const auto *const fp_t = val_t->getScalarType();

    if (fp_t->isDoubleTy() || fp_t->isFloatTy()) {
        // Mask of the rounding control field (bits 13 and 14) in the MXCSR register.
        constexpr std::uint32_t mxcsr_rc_mask = 0x6000;

        // Value of the rounding control field for rounding up.
        constexpr std::uint32_t mxcsr_rc_up = 0x4000;

        // Load the current value of the MXCSR register via a stack slot.
        auto *const int32_tp = builder.getInt32Ty();
        auto *const mxcsr_ptr = llvm_entry_alloca(s, int32_tp);
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.stmxcsr", {}, {mxcsr_ptr});
        auto *const mxcsr = builder.CreateLoad(int32_tp, mxcsr_ptr);

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

// Wrapper to restore the original rounding mode returned by llvm_set_rounding_up() for a builtin scalar/vector
// floating-point type val_t.
void llvm_restore_rounding(llvm_state &s, llvm::Value *const rounding, [[maybe_unused]] const llvm::Type *const val_t)
{
    assert(val_t != nullptr);
    assert(val_t->getScalarType()->isFloatingPointTy());

    auto &builder = s.builder();

    assert(rounding != nullptr);
    assert(rounding->getType() == builder.getInt32Ty());

    // NOTE: see the explanation in llvm_set_rounding_up().
#if BOOST_ARCH_X86_64

    // Fetch the scalar floating-point type.
    const auto *const fp_t = val_t->getScalarType();

    if (fp_t->isDoubleTy() || fp_t->isFloatTy()) {
        // Load the current value of the MXCSR register via a stack slot.
        auto *const int32_tp = builder.getInt32Ty();
        auto *const mxcsr_ptr = llvm_entry_alloca(s, int32_tp);
        llvm_invoke_intrinsic(builder, "llvm.x86.sse.stmxcsr", {}, {mxcsr_ptr});
        auto *const mxcsr = builder.CreateLoad(int32_tp, mxcsr_ptr);

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

// Enum listing the interval arithmetic implementation strategies.
enum class ival_mode : std::uint8_t {
    // The portable implementation, based on the global floating-point rounding mode. Always available for all builtin
    // FP types (including vector variants).
    portable,
    // The AVX512-based implementation, using intrinsics with explicit rounding. Requires AVX512F, the fp type to be
    // float or double, and a batch size of either 1 or the size of a 512-bit vector (i.e., 8 for double, 16 for
    // float).
    avx512,
    // The implementation for mppp::real, using MPFR primitives with explicit rounding.
    multiprecision,
};

// Helper to determine the interval arithmetic implementation strategy to be used for an input scalar or vector
// floating-point type val_t.
ival_mode llvm_get_ival_mode(const llvm::Type *const val_t)
{
    assert(val_t != nullptr);

    // Fetch the scalar floating-point type.
    const auto *const fp_t = val_t->getScalarType();

    // Fetch the batch size.
    const std::uint32_t batch_size = get_vector_size(val_t);

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
            return ival_mode::avx512;
        }
        // LCOV_EXCL_STOP
    }

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(fp_t) != 0) {
        return ival_mode::multiprecision;
    }

#endif

    // NOTE: at this point we are sure that fp_t must be a builtin fp type.
    assert(fp_t->isFloatingPointTy());

    return ival_mode::portable;
}

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

// LCOV_EXCL_STOP

} // namespace

// Implementation of llvm_ival_guard.
struct llvm_ival_guard::impl {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    llvm_state &m_state;
    llvm::IRBuilderBase::FastMathFlagGuard m_fmfg;
    llvm::Type *m_val_t = nullptr;
    ival_mode m_mode = ival_mode::portable;
    llvm::Value *m_orig_rounding = nullptr;

    // The ctor sets up a guard for the function f, assuming that interval arithmetic will operate on the scalar/vector
    // type val_t.
    impl(llvm_state &s, llvm::Function *const f, llvm::Type *const val_t)
        : m_state(s), m_fmfg(m_state.builder()), m_val_t(val_t), m_mode(llvm_get_ival_mode(m_val_t))
    {
        assert(f != nullptr);
        assert(m_val_t != nullptr);

        auto &builder = s.builder();

        // Disable the fast math flags and the fpmath metadata. For an interval arithmetic algorithm, we want to ensure
        // that the generated code is a faithful transcription of the original algorithm, without any reassociation,
        // contraction, reduced accuracy, etc.
        builder.setFastMathFlags(llvm::FastMathFlags{});
        builder.setDefaultFPMathTag(nullptr);

        if (m_mode == ival_mode::portable) {
            // NOTE: in the portable implementation, we must make sure that:
            //
            // - the strictfp flag is added to the function definition,
            // - the constrained FP intrinsics are being used,
            // - the rounding mode is declared dynamic, i.e., LLVM must assume that it is set at runtime (which we do
            //   explicitly in a separate step below) and must not perform any transformation depending on it,
            // - floating-point exceptions are ignored.
            //
            // The builder settings will be reset to their original values when m_fmfg is destroyed.
            f->addFnAttr(llvm::Attribute::StrictFP);
            builder.setIsFPConstrained(true);
            builder.setDefaultConstrainedRounding(llvm::RoundingMode::Dynamic);
            builder.setDefaultConstrainedExcept(llvm::fp::ebIgnore);

            // Set the rounding to +inf and save the original rounding mode.
            m_orig_rounding = llvm_set_rounding_up(s, val_t);
        }
    }

    // Common helper to compute 'a name b' rounded toward +inf.
    //
    // 'name' is the name of the operation ("add", "sub" or "mul"), used to select the AVX512 intrinsic and the MPFR
    // primitive. portable_op is the function implementing the operation in the portable implementation.
    llvm::Value *arith_op_up(const char *name, auto *const portable_op, llvm::Value *const a,
                             llvm::Value *const b) const
    {
        assert(a != nullptr);
        assert(b != nullptr);
        assert(a->getType() == b->getType());
        assert(a->getType() == m_val_t);

        switch (m_mode) {
            // LCOV_EXCL_START
            case ival_mode::avx512:
                return avx512_binop_up(m_state, name, a, b);
                // LCOV_EXCL_STOP
#if defined(HEYOKA_HAVE_REAL)
            case ival_mode::multiprecision:
                // NOTE: explicit rounding mode passed to the MPFR primitive.
                //
                // NOTE: real_nary_op() may generate a wrapper function here. This is allowed while the guard is
                // alive, see the class documentation in llvm_helpers.hpp.
                assert(!m_val_t->isVectorTy());
                return m_state.builder().CreateCall(
                    real_nary_op(m_state, m_val_t, fmt::format("mpfr_{}", name), 2, MPFR_RNDU), {a, b});
#endif
            default:
                assert(m_mode == ival_mode::portable);
                // NOTE: in portable mode the rounding mode is set to up already, and the builder setup will ensure
                // that a constrained intrinsic is emitted.
                return portable_op(m_state, a, b);
        }
    }
};

// The only ctor sets up a guard for the function f, assuming that interval arithmetic will operate on the scalar/vector
// type val_t.
llvm_ival_guard::llvm_ival_guard(llvm_state &s, llvm::Function *const f, llvm::Type *const val_t)
    : m_impl(std::make_unique<impl>(s, f, val_t))
{
}

// NOTE: the dtor of m_impl->m_fmfg will ensure that the FP flags are restored to their original values.
llvm_ival_guard::~llvm_ival_guard() = default;

llvm::Type *llvm_ival_guard::get_val_t() const
{
    return m_impl->m_val_t;
}

llvm_state &llvm_ival_guard::get_state() const
{
    return m_impl->m_state;
}

// NOTE: finalise() must be called immediately before every return statement in the function.
void llvm_ival_guard::finalise() const
{
    if (m_impl->m_mode == ival_mode::portable) {
        assert(m_impl->m_orig_rounding != nullptr);
        llvm_restore_rounding(m_impl->m_state, m_impl->m_orig_rounding, m_impl->m_val_t);
    }
}

// Helper to compute a < b.
//
// NOTE: LLVM cannot lower constrained comparisons of vectors of fp128 on some platforms (e.g., x86-64 and aarch64 as of
// LLVM 23), nor constrained comparisons of vectors of x86_fp80 on x86-64 with AVX-512 (LLVM 19-21). Thus, for these
// types we compare element by element. This has no runtime cost, as the backend operates on these vectors element by
// element anyway.
llvm::Value *llvm_ival_guard::lt(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    assert(a->getType() == m_impl->m_val_t);

    auto &s = m_impl->m_state;
    auto &builder = s.builder();

    // Fetch the scalar type.
    const auto *const fp_t = m_impl->m_val_t->getScalarType();

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
}

// Helper to compute max(a, b).
//
// NOTE: for performance reasons, for float and double we use the maximumnum LLVM intrinsic, which maps to hardware max
// instructions, and fall back to comparison + select for the other types (for which maximumnum has worse code
// generation). Note that these two paths can give different results when NaN is involved or when a and b are zeros of
// opposite sign, thus callers must not rely on which result is returned in these cases.
//
// NOTE: the only potential issue with maximumnum is that apparently it currently does not have a constrained
// counterpart, and this seemingly breaks the LLVM rule that "if any FP operation in a function is constrained then they
// all must be constrained". There are exceptions to this rule (e.g., abs()) but it is not clear if maximumnum also
// falls within the exception. In any case, in case of issues, we can always revert to using the comparison + select.
//
// NOTE: maximumnum is available since LLVM 20, but LLVM 20 and 21 miscompile it on x86-64. With earlier versions,
// we always use the comparison + select.
llvm::Value *llvm_ival_guard::max(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    assert(a->getType() == m_impl->m_val_t);

    auto &builder = m_impl->m_state.builder();

#if LLVM_VERSION_MAJOR >= 22
    // Fetch the scalar type.
    const auto *const fp_t = m_impl->m_val_t->getScalarType();

    if (fp_t->isFloatTy() || fp_t->isDoubleTy()) {
        return builder.CreateMaximumNum(a, b);
    }
#endif

    return builder.CreateSelect(lt(a, b), b, a);
}

llvm::Value *llvm_ival_guard::add_up(llvm::Value *const a, llvm::Value *const b) const
{
    return m_impl->arith_op_up("add", llvm_fadd, a, b);
}

llvm::Value *llvm_ival_guard::sub_up(llvm::Value *const a, llvm::Value *const b) const
{
    return m_impl->arith_op_up("sub", llvm_fsub, a, b);
}

llvm::Value *llvm_ival_guard::mul_up(llvm::Value *const a, llvm::Value *const b) const
{
    return m_impl->arith_op_up("mul", llvm_fmul, a, b);
}

// Helper to compute -a.
//
// NOTE: negation is exact. For builtin FP types and vector counterparts, this emits an fneg instruction (which has no
// constrained counterpart), for mppp::real an inline sign flip.
llvm::Value *llvm_ival_guard::neg(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_val_t);

    return llvm_fneg(m_impl->m_state, a);
}

// Helper to compute |a|.
//
// NOTE: the absolute value is exact. For float, double and long double (scalar and vector), this emits the llvm.fabs
// intrinsic, for real128 (when distinct from long double) a call to fabsq() for each element (with fabsq() only
// declared, never defined), for mppp::real a call to a wrapper of mpfr_abs() generated via real_nary_op() (which is
// allowed while the guard is alive, see the class documentation).
llvm::Value *llvm_ival_guard::abs(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_val_t);

    return llvm_abs(m_impl->m_state, a);
}

} // namespace detail

HEYOKA_END_NAMESPACE
