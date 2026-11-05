// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <boost/predef/architecture.h>
#include <boost/scope/scope_exit.hpp>

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
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
// "add", "sub", "mul" or "div". a and b must be of the same type, which must be either float/double or a 512-bit vector
// of float/double (i.e., of size 16/8).
//
// NOTE: static rounding is available only for scalar operations and for 512-bit vectors. The scalar intrinsics operate
// on the first element of 128-bit vectors, thus scalar operands are inserted into (and the result is extracted from)
// the first element of 128-bit vectors.
llvm::Value *avx512_binop_up(llvm_state &s, const std::string_view op, llvm::Value *const a, llvm::Value *const b)
{
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    assert(op == "add" || op == "sub" || op == "mul" || op == "div");

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

// Helper to widen a size-2 float/double SIMD vector to the full 512-bit AVX512 size.
llvm::Value *avx512_widen_vec2(llvm_state &s, llvm::Value *const x)
{
    assert(x != nullptr);
    assert(get_vector_size(x) == 2u);
    assert(x->getType()->getScalarType()->isDoubleTy() || x->getType()->getScalarType()->isFloatTy());

    // NOTE: the widening is done via a shuffle: keep the first two elements in their original positions, fill the rest
    // with poison values.
    const auto impl = [&s, x]<std::size_t N> {
        std::array<int, N> widen_mask{};
        std::ranges::fill(widen_mask, llvm::PoisonMaskElem);
        widen_mask[0] = 0;
        widen_mask[1] = 1;
        return s.builder().CreateShuffleVector(x, widen_mask);
    };

    if (x->getType()->getScalarType()->isDoubleTy()) {
        return impl.operator()<8>();
    } else {
        return impl.operator()<16>();
    }
}

// Helper to shrink a full 512-bit AVX512 vector to size 2 (keeping only the first 2 elements).
llvm::Value *avx512_shrink_to_vec2(llvm_state &s, llvm::Value *const x)
{
    assert(x != nullptr);
    assert((x->getType()->getScalarType()->isDoubleTy() && get_vector_size(x) == 8u)
           || (x->getType()->getScalarType()->isFloatTy() && get_vector_size(x) == 16u));

    return s.builder().CreateShuffleVector(x, {0, 1});
}

// LCOV_EXCL_STOP

} // namespace

// Implementation of llvm_ival_guard.
struct llvm_ival_guard::impl {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    llvm_state &m_state;
    llvm::IRBuilderBase::FastMathFlagGuard m_fmfg;
    // The value type of the interval, that is, the type used to represent the lower/upper bounds.
    llvm::Type *m_val_t = nullptr;
    // The type representing the interval [lo, hi] as a (-lo, hi) pair. This can be a size-2 vector or a struct with two
    // fields.
    llvm::Type *m_ival_t = nullptr;
    ival_mode m_mode = ival_mode::portable;
    llvm::Value *m_orig_rounding = nullptr;

    // The ctor sets up a guard for the function f, assuming that the bounds of the intervals are of the scalar/vector
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

        // Determine the interval type.
        //
        // An interval [lo, hi], with lo<=hi, is represented as the pair (mlo, hi), where mlo=-lo. Carrying around the
        // negated lower bound allows interval arithmetic to be implemented using only rounding-up operations, as
        // opposed to having to mix rounding up with rounding down.
        auto *const fp_t = m_val_t->getScalarType();
        if (!m_val_t->isVectorTy() && (fp_t->isFloatTy() || fp_t->isDoubleTy())) {
            // For scalar float and double, intervals are represented as vectors of size 2. This allows the interval
            // arithmetic primitives to be vectorized.
            m_ival_t = make_vector_type(fp_t, 2);
        } else {
            // Otherwise, the default representation of an interval is a literal struct with two members of type
            // m_val_t.
            m_ival_t = llvm::StructType::get(m_state.context(), {m_val_t, m_val_t});
        }
    }

    // Common helper to compute 'a name b' rounded toward +inf, where the type of a and b is assumed to be the value
    // type of the interval.
    //
    // 'name' is the name of the operation ("add", "sub", "mul" or "div"), used to select the AVX512 intrinsic and the
    // MPFR primitive. portable_op is the function implementing the operation in the portable implementation.
    llvm::Value *val_arith_op_up(const std::string_view name, auto *const portable_op, llvm::Value *const a,
                                 llvm::Value *const b) const
    {
        assert(a != nullptr);
        assert(b != nullptr);
        assert(a->getType() == b->getType());
        assert(a->getType() == m_val_t);
        assert(name == "add" || name == "sub" || name == "mul" || name == "div");

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

    // Helper to compute the maximum of a and b, where the type of a and b is assumed to be the value type of the
    // interval.
    //
    // NOTE: for performance reasons, for float and double we use the maximumnum() LLVM intrinsic, which maps to
    // hardware max instructions, and fall back to comparison + select for the other types (for which maximumnum() has
    // worse code generation). Note that these two paths can give different results when NaN is involved or when a and b
    // are zeros of opposite sign, thus callers must not rely on which result is returned in these cases.
    //
    // NOTE: the only potential issue with maximumnum() is that apparently it currently does not have a constrained
    // counterpart, and this seemingly breaks the LLVM rule that "if any FP operation in a function is constrained then
    // they all must be constrained". There are exceptions to this rule (e.g., abs()) but it is not clear if
    // maximumnum() also falls within the exception. In any case, in case of issues, we can always revert to using the
    // comparison + select.
    //
    // NOTE: maximumnum() is available since LLVM 20, but LLVM 20 and 21 miscompile it on x86-64. With earlier versions,
    // we always use the comparison + select.
    llvm::Value *val_max(llvm::Value *const a, llvm::Value *const b) const
    {
        assert(a != nullptr);
        assert(b != nullptr);
        assert(a->getType() == b->getType());
        assert(a->getType() == m_val_t);

        auto &builder = m_state.builder();

#if LLVM_VERSION_MAJOR >= 22
        // Fetch the scalar type.
        const auto *const fp_t = m_val_t->getScalarType();

        if (fp_t->isFloatTy() || fp_t->isDoubleTy()) {
            return builder.CreateMaximumNum(a, b);
        }
#endif

        return builder.CreateSelect(val_lt(a, b), b, a);
    }

    // Helper to compute the result of a<b, where the type of a and b is assumed to be the value type of the interval.
    //
    // NOTE: LLVM cannot lower constrained comparisons of vectors of fp128 on some platforms (e.g., x86-64 and aarch64
    // as of LLVM 23), nor constrained comparisons of vectors of x86_fp80 on x86-64 with AVX-512 (LLVM 19-21). Thus, for
    // these types we compare element by element. This has no runtime cost, as the backend operates on these vectors
    // element by element anyway.
    //
    // NOTE: here we are adopting a performance workaround for scalar float and double computations in portable mode on
    // x86-64, where we force the usage of a plain (i.e., non-constrained) comparison. On x86-64, a select on the result
    // of a constrained float/double scalar comparison becomes a conditional branch (because the comparison sets the
    // flags register), whereas with a plain comparison it becomes a compare + blend instruction sequence: in a
    // benchmark, this made the rescaling of a polynomial of degree 20 about 4 times faster with coefficients of random
    // signs. This technically breaks the LLVM rule that "if any FP operation in a function is constrained then they all
    // must be constrained" (like the use of maximumnum() in val_max()), but it should be OK in practice since
    // comparisons are not affected by the active rounding mode. If this ever becomes an issue, we can just remove the
    // workaround code block.
    llvm::Value *val_lt(llvm::Value *const a, llvm::Value *const b) const
    {
        assert(a != nullptr);
        assert(b != nullptr);
        assert(a->getType() == b->getType());
        assert(a->getType() == m_val_t);

        auto &builder = m_state.builder();

#if BOOST_ARCH_X86_64

        if (m_mode == ival_mode::portable && (m_val_t->isFloatTy() || m_val_t->isDoubleTy())) {
            // LCOV_EXCL_START

            // Temporarily disable the constrained flag and make sure it is restored on function exit.
            const boost::scope::scope_exit constrained_guard(
                [&builder, orig_constr = builder.getIsFPConstrained()] noexcept {
                    builder.setIsFPConstrained(orig_constr);
                });
            builder.setIsFPConstrained(false);

            return llvm_fcmp_olt(m_state, a, b);

            // LCOV_EXCL_STOP
        }

#endif

        // Fetch the scalar type.
        const auto *const fp_t = m_val_t->getScalarType();

        if (fp_t->isFP128Ty() || fp_t->isX86_FP80Ty()) {
            const auto a_scalars = vector_to_scalars(builder, a);
            const auto b_scalars = vector_to_scalars(builder, b);

            std::vector<llvm::Value *> res;
            res.reserve(a_scalars.size());
            for (decltype(a_scalars.size()) i = 0; i < a_scalars.size(); ++i) {
                res.push_back(llvm_fcmp_olt(m_state, a_scalars[i], b_scalars[i]));
            }

            return scalars_to_vector(builder, res);
        }

        return llvm_fcmp_olt(m_state, a, b);
    }

    // Helper to compute a + b rounded toward +inf, where the type of a and b is assumed to be the value type of the
    // interval (i.e., a and b are interval bounds, not intervals).
    llvm::Value *val_add_up(llvm::Value *const a, llvm::Value *const b) const
    {
        return val_arith_op_up("add", llvm_fadd, a, b);
    }

    // Helper to compute a - b rounded toward +inf, where the type of a and b is assumed to be the value type of the
    // interval (i.e., a and b are interval bounds, not intervals).
    llvm::Value *val_sub_up(llvm::Value *const a, llvm::Value *const b) const
    {
        return val_arith_op_up("sub", llvm_fsub, a, b);
    }

    // Helper to compute a * b rounded toward +inf, where the type of a and b is assumed to be the value type of the
    // interval (i.e., a and b are interval bounds, not intervals).
    llvm::Value *val_mul_up(llvm::Value *const a, llvm::Value *const b) const
    {
        return val_arith_op_up("mul", llvm_fmul, a, b);
    }

    // Helper to compute a / b rounded toward +inf, where the type of a and b is assumed to be the value type of the
    // interval (i.e., a and b are interval bounds, not intervals).
    llvm::Value *val_div_up(llvm::Value *const a, llvm::Value *const b) const
    {
        return val_arith_op_up("div", llvm_fdiv, a, b);
    }
};

// The only ctor sets up a guard for the function f, assuming that the bounds of the intervals are of the scalar/vector
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

llvm::Type *llvm_ival_guard::get_ival_t() const
{
    return m_impl->m_ival_t;
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

// Helper to construct an interval from the negated lower bound mlo and the upper bound hi.
//
// It is assumed that the two components result in a valid interval.
llvm::Value *llvm_ival_guard::ival_construct(llvm::Value *const mlo, llvm::Value *const hi) const
{
    assert(mlo != nullptr);
    assert(mlo->getType() == m_impl->m_val_t);
    assert(hi != nullptr);
    assert(hi->getType() == m_impl->m_val_t);

    auto &bld = m_impl->m_state.builder();

    // Init the interval as a poison value.
    auto *const ival_t = m_impl->m_ival_t;
    llvm::Value *ret = llvm::PoisonValue::get(ival_t);

    // Fill in the components.
    if (ival_t->isVectorTy()) {
        ret = bld.CreateInsertElement(ret, mlo, static_cast<std::uint64_t>(0));
        ret = bld.CreateInsertElement(ret, hi, static_cast<std::uint64_t>(1));
    } else {
        ret = bld.CreateInsertValue(ret, mlo, {0u});
        ret = bld.CreateInsertValue(ret, hi, {1u});
    }

    return ret;
}

// Extract and return the two components mlo and hi of the input interval (mlo, hi).
std::pair<llvm::Value *, llvm::Value *> llvm_ival_guard::ival_extract(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);

    auto &bld = m_impl->m_state.builder();

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        auto *const mlo = bld.CreateExtractElement(a, static_cast<std::uint64_t>(0));
        auto *const hi = bld.CreateExtractElement(a, static_cast<std::uint64_t>(1));
        return {mlo, hi};
    } else {
        assert(ival_t->isStructTy());
        auto *const mlo = bld.CreateExtractValue(a, {0u});
        auto *const hi = bld.CreateExtractValue(a, {1u});
        return {mlo, hi};
    }
}

// Helper to compute the sum of two intervals.
llvm::Value *llvm_ival_guard::ival_add(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_ival_t);

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        auto &s = m_impl->m_state;

        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, a);
            auto *const bw = avx512_widen_vec2(s, b);

            // 512-bit addition with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "add", aw, bw);

            // Narrow back to the first two lanes.
            return avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just add the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            return llvm_fadd(s, a, b);
        }
        // LCOV_EXCL_STOP
    } else {
        assert(ival_t->isStructTy());

        // Extract the components.
        const auto [a_mlo, a_hi] = ival_extract(a);
        const auto [b_mlo, b_hi] = ival_extract(b);

        // Compute the addition.
        auto *const ret_mlo = m_impl->val_add_up(a_mlo, b_mlo);
        auto *const ret_hi = m_impl->val_add_up(a_hi, b_hi);

        // Construct and return the result.
        return ival_construct(ret_mlo, ret_hi);
    }
}

// Helper to compute the sum of an interval a and a point b.
//
// b must be finite.
llvm::Value *llvm_ival_guard::ival_add_val(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_val_t);

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        // For the vector implementation, we construct an interval from b and then add it to a.
        auto *const b_ival = ival_construct(val_neg(b), b);
        return ival_add(a, b_ival);
    } else {
        // For the struct implementation, we can optimise slightly by replacing a neg+sum with a sub.
        assert(ival_t->isStructTy());

        // Extract the components of a.
        const auto [a_mlo, a_hi] = ival_extract(a);

        // Compute the addition.
        auto *const ret_mlo = m_impl->val_sub_up(a_mlo, b);
        auto *const ret_hi = m_impl->val_add_up(a_hi, b);

        // Construct and return the result.
        return ival_construct(ret_mlo, ret_hi);
    }
}

// Helper to negate a, where the type of a is assumed to be the value type of the interval.
//
// NOTE: negation is exact. For builtin FP types and vector counterparts, this emits an fneg instruction (which has
// no constrained counterpart), for mppp::real an inline sign flip.
llvm::Value *llvm_ival_guard::val_neg(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_val_t);

    return llvm_fneg(m_impl->m_state, a);
}

// Multiply the interval a by the interval [0, b], where b is a non-negative value.
//
// The standard interval arithmetic formula is:
//
// [lo, hi] * [0, b] = [min(lo*0, lo*b, hi*0, hi*b), max(lo*0, lo*b, hi*0, hi*b)],
//
// which simplifies to [min(lo*b, 0), max(hi*b, 0)], because lo<=hi and b>=0. Note that the simplification relies on
// hi*0 and lo*0 going to zero, which looks a bit iffy if hi/lo are infinite. We will be assuming here that infinities
// in the interval a originate from overflow, not from poles/singularities. The infinities can thus be regarded as
// representing large values that, when multiplied by zero, produce zero as a result, thus the simplification still
// holds.
//
// Because in our internal representation we are carrying around mlo=-lo, instead of lo, the final formula is:
//
// (mlo, hi) * (0, b) = (max(mlo*b, 0), max(hi*b, 0))
//
// where the two multiplications are performed while rounding up.
//
// In order to guarantee that the interval returned by this function is valid, if b is zero then no bound of a may be
// infinite, and if b is +inf, no interval bound may be zero.
llvm::Value *llvm_ival_guard::ival_mul_zlb(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_val_t);

    if (auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        auto &s = m_impl->m_state;
        auto &bld = s.builder();

        // Splat b into a size-2 vector.
        auto *const b_vec = bld.CreateVectorSplat(2, b);

        // Compute the multiplications with a single rounding-up size-2 vector mul.
        llvm::Value *mul_res = nullptr;
        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, a);
            auto *const bw = avx512_widen_vec2(s, b_vec);

            // 512-bit multiplication with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "mul", aw, bw);

            // Narrow back to the first two lanes.
            mul_res = avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just mul the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            mul_res = llvm_fmul(s, a, b_vec);
        }
        // LCOV_EXCL_STOP

        // Run the max with zero.
        auto *const zero = llvm_constantfp(s, ival_t, 0.);

        // NOTE: if possible we use maximumnum(), with the same caveats as explained in impl::val_max(). Otherwise, we
        // use an fcmp olt + select: in portable mode the constrained intrinsic will be used, in AVX512 mode a plain
        // fcmp will be issued. Either way, comparisons are not affected by FP rounding.
        //
        // NOTE: the behaviour of this block is essentially equivalent to impl::val_max(). Note that mul_res cannot
        // contain NaN under the function's preconditions, and the sign of a zero in mul_res is irrelevant. Also note
        // that we don't apply the workarounds from impl::val_lt() here: we are sure that ival_t is a vector of
        // float/double, for which LLVM lowers constrained comparisons correctly, and a vector compare + select does not
        // produce a branch.
#if LLVM_VERSION_MAJOR >= 22
        auto *const ret = bld.CreateMaximumNum(mul_res, zero);
#else
        auto *const ret = bld.CreateSelect(llvm_fcmp_olt(s, mul_res, zero), zero, mul_res);
#endif

        return ret;
    } else {
        assert(ival_t->isStructTy());

        // Extract the components of a.
        const auto [a_mlo, a_hi] = ival_extract(a);

        // Compute the multiplications.
        auto *const mul_mlo = m_impl->val_mul_up(a_mlo, b);
        auto *const mul_hi = m_impl->val_mul_up(a_hi, b);

        // Compute the result's components.
        auto *const zero = llvm_constantfp(m_impl->m_state, m_impl->m_val_t, 0.);
        // NOTE: val_max() is safe here, mul_mlo and mul_hi cannot be NaN under the function's preconditions, and the
        // sign of a zero result is irrelevant.
        auto *const ret_mlo = m_impl->val_max(mul_mlo, zero);
        auto *const ret_hi = m_impl->val_max(mul_hi, zero);

        // Construct and return the result.
        return ival_construct(ret_mlo, ret_hi);
    }
}

// Multiply the interval a by the non-negative value b.
//
// The standard interval arithmetic formula is:
//
// [lo, hi] * [b, b] = [min(lo*b, lo*b, hi*b, hi*b), max(lo*b, lo*b, hi*b, hi*b)],
//
// which simplifies to [lo*b, hi*b], because lo<=hi and b>=0.
//
// Because in our internal representation we are carrying around mlo=-lo, instead of lo, the final result is:
//
// (mlo*b, hi*b)
//
// where the two multiplications are performed while rounding up.
//
// In order to guarantee that the interval returned by this function is valid, the preconditions are that b is finite
// and that, if b is zero, no bound of a is infinite.
llvm::Value *llvm_ival_guard::ival_mul_nn_val(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_val_t);

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        auto &s = m_impl->m_state;
        auto &bld = s.builder();

        // Splat b into a size-2 vector.
        auto *const b_vec = bld.CreateVectorSplat(2, b);

        // Compute the multiplications with a single rounding-up size-2 vector mul.
        llvm::Value *mul_res = nullptr;
        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, a);
            auto *const bw = avx512_widen_vec2(s, b_vec);

            // 512-bit multiplication with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "mul", aw, bw);

            // Narrow back to the first two lanes.
            mul_res = avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just mul the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            mul_res = llvm_fmul(s, a, b_vec);
        }
        // LCOV_EXCL_STOP

        return mul_res;
    } else {
        assert(ival_t->isStructTy());

        // Extract the components of a.
        const auto [a_mlo, a_hi] = ival_extract(a);

        // Compute the multiplications.
        auto *const mul_mlo = m_impl->val_mul_up(a_mlo, b);
        auto *const mul_hi = m_impl->val_mul_up(a_hi, b);

        // Construct and return the result.
        return ival_construct(mul_mlo, mul_hi);
    }
}

// Multiply the value a by the non-negative interval b.
//
// The standard interval arithmetic formula is:
//
// [a, a] * [lo, hi] = [min(lo*a, lo*a, hi*a, hi*a), max(lo*a, lo*a, hi*a, hi*a)],
//
// which simplifies to [min(lo*a, hi*a), max(lo*a, hi*a)]. This becomes:
//
// - [lo*a, hi*a] if a>=0,
// - [hi*a, lo*a] if a<0.
//
// This can be further rewritten as:
//
// - [lo*|a|, hi*|a|] if a>=0,
// - [-hi*|a|, -lo*|a|] if a<0.
//
// In the lo-negated format, this becomes:
//
// - (mlo*|a|, hi*|a|) if a>=0,
// - (hi*|a|, mlo*|a|) if a<0,
//
// where all multiplications are performed while rounding up.
//
// The result is a valid interval if a is finite and b is a non-negative interval. Note that the non-negativity of b is
// not necessary in the mathematical derivation above, but it allows the struct implementation to handle a==0 without
// additional selects.
llvm::Value *llvm_ival_guard::val_mul_nn_ival(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_val_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_ival_t);

    auto &s = m_impl->m_state;
    auto &bld = s.builder();

    // Generate a scalar zero constant.
    auto *const zero = llvm_constantfp(s, m_impl->m_val_t, 0.);

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        // NOTE: in this branch we use direct calls to llvm_abs() and llvm_fcmp_olt() on size-2 vectors. We cannot use
        // this class' primitives for this, as they require different types. Both abs and fcmp are ok here: the former
        // has no constrained counterpart, the latter will produce a constrained comparison in portable mode, a normal
        // comparison in AVX512 mode. Either way, these operations are not affected by the rounding mode.

        // Splat a into a size-2 vector.
        //
        // NOTE: splatting a is not necessary per se, but it improves code generation in the selects below wrt the
        // scalar version (at least on x86): the scalar version results in branches, while the vector version yields
        // vector masks, letting the selects compile to blends instead of branches.
        auto *const a_vec = bld.CreateVectorSplat(2, a);

        // Compute |a|.
        auto *const abs_a_vec = llvm_abs(s, a_vec);

        // Generate a size-2 vector zero.
        auto *const zero_vec = bld.CreateVectorSplat(2, zero);

        // Compute the multiplications with a single rounding-up size-2 vector mul.
        llvm::Value *mul_res = nullptr;
        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, abs_a_vec);
            auto *const bw = avx512_widen_vec2(s, b);

            // 512-bit multiplication with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "mul", aw, bw);

            // Narrow back to the first two lanes.
            mul_res = avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just mul the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            mul_res = llvm_fmul(s, abs_a_vec, b);
        }
        // LCOV_EXCL_STOP

        // NOTE: the computation of the result requires swapping around the mul_res components if a is negative.
        // Moreover, if a==0, we want to forcibly set the result to zero in order to avoid generating NaNs via 0*inf
        // multiplications.

        // Build the swapped-around version of mul_res.
        auto *const mul_res_sw = bld.CreateShuffleVector(mul_res, {1, 0});

        // res = a < 0 ? mul_res_sw : (0, 0).
        auto *res = bld.CreateSelect(llvm_fcmp_olt(s, a_vec, zero_vec), mul_res_sw, zero_vec);
        // res = a > 0 ? mul_res : res.
        res = bld.CreateSelect(llvm_fcmp_olt(s, zero_vec, a_vec), mul_res, res);

        return res;
    } else {
        assert(ival_t->isStructTy());

        // Compute |a|.
        auto *const abs_a = val_abs(a);

        // Extract the components of b.
        const auto [b_mlo, b_hi] = ival_extract(b);

        // Compute the multiplications.
        //
        // NOTE: if a==0, both components are computed from b_mlo, which is finite because b is non-negative. This
        // avoids the NaN that 0*b_hi would produce if b_hi is +inf.
        auto *const mul_mlo = m_impl->val_mul_up(abs_a, bld.CreateSelect(val_lt(a, zero), b_hi, b_mlo));
        auto *const mul_hi = m_impl->val_mul_up(abs_a, bld.CreateSelect(val_lt(zero, a), b_hi, b_mlo));

        // Construct and return the result.
        return ival_construct(mul_mlo, mul_hi);
    }
}

// Compute a < b, where the type of a and b is assumed to be the value type of the interval.
llvm::Value *llvm_ival_guard::val_lt(llvm::Value *const a, llvm::Value *const b) const
{
    return m_impl->val_lt(a, b);
}

// Helper to compute |a|, where the type of a is assumed to be the value type of the interval.
//
// NOTE: the absolute value is exact. For float, double and long double (scalar and vector), this emits the llvm.fabs()
// intrinsic, for real128 (when distinct from long double) a call to fabsq() for each element (with fabsq() only
// declared, never defined), for mppp::real a call to a wrapper of mpfr_abs() generated via real_nary_op() (which is
// allowed while the guard is alive, see the class documentation).
llvm::Value *llvm_ival_guard::val_abs(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_val_t);

    return llvm_abs(m_impl->m_state, a);
}

// Helper to detect if the interval a does NOT contain zero.
//
// The return value is i1 for a scalar m_val_t, and a vector of i1 (one per batch lane) for a vector m_val_t.
llvm::Value *llvm_ival_guard::ival_nz(llvm::Value *const a) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);

    auto &s = m_impl->m_state;
    auto &bld = s.builder();

    if (auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        // The zero constant as a size-2 vector.
        auto *const zero = llvm_constantfp(s, ival_t, 0.);

        // The interval does not contain zero if lo > 0 (i.e., mlo < 0) or hi < 0.
        //
        // NOTE: in portable mode, a constrained fcmp will be emitted, in AVX512 mode a plain fcmp will be emitted.
        // Either way, this is correct and unaffected by rounding.
        return bld.CreateOrReduce(llvm_fcmp_olt(s, a, zero));
    } else {
        assert(ival_t->isStructTy());

        // The zero constant as an m_val_t.
        auto *const zero = llvm_constantfp(s, m_impl->m_val_t, 0.);

        // Extract the components of a.
        const auto [a_mlo, a_hi] = ival_extract(a);

        // The interval does not contain zero if lo > 0 (i.e., mlo < 0) or hi < 0.
        return bld.CreateOr(val_lt(a_mlo, zero), val_lt(a_hi, zero));
    }
}

// Divide the input interval a by the positive value b.
//
// The standard interval arithmetic formula is:
//
// [lo, hi] / [b, b] = [min(lo/b, hi/b), max(lo/b, hi/b)],
//
// which simplifies to [lo/b, hi/b] because b>0. In our format, this becomes
//
// (mlo/b, hi/b),
//
// with both divisions rounded up.
//
// The result is a valid interval if b is finite and positive.
llvm::Value *llvm_ival_guard::ival_div_pos_val(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_val_t);

    if (const auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        auto &s = m_impl->m_state;
        auto &bld = s.builder();

        // Splat b into a size-2 vector.
        auto *const b_vec = bld.CreateVectorSplat(2, b);

        llvm::Value *div_res = nullptr;
        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, a);
            auto *const bw = avx512_widen_vec2(s, b_vec);

            // 512-bit division with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "div", aw, bw);

            // Narrow back to the first two lanes.
            div_res = avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just div the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            div_res = llvm_fdiv(s, a, b_vec);
        }
        // LCOV_EXCL_STOP

        return div_res;
    } else {
        assert(ival_t->isStructTy());

        // Extract the components of a.
        const auto [a_mlo, a_hi] = ival_extract(a);

        // Compute the divisions.
        auto *const div_mlo = m_impl->val_div_up(a_mlo, b);
        auto *const div_hi = m_impl->val_div_up(a_hi, b);

        // Construct and return the result.
        return ival_construct(div_mlo, div_hi);
    }
}

// Divide the input interval a by the positive interval b.
//
// The standard interval arithmetic formula is:
//
// [a_lo, a_hi] / [b_lo, b_hi] =
// [min(a_lo/b_lo, a_lo/b_hi, a_hi/b_lo, a_hi/b_hi), max(a_lo/b_lo, a_lo/b_hi, a_hi/b_lo, a_hi/b_hi)].
//
// Because b is positive (0<b_lo<=b_hi), a_lo/b_lo <= a_hi/b_lo and a_lo/b_hi<=a_hi/b_hi, and thus this simplifies to:
//
// [min(a_lo/b_lo, a_lo/b_hi), max(a_hi/b_lo, a_hi/b_hi)].
//
// In other words:
//
// [a_lo < 0 ? a_lo/b_lo : a_lo/b_hi, a_hi < 0 ? a_hi/b_hi : a_hi/b_lo].
//
// In our lo-negated format, this becomes:
//
// (a_mlo > 0 ? a_mlo/(-b_mlo) : a_mlo/b_hi, a_hi < 0 ? a_hi/b_hi : a_hi/(-b_mlo)),
//
// where all divisions are rounded up.
//
// In order to guarantee that the interval returned by this function is valid, b must be positive and finite
// (i.e., b_lo>0 and b_hi<+inf).
llvm::Value *llvm_ival_guard::ival_div_pos_ival(llvm::Value *const a, llvm::Value *const b) const
{
    assert(a != nullptr);
    assert(a->getType() == m_impl->m_ival_t);
    assert(b != nullptr);
    assert(b->getType() == m_impl->m_ival_t);

    auto &s = m_impl->m_state;
    auto &bld = s.builder();

    if (auto *const ival_t = m_impl->m_ival_t; ival_t->isVectorTy()) {
        // In vector mode, the dividend is the interval (a_mlo, a_hi) represented as a vector. The divisor vector is
        // built by choosing, lane by lane, between the divisors used if the conditions in the formula above are true
        // (yes_div) and the divisors used otherwise (no_div).

        // NOTE: in this branch we use direct calls to llvm_fneg() and llvm_fcmp_olt() on size-2 vectors. We cannot use
        // this class' primitives for this, as they require different types. Both negation and fcmp are ok here: the
        // former has no constrained counterpart and it will result in a simple vector negation in both portable and
        // AVX512 mode, the latter will produce a constrained comparison in portable mode, a normal comparison in AVX512
        // mode. Either way, these operations are not affected by the rounding mode.

        // yes_div = (b_lo, b_hi), built by taking the first lane from -b and the second lane from b.
        auto *const yes_div = bld.CreateShuffleVector(b, llvm_fneg(s, b), {2, 1});
        // no_div = (b_hi, b_lo), built by swapping the lanes of yes_div.
        auto *const no_div = bld.CreateShuffleVector(yes_div, {1, 0});

        // mask = (a_mlo > 0, a_hi < 0), built by comparing with zero the vector a_cmp = (a_mlo, -a_hi), which in turn
        // is built by taking the first lane from a and the second lane from -a.
        auto *const a_cmp = bld.CreateShuffleVector(a, llvm_fneg(s, a), {0, 3});
        auto *const mask = llvm_fcmp_olt(s, llvm_constantfp(s, ival_t, 0.), a_cmp);

        // Choose the divisors lane by lane.
        auto *const divisor = bld.CreateSelect(mask, yes_div, no_div);

        // Divide a by the divisors.
        // LCOV_EXCL_START
        if (m_impl->m_mode == ival_mode::avx512) {
            // Widen to full 512-bit vectors.
            auto *const aw = avx512_widen_vec2(s, a);
            auto *const dw = avx512_widen_vec2(s, divisor);

            // 512-bit division with static rounding up: avx512_binop_up() already handles 512-bit vectors.
            auto *const rw = avx512_binop_up(s, "div", aw, dw);

            // Narrow back to the first two lanes.
            return avx512_shrink_to_vec2(s, rw);
        } else {
            // NOTE: if we get here, we must be in portable mode. The rounding has been set to up by the guard and the
            // builder is set up to emit constrained operations, hence we can just div the vectors component-wise.
            assert(m_impl->m_mode == ival_mode::portable);
            return llvm_fdiv(s, a, divisor);
        }
        // LCOV_EXCL_STOP
    } else {
        assert(ival_t->isStructTy());

        // Extract the components of a and b, and compute b_lo.
        const auto [a_mlo, a_hi] = ival_extract(a);
        const auto [b_mlo, b_hi] = ival_extract(b);
        auto *const b_lo = val_neg(b_mlo);

        // Compute the divisions as in the formula above.
        auto *const zero = llvm_constantfp(s, m_impl->m_val_t, 0.);
        auto *const div_mlo = m_impl->val_div_up(a_mlo, bld.CreateSelect(val_lt(zero, a_mlo), b_lo, b_hi));
        auto *const div_hi = m_impl->val_div_up(a_hi, bld.CreateSelect(val_lt(a_hi, zero), b_hi, b_lo));

        // Construct and return the result.
        return ival_construct(div_mlo, div_hi);
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
