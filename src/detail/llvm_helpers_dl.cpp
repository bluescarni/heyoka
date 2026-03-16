// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <utility>

#include <llvm/IR/FMF.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// RAII helper to temporarily disable fast math flags in a builder.
class fmf_disabler
{
    ir_builder *m_builder;
    const llvm::FastMathFlags m_orig_fmf;

public:
    explicit fmf_disabler(ir_builder &b) : m_builder(&b), m_orig_fmf(m_builder->getFastMathFlags())
    {
        // Reset the fast math flags.
        m_builder->setFastMathFlags(llvm::FastMathFlags{});
    }
    ~fmf_disabler()
    {
        // Restore the original fast math flags.
        m_builder->setFastMathFlags(m_orig_fmf);
    }

    fmf_disabler(const fmf_disabler &) = delete;
    fmf_disabler(fmf_disabler &&) = delete;

    fmf_disabler &operator=(const fmf_disabler &) = delete;
    fmf_disabler &operator=(fmf_disabler &&) = delete;
};

} // namespace

// Addition.
// NOTE: this is an LLVM port of the original code in NTL.
// See the C++ implementation in dfloat.hpp for an explanation.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_add(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *S = llvm_fadd(state, x_hi, y_hi);
    auto *T = llvm_fadd(state, x_lo, y_lo);
    auto *e = llvm_fsub(state, S, x_hi);
    auto *f = llvm_fsub(state, T, x_lo);

    auto *t1 = llvm_fsub(state, S, e);
    t1 = llvm_fsub(state, x_hi, t1);
    auto *s = llvm_fsub(state, y_hi, e);
    s = llvm_fadd(state, s, t1);

    t1 = llvm_fsub(state, T, f);
    t1 = llvm_fsub(state, x_lo, t1);
    auto *t = llvm_fsub(state, y_lo, f);
    t = llvm_fadd(state, t, t1);

    s = llvm_fadd(state, s, T);
    auto *H = llvm_fadd(state, S, s);
    auto *h = llvm_fsub(state, S, H);
    h = llvm_fadd(state, h, s);

    h = llvm_fadd(state, h, t);
    e = llvm_fadd(state, H, h);
    f = llvm_fsub(state, H, e);
    f = llvm_fadd(state, f, h);

    return {e, f};
}

// Subtraction.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_sub(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    return llvm_dl_add(state, x_hi, x_lo, llvm_fneg(state, y_hi), llvm_fneg(state, y_lo));
}

// Error-free transformation of the product of two floating point numbers
// using an FMA. This is algorithm 2.5 here:
// https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
std::pair<llvm::Value *, llvm::Value *> llvm_eft_product(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    // LCOV_EXCL_START
    assert(a != nullptr);
    assert(b != nullptr);
    assert(a->getType() == b->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *x = llvm_fmul(s, a, b);
    auto *y = llvm_fma(s, a, b, llvm_fneg(s, x));

    return {x, y};
}

// Multiplication.
// NOTE: this is procedure mul2() from here:
// https://link.springer.com/content/pdf/10.1007/BF01397083.pdf
// The mul12() function is replaced with the FMA-based llvm_eft_product().
// NOTE: the code in NTL looks identical to Dekker's.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_mul(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo,
                                                    llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto [c, cc] = llvm_eft_product(s, x_hi, y_hi);

    // cc = x*yy + xx*y + cc.
    auto *x_yy = llvm_fmul(s, x_hi, y_lo);
    auto *xx_y = llvm_fmul(s, x_lo, y_hi);
    cc = llvm_fadd(s, llvm_fadd(s, x_yy, xx_y), cc);

    // The normalisation step.
    auto *z = llvm_fadd(s, c, cc);
    auto *zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

    return {z, zz};
}

// Division.
// NOTE: this is procedure div2() from here:
// https://link.springer.com/content/pdf/10.1007/BF01397083.pdf
// The mul12() function is replaced with the FMA-based llvm_eft_product().
// NOTE: the code in NTL looks identical to Dekker's.
std::pair<llvm::Value *, llvm::Value *> llvm_dl_div(llvm_state &s,
                                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                                    llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi,
                                                    llvm::Value *y_lo)
{
    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *c = llvm_fdiv(s, x_hi, y_hi);

    auto [u, uu] = llvm_eft_product(s, c, y_hi);

    // cc = (x_hi - u - uu + x_lo - c * y_lo) / y_hi.
    auto *cc = llvm_fsub(s, x_hi, u);
    cc = llvm_fsub(s, cc, uu);
    cc = llvm_fadd(s, cc, x_lo);
    cc = llvm_fsub(s, cc, llvm_fmul(s, c, y_lo));
    cc = llvm_fdiv(s, cc, y_hi);

    // The normalisation step.
    auto *z = llvm_fadd(s, c, cc);
    auto *zz = llvm_fadd(s, llvm_fsub(s, c, z), cc);

    return {z, zz};
}

// Floor.
// NOTE: code taken from NTL:
// https://github.com/libntl/ntl/blob/main/src/quad_float1.cpp#L239
std::pair<llvm::Value *, llvm::Value *> llvm_dl_floor(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo)
{
    // LCOV_EXCL_START
    assert(x_hi != nullptr);
    assert(x_lo != nullptr);
    assert(x_hi->getType() == x_lo->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    auto *fp_t = x_hi->getType();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    // Floor x_hi.
    auto *fhi = llvm_floor(s, x_hi);

    // NOTE: we want to distinguish the scalar/vector codepaths, as the vectorised implementation
    // does more work.
    const auto vec_size = get_vector_size(x_hi);

    if (vec_size == 1u) {
        auto *ret_hi_ptr = builder.CreateAlloca(fp_t);
        auto *ret_lo_ptr = builder.CreateAlloca(fp_t);

        llvm_if_then_else(
            s, llvm_fcmp_oeq(s, fhi, x_hi),
            [&]() {
                // floor(x_hi) == x_hi, that is, x_hi is already
                // an integral value.

                // Floor the low part.
                auto *flo = llvm_floor(s, x_lo);

                // Normalise.
                auto *z = llvm_fadd(s, fhi, flo);
                auto *zz = llvm_fadd(s, llvm_fsub(s, fhi, z), flo);

                // Store.
                builder.CreateStore(z, ret_hi_ptr);
                builder.CreateStore(zz, ret_lo_ptr);
            },
            [&]() {
                // floor(x_hi) != x_hi. Just need to set the low part to zero.
                builder.CreateStore(fhi, ret_hi_ptr);
                builder.CreateStore(llvm_constantfp(s, fp_t, 0.), ret_lo_ptr);
            });

        return {builder.CreateLoad(fp_t, ret_hi_ptr), builder.CreateLoad(fp_t, ret_lo_ptr)};
    } else {
        // Get a vector of zeroes.
        auto *zero_vec = llvm_constantfp(s, fp_t, 0.);

        // Floor the low part.
        auto *flo = llvm_floor(s, x_lo);

        // Select flo or zero_vec, depending on fhi == x_hi.
        auto *ret_lo = builder.CreateSelect(llvm_fcmp_oeq(s, fhi, x_hi), flo, zero_vec);

        // Normalise.
        auto *z = llvm_fadd(s, fhi, ret_lo);
        auto *zz = llvm_fadd(s, llvm_fsub(s, fhi, z), ret_lo);

        return {z, zz};
    }
}

// Helper to reduce x modulo y, that is, to compute:
// x - y * floor(x / y).
std::pair<llvm::Value *, llvm::Value *> llvm_dl_modulus(llvm_state &s, llvm::Value *x_hi, llvm::Value *x_lo,
                                                        llvm::Value *y_hi, llvm::Value *y_lo)
{
    // LCOV_EXCL_START
    assert(x_hi != nullptr);
    assert(x_lo != nullptr);
    assert(y_hi != nullptr);
    assert(y_lo != nullptr);
    assert(x_hi->getType() == x_lo->getType());
    assert(x_hi->getType() == y_hi->getType());
    assert(x_hi->getType() == y_lo->getType());
    // LCOV_EXCL_STOP

    auto &builder = s.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto [xoy_hi, xoy_lo] = llvm_dl_div(s, x_hi, x_lo, y_hi, y_lo);
    auto [fl_hi, fl_lo] = llvm_dl_floor(s, xoy_hi, xoy_lo);
    auto [prod_hi, prod_lo] = llvm_dl_mul(s, y_hi, y_lo, fl_hi, fl_lo);

    return llvm_dl_sub(s, x_hi, x_lo, prod_hi, prod_lo);
}

// Less-than.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_dl_lt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *cond1 = llvm_fcmp_olt(state, x_hi, y_hi);
    auto *cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto *cond3 = llvm_fcmp_olt(state, x_lo, y_lo);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto *cond4 = builder.CreateLogicalAnd(cond2, cond3);
    auto *cond = builder.CreateLogicalOr(cond1, cond4);

    return cond;
}

// Greater-than.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
llvm::Value *llvm_dl_gt(llvm_state &state, llvm::Value *x_hi, llvm::Value *x_lo, llvm::Value *y_hi, llvm::Value *y_lo)
{
    auto &builder = state.builder();

    // Temporarily disable the fast math flags.
    const fmf_disabler fd(builder);

    auto *cond1 = llvm_fcmp_ogt(state, x_hi, y_hi);
    auto *cond2 = llvm_fcmp_oeq(state, x_hi, y_hi);
    auto *cond3 = llvm_fcmp_ogt(state, x_lo, y_lo);
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto *cond4 = builder.CreateLogicalAnd(cond2, cond3);
    auto *cond = builder.CreateLogicalOr(cond1, cond4);

    return cond;
}

} // namespace detail

HEYOKA_END_NAMESPACE
