// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/ModRef.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <heyoka/detail/real_helpers.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Floor.
llvm::Value *llvm_floor(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.floor",
#if defined(HEYOKA_HAVE_REAL128)
                          "floorq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_floor",
#endif
                          {x});
}

// Trunc.
llvm::Value *llvm_trunc(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.trunc",
#if defined(HEYOKA_HAVE_REAL128)
                          "truncq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_trunc",
#endif
                          {x});
}

// Two-argument arctan.
llvm::Value *llvm_atan2(llvm_state &s, llvm::Value *y, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan2", {y, x});
}

// Exponential.
llvm::Value *llvm_exp(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.exp",
#if defined(HEYOKA_HAVE_REAL128)
                          "expq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_exp",
#endif
                          {x});
}

// Cosine.
llvm::Value *llvm_cos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.cos",
#if defined(HEYOKA_HAVE_REAL128)
                          "cosq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_cos",
#endif
                          {x});
}

// Sine.
llvm::Value *llvm_sin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sin",
#if defined(HEYOKA_HAVE_REAL128)
                          "sinq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sin",
#endif
                          {x});
}

// Hyperbolic cosine.
llvm::Value *llvm_cosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "cosh", {x});
}

// Error function.
llvm::Value *llvm_erf(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "erf", {x});
}

// Natural logarithm.
llvm::Value *llvm_log(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.log",
#if defined(HEYOKA_HAVE_REAL128)
                          "logq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_log",
#endif
                          {x});
}

// Inverse sine.
llvm::Value *llvm_asin(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asin", {x});
}

// Inverse hyperbolic sine.
llvm::Value *llvm_asinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "asinh", {x});
}

// Inverse tangent.
llvm::Value *llvm_atan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atan", {x});
}

// Inverse hyperbolic tangent.
llvm::Value *llvm_atanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "atanh", {x});
}

// Sigmoid.
llvm::Value *llvm_sigmoid(llvm_state &s, llvm::Value *x)
{
    // LCOV_EXCL_START
    assert(x != nullptr);
    // LCOV_EXCL_STOP

    // Create the 1 constant.
    auto *one_fp = llvm_constantfp(s, x->getType(), 1.);

    // Compute -x.
    auto *m_x = llvm_fneg(s, x);

    // Compute e^(-x).
    auto *e_m_x = llvm_exp(s, m_x);

    // Return 1 / (1 + e_m_arg).
    return llvm_fdiv(s, one_fp, llvm_fadd(s, one_fp, e_m_x));
}

// Inverse cosine.
llvm::Value *llvm_acos(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acos", {x});
}

// Inverse hyperbolic cosine.
llvm::Value *llvm_acosh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "acosh", {x});
}

// Hyperbolic sine.
llvm::Value *llvm_sinh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "sinh", {x});
}

// Square root.
llvm::Value *llvm_sqrt(llvm_state &s, llvm::Value *x)
{
    return llvm_math_intr(s, "llvm.sqrt",
#if defined(HEYOKA_HAVE_REAL128)
                          "sqrtq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_sqrt",
#endif
                          {x});
}

// Tangent.
llvm::Value *llvm_tan(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tan", {x});
}

// Hyperbolic tangent.
llvm::Value *llvm_tanh(llvm_state &s, llvm::Value *x)
{
    return llvm_math_cmath(s, "tanh", {x});
}

// Exponentiation.
llvm::Value *llvm_pow(llvm_state &s, llvm::Value *x, llvm::Value *y)
{
    return llvm_math_intr(s, "llvm.pow",
#if defined(HEYOKA_HAVE_REAL128)
                          "powq",
#endif
#if defined(HEYOKA_HAVE_REAL)
                          "mpfr_pow",
#endif
                          {x, y});
}

namespace
{

#if defined(HEYOKA_HAVE_REAL128)

// Helper to create a wrapper for sincosq() from libquadmath which returns the two values packed in an array.
llvm::Function *create_sincosq_wrapper(llvm_state &s)
{
    constexpr auto fname = "heyoka.sincosq_wrapper";

    // Look the wrapper up in the module.
    auto &md = s.module();
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        // The function was created before, return it.
        return f;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the mppp::real128 type.
    auto *fp_t = to_external_llvm_type<mppp::real128>(ctx, false);

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The function returns an array of 2 values.
    auto *arr_t = llvm::ArrayType::get(fp_t, 2);

    // Create the function.
    auto *ft = llvm::FunctionType::get(arr_t, {fp_t}, false);
    f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
    assert(f != nullptr);
    // NOTE: these are intended to inform the optimizer that this will be a pure function.
    f->addFnAttr(llvm::Attribute::NoUnwind);
    f->addFnAttr(llvm::Attribute::Speculatable);
    f->addFnAttr(llvm::Attribute::WillReturn);
    // NOTE: we are reading from memory in this function, but only from a local alloca and this is allowed by the
    // semantics of memory(none).
    f->setMemoryEffects(llvm::MemoryEffects::none());
    // NOTE: it is important that we prevent inlining - if the function is inlined, the information about its pureness
    // is lost and CSE won't work. This is a minor pessimisation but it should hopefully have a very small impact.
    f->addFnAttr(llvm::Attribute::NoInline);

    // Fetch the function argument.
    auto *x = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Create storage for the return values.
    auto *sin_ret = bld.CreateAlloca(fp_t);
    auto *cos_ret = bld.CreateAlloca(fp_t);

    // Invoke the quadmath function.
    llvm_invoke_external(s, "sincosq", bld.getVoidTy(), {x, sin_ret, cos_ret},
                         llvm::AttributeList::get(ctx, llvm::AttributeList::FunctionIndex,
                                                  {llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn}));

    // Create the return value.
    llvm::Value *ret = llvm::UndefValue::get(arr_t);
    ret = bld.CreateInsertValue(ret, bld.CreateLoad(fp_t, sin_ret), {0});
    ret = bld.CreateInsertValue(ret, bld.CreateLoad(fp_t, cos_ret), {1});

    // Return it.
    bld.CreateRet(ret);

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return f;
}

#endif

// Helper to create the combined sin/cos scalar wrappers.
void create_combined_sincos_scalar_wrapper(llvm_state &s, const std::string_view sin_or_cos, llvm::Value *v)
{
    assert(v != nullptr);
    assert(sin_or_cos == "sin" || sin_or_cos == "cos");
    auto *scal_t = v->getType()->getScalarType();

    // Crete the function name.
    auto fname = fmt::format("heyoka.combined_{}.{}", sin_or_cos, llvm_type_name(scal_t));

    // Look it up in the module.
    auto &md = s.module();
    if (md.getFunction(fname) != nullptr) {
        // The function was created before, return.
        return;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

#if defined(HEYOKA_HAVE_REAL128)

    // NOTE: we need to handle real128 first because it is classified as a floating-point type in LLVM, and thus the
    // isFloatingPointTy() check below would evaluate to true.
    if (scal_t == to_external_llvm_type<mppp::real128>(ctx, false)) {
        // Ensure that the sincosq wrapper exists.
        auto *wrapper = create_sincosq_wrapper(s);

        // Fetch the current insertion block.
        auto *orig_bb = bld.GetInsertBlock();

        // Create the function.
        auto *ft = llvm::FunctionType::get(scal_t, {scal_t}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
        assert(f != nullptr);
        // NOTE: force inlining, this is essential in order to make sure that the compiler is able to effectively CSE on
        // the array wrapper calls.
        f->addFnAttr(llvm::Attribute::AlwaysInline);

        // Fetch the function argument.
        auto *x = f->args().begin();

        // Create a new basic block to start insertion into.
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

        // Execute the sincosq wrapper.
        auto *arr_res = bld.CreateCall(wrapper, x);

        // Extract the result from the array and return it.
        bld.CreateRet(bld.CreateExtractValue(arr_res, {sin_or_cos == "sin" ? 0u : 1u}));

        // Restore the original insertion block.
        bld.SetInsertPoint(orig_bb);

        return;
    }

#endif

    if (scal_t->isFloatingPointTy()) {
        // Fetch the current insertion block.
        auto *orig_bb = bld.GetInsertBlock();

        // Create the function.
        auto *ft = llvm::FunctionType::get(scal_t, {scal_t}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the function argument.
        auto *x = f->args().begin();

        // Create a new basic block to start insertion into.
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

        // Compute and return.
        bld.CreateRet(sin_or_cos == "sin" ? llvm_sin(s, x) : llvm_cos(s, x));

        // Restore the original insertion block.
        bld.SetInsertPoint(orig_bb);

        return;
    }

#if defined(HEYOKA_HAVE_REAL)

    if (llvm_is_real(scal_t) != 0) {
        // Fetch the current insertion block.
        auto *orig_bb = bld.GetInsertBlock();

        // Create the function.
        auto *ft = llvm::FunctionType::get(scal_t, {scal_t}, false);
        auto *f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
        assert(f != nullptr);
        // NOTE: force inlining, this is essential in order to make sure that the compiler is able to effectively CSE on
        // the wrapper calls.
        f->addFnAttr(llvm::Attribute::AlwaysInline);

        // Fetch the function argument.
        auto *x = f->args().begin();

        // Create a new basic block to start insertion into.
        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

        // Invoke the MPFR sincos primitive.
        const auto sincos_res = llvm_real_sincos(s, x);

        // Extract the result from sincos_res and return it.
        bld.CreateRet(sin_or_cos == "sin" ? sincos_res.first : sincos_res.second);

        // Restore the original insertion block.
        bld.SetInsertPoint(orig_bb);

        return;
    }

#endif

    // LCOV_EXCL_START
    throw std::invalid_argument(fmt::format(
        "Unsupported type for the creation of a combined sincos scalar wrapper: '{}'", llvm_type_name(scal_t)));
    // LCOV_EXCL_STOP
}

} // namespace

// Combined sin/cos variants.
//
// For scalar float/double/long double arguments, these should be equivalent to llvm_sin/cos. For real, real128 and
// vector arguments, these will result in the invocation of combined sincos primitives from which sin and cos results
// will be extracted. Thus, these should only be used in pairs - i.e., when one knows that both the sine and cosine of
// the same argument is needed. Through LLVM's CSE optimisation pass, redundant double calls to the sincos primitives
// will be reduced to a single sincos invocation, providing a performance benefit.
//
// All of this is achieved through an orchestration of wrappers at several levels:
//
// - first of all, we *always* need IR-defined combined sine and cosine scalar wrappers to be present, even if (like in
//   the case of builtin floating-point types) they just reduce to direct intrinsic calls. This is needed because we can
//   then associate vector variants information to the scalar wrappers through the machinery in vector_math.cpp. For
//   real and real128, the scalar wrappers end up explicitly invoking the respective combined sincos primitives, while
//   for the builtin floating-point types we just invoke the intrinsics and we let LLVM fuse them into combined sincos
//   calls;
// - for vector arguments, we register vector variants for the combined sine and cosine scalar wrappers in
//   vector_math.cpp and then in llvm_math_ir_defined() we go through the usual process of either explicitly calling the
//   vector variants or attaching vfabi attributes to the combined sine and cosine scalar wrappers in order to assist
//   the SLP vectorizer.
//
// Note that at this time, SLP vectorization is not able to both vectorize and fuse the sin/cos calls into combined
// sincos invocations. This would require us to:
//
// - explicitly mark as noinline the combined sine and cosine scalar wrappers (inlining defeats the vfabi attributes,
//   which require a function call to be present), but this results in function call overhead and also prevents LLVM
//   from automatically fuse sin/cos calls into sincos calls in the scalar case;
// - append additional optimisation passes in llvm_state to perform further inlining and CSE *after* the SLP vectorizer
//   has run. This is necessary because SLP vectorization exposes further inlining and CSE opportunities which must be
//   exploited in order to achieve full vectorization and fusion, but no inlining or CSE happens in the standard
//   pipeline after SLP vectorization.
//
// For reference, this is the code we used to enable the additional LLVM passes:
//
// PB.registerOptimizerLastEPCallback([](llvm::ModulePassManager &MPM,
//     llvm::OptimizationLevel, llvm::ThinOrFullLTOPhase) {
//     MPM.addPass(llvm::ModuleInlinerPass());
//     llvm::FunctionPassManager FPM;
//     FPM.addPass(llvm::GVNPass());
//     MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
// });
//
// In the future we may investigate this further if needed.
llvm::Value *llvm_combined_sin(llvm_state &s, llvm::Value *x)
{
    create_combined_sincos_scalar_wrapper(s, "sin", x);
    return llvm_math_ir_defined(s, "heyoka.combined_sin", {x});
}

llvm::Value *llvm_combined_cos(llvm_state &s, llvm::Value *x)
{
    create_combined_sincos_scalar_wrapper(s, "cos", x);
    return llvm_math_ir_defined(s, "heyoka.combined_cos", {x});
}

} // namespace detail

HEYOKA_END_NAMESPACE
