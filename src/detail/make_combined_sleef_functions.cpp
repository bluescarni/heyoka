// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <string>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/ModRef.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/vector_math.hpp>
#include <heyoka/llvm_state.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to generate the LLVM function that invokes the combined C++ sleef wrappers.
llvm::Function *make_combined_sleef_array_wrapper(llvm_state &s, const std::string &arr_wrapper_name, llvm::Type *vec_t,
                                                  const std::string &cpp_wrapper_name)
{
    auto &md = s.module();

    // Try to see if we already created the function.
    auto *arr_wrapper_f = md.getFunction(arr_wrapper_name);
    if (arr_wrapper_f != nullptr) {
        return arr_wrapper_f;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The return type is an array of 2 SIMD vectors.
    auto *arr_t = llvm::ArrayType::get(vec_t, 2);
    auto *ft = llvm::FunctionType::get(arr_t, {vec_t}, false);

    // Create the function
    arr_wrapper_f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, arr_wrapper_name, &md);
    assert(arr_wrapper_f != nullptr);
    // NOTE: these are intended to inform the optimizer that this will be a pure function.
    arr_wrapper_f->addFnAttr(llvm::Attribute::NoUnwind);
    arr_wrapper_f->addFnAttr(llvm::Attribute::Speculatable);
    arr_wrapper_f->addFnAttr(llvm::Attribute::WillReturn);
    // NOTE: we are reading from memory in this function, but only from a local alloca and this is allowed by the
    // semantics of memory(none).
    arr_wrapper_f->setMemoryEffects(llvm::MemoryEffects::none());
    // NOTE: it is important that we prevent inlining - if the array wrapper is inlined, the information about its
    // pureness is lost and CSE won't work. This is a minor pessimisation but it should hopefully have a very small
    // impact.
    arr_wrapper_f->addFnAttr(llvm::Attribute::NoInline);

    // Fetch the function argument.
    auto *x = arr_wrapper_f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", arr_wrapper_f));

    // Create the return value.
    auto *retval = bld.CreateAlloca(arr_t);

    // Fetch a pointer to the beginning of retval.
    auto *ptr = bld.CreateInBoundsGEP(arr_t, retval, {bld.getInt32(0), bld.getInt32(0)});

    // Invoke the C++ function.
    llvm_invoke_external(s, cpp_wrapper_name, bld.getVoidTy(), {ptr, x});

    // Return the result.
    bld.CreateRet(bld.CreateLoad(arr_t, retval));

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);

    return arr_wrapper_f;
}

// Helper to generate a combined sleef vector function. Internally it will invoke arr_wrapper_f (the the LLVM function
// that invokes the combined C++ sleef wrapper) and will extract the idx-th component from its return value.
void make_combined_sleef_function(llvm_state &s, const std::string &fname, llvm::Function *arr_wrapper_f,
                                  llvm::Type *vec_t, const std::uint32_t idx)
{
    assert(arr_wrapper_f != nullptr);

    auto &md = s.module();

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);
    if (f != nullptr) {
        return;
    }

    // The function was not created before, do it now.
    auto &bld = s.builder();
    auto &ctx = s.context();

    // Fetch the current insertion block.
    auto *orig_bb = bld.GetInsertBlock();

    // The return type is a SIMD vector.
    auto *ft = llvm::FunctionType::get(vec_t, {vec_t}, false);

    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::PrivateLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the function argument.
    auto *x = f->args().begin();

    // Create a new basic block to start insertion into.
    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Invoke the array wrapper.
    auto *arr_val = bld.CreateCall(arr_wrapper_f, {x});

    // Extract the value.
    auto *ret = bld.CreateExtractValue(arr_val, {boost::numeric_cast<unsigned>(idx)});

    // Return the result.
    bld.CreateRet(ret);

    // Restore the original insertion block.
    bld.SetInsertPoint(orig_bb);
}

} // namespace

// Helper to generate the IR code for the combined sleef vector functions.
void make_combined_sleef_functions(llvm_state &s, const std::string &scalar_base_name,
                                   const std::string &sleef_base_name, const std::string &sleef_tp,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   const std::uint32_t idx, const std::uint32_t width, const std::string &iset)
{
    assert(sleef_tp == "d" || sleef_tp == "f");
    assert(idx == 0u || idx == 1u);
    assert(width > 0u);

    auto &ctx = s.context();

    // Fetch the scalar floating-point type.
    auto *scal_t = (sleef_tp == "f") ? to_external_llvm_type<float>(ctx) : to_external_llvm_type<double>(ctx);

    // Fetch the vector floating-point type.
    auto *vec_t = make_vector_type(scal_t, width);

    // NOTE: iterate to generate both the high-precision and low-precision versions.
    for (const auto *prec_str : {"u10", "u35"}) {
        // Step 1: generate, if necessary, the combined wrapper that returns the two values as an array.
        const auto sleef_fname = fmt::format("Sleef_{}{}{}_{}{}", sleef_base_name, sleef_tp, width, prec_str, iset);
        const auto arr_wrapper_name = fmt::format("heyoka.{}", sleef_fname);
        const auto cpp_wrapper_name = fmt::format("heyoka_{}", sleef_fname);
        auto *arr_wrapper_f = make_combined_sleef_array_wrapper(s, arr_wrapper_name, vec_t, cpp_wrapper_name);

        // Step 2: generate, if necessary, the actual vector variant.
        const auto fname = fmt::format("heyoka.combined_vector_sleef.{}.{}{}_{}{}", scalar_base_name, sleef_tp, width,
                                       prec_str, iset);
        make_combined_sleef_function(s, fname, arr_wrapper_f, vec_t, idx);
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
