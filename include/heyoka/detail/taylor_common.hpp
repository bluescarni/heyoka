// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TAYLOR_COMMON_HPP
#define HEYOKA_DETAIL_TAYLOR_COMMON_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka::detail
{

// Helper to implement the function for the differentiation of
// 'func(number)' in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of the codegen).
template <typename T>
inline llvm::Function *taylor_c_diff_func_unary_num(llvm_state &s, const function &func, std::uint32_t batch_size,
                                                    const std::string &fname, const std::string &desc)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - number argument.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context),
                                    llvm::PointerType::getUnqual(val_t), to_llvm_type<T>(context)};

    // Try to see if we already created the function.
    auto f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto ord = f->args().begin();
        auto num = f->args().begin() + 3;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                builder.CreateStore(codegen_from_values<T>(s, func, {vector_splat(builder, num, batch_size)}), retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of " + desc
                                        + " in compact mode detected");
        }
    }

    return f;
}

} // namespace heyoka::detail

#endif
