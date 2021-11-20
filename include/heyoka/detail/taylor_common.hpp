// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <boost/numeric/conversion/cast.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

namespace heyoka::detail
{

// Helper to implement the function for the differentiation of
// 'func(number)' in compact mode. The function will always return zero,
// unless the order is 0 (in which case it will return the result of the codegen).
template <typename T, typename F, typename U>
inline llvm::Function *taylor_c_diff_func_unary_num_det(llvm_state &s, const F &fn, const U &n, std::uint32_t n_uvars,
                                                        std::uint32_t batch_size, const std::string &name,
                                                        std::uint32_t n_deps = 0)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args<T>(context, name, n_uvars, batch_size, {n}, n_deps);
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

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
        auto par_ptr = f->args().begin() + 3;
        auto num = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto retval = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, run the codegen.
                builder.CreateStore(
                    codegen_from_values<T>(s, fn, {taylor_c_diff_numparam_codegen(s, n, num, par_ptr, batch_size)}),
                    retval);
            },
            [&]() {
                // Otherwise, return zero.
                builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the derivative function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument("Inconsistent function signature for the Taylor derivative of " + name
                                        + "() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace heyoka::detail

#endif
