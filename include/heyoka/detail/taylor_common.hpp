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
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/function.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

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

// Derivative of pow(variable, number).
// NOTE: this is currently shared with the sqrt implementation. Eventually, we will
// have a dedicated implementation for sqrt and this can be moved back to pow.cpp.
template <typename T>
inline llvm::Value *taylor_diff_pow_impl_det(llvm_state &s, const variable &var, const number &num,
                                             const std::vector<llvm::Value *> &arr, std::uint32_t n_uvars,
                                             std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    // NOTE: pairwise summation requires order 1 at least.
    // NOTE: also not much use in allowing zero-order
    // derivatives, which in general might complicate
    // the implementation.
    if (order == 0u) {
        throw std::invalid_argument(
            "Cannot compute the Taylor derivative of order 0 of pow() (the order must be at least one)");
    }

    auto &builder = s.builder();

    // Fetch the index of the variable.
    const auto u_idx = uname_to_index(var.name());

    // NOTE: iteration in the [0, order) range
    // (i.e., order *not* included).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 0; j < order; ++j) {
        auto v0 = taylor_fetch_diff(arr, u_idx, order - j, n_uvars);
        auto v1 = taylor_fetch_diff(arr, idx, j, n_uvars);

        // Compute the scalar factor: order * num - j * (num + 1).
        auto scal_f = vector_splat(builder,
                                   codegen<T>(s, number(static_cast<T>(order)) * num
                                                     - number(static_cast<T>(j)) * (num + number(static_cast<T>(1)))),
                                   batch_size);

        // Add scal_f*v0*v1 to the sum.
        sum.push_back(builder.CreateFMul(scal_f, builder.CreateFMul(v0, v1)));
    }

    // Init the return value as the result of the sum.
    auto ret_acc = pairwise_sum(builder, sum);

    // Compute the final divisor: order * (zero-th derivative of u_idx).
    auto ord_f = vector_splat(builder, codegen<T>(s, number(static_cast<T>(order))), batch_size);
    auto b0 = taylor_fetch_diff(arr, u_idx, 0, n_uvars);
    auto div = builder.CreateFMul(ord_f, b0);

    // Compute and return the result: ret_acc / div.
    return builder.CreateFDiv(ret_acc, div);
}

} // namespace heyoka::detail

#endif
