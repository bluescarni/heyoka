// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

time_impl::time_impl() : func_base("time", {}) {}

void time_impl::to_stream(std::ostream &os) const
{
    os << 't';
}

std::vector<expression> time_impl::gradient() const
{
    assert(args().empty());
    return {};
}

llvm::Value *time_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &, llvm::Value *,
                                  llvm::Value *time_ptr, llvm::Value *, std::uint32_t batch_size, bool) const
{
    return ext_load_vector_from_memory(s, fp_t, time_ptr, batch_size);
}

// NOTE: there's some repetition with llvm_c_eval_func_helper here.
llvm::Function *time_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size, bool) const
{
    // LCOV_EXCL_START
    assert(batch_size > 0u);
    assert(args().empty());
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = llvm_c_eval_func_name_args(context, fp_t, "time", batch_size, args());
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Fetch the necessary arguments (only time needed).
        auto *t_ptr = f->args().begin() + 3;

        // Load the time value(s).
        auto *ret = ext_load_vector_from_memory(s, fp_t, t_ptr, batch_size);

        // Return it.
        builder.CreateRet(ret);

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(fmt::format(
                "Inconsistent function signature for the evaluation of {}() in compact mode detected", "time"));
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

namespace
{

llvm::Value *time_taylor_diff_impl(llvm_state &s, llvm::Type *fp_t, llvm::Value *time_ptr, std::uint32_t order,
                                   std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // NOTE: no need for normalisation of the derivative,
    // as the only nonzero retvals are for orders 0 and 1
    // for which the normalised derivatives coincide with
    // the non-normalised derivatives.
    switch (order) {
        case 0u:
            return ext_load_vector_from_memory(s, fp_t, time_ptr, batch_size);
        case 1u:
            return vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size);
        default:
            return vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

} // namespace

llvm::Value *time_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &,
                                    const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *time_ptr,
                                    std::uint32_t, std::uint32_t order, std::uint32_t, std::uint32_t batch_size,
                                    bool) const
{
    return time_taylor_diff_impl(s, fp_t, time_ptr, order, batch_size);
}

namespace
{

// NOTE: perhaps later on this can become a generic implementation
// for nullary functions, in the same mold as
// taylor_c_diff_func_numpar().
llvm::Function *taylor_c_diff_time_impl(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                        std::uint32_t batch_size)
{
    auto &module = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "time", n_uvars, batch_size, {});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = module.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &module);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *t_ptr = f->args().begin() + 4;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // NOTE: no need for normalisation of the derivative,
        // as the only nonzero retvals are for orders 0 and 1
        // for which the normalised derivatives coincide with
        // the non-normalised derivatives.
        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // If the order is zero, return the time itself.
                builder.CreateStore(ext_load_vector_from_memory(s, fp_t, t_ptr, batch_size), retval);
            },
            [&]() {
                llvm_if_then_else(
                    s, builder.CreateICmpEQ(ord, builder.getInt32(1)),
                    [&]() {
                        // If the order is one, return 1.
                        builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{1.}), batch_size),
                                            retval);
                    },
                    [&]() {
                        // If order > 1, return zero.
                        builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size),
                                            retval);
                    });
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
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of time() in compact mode detected");
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace

llvm::Function *time_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                              std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_time_impl(s, fp_t, n_uvars, batch_size);
}

bool time_impl::is_time_dependent() const
{
    return true;
}

} // namespace detail

const expression time{func{detail::time_impl{}}};

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::time_impl)
