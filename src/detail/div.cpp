// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/div.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

div_impl::div_impl() : div_impl(0_dbl, 1_dbl) {}

div_impl::div_impl(expression a, expression b) : func_base("div", {std::move(a), std::move(b)}) {}

llvm::Value *div_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                 llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                 bool high_accuracy) const
{
    assert(args().size() == 2u);

    return llvm_eval_helper(
        [&s](const auto &args, bool) {
            assert(args.size() == 2u);
            return llvm_fdiv(s, args[0], args[1]);
        },
        *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *div_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                           bool high_accuracy) const
{
    assert(args().size() == 2u);

    return llvm_c_eval_func_helper(
        "div",
        [&s](const auto &args, bool) {
            assert(args.size() == 2u);
            return llvm_fdiv(s, args[0], args[1]);
        },
        *this, s, fp_t, batch_size, high_accuracy);
}

namespace
{

// Derivative of numpar / numpar.
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Value *taylor_diff_div_impl(llvm_state &s, llvm::Type *fp_t, const U &num0, const V &num1,
                                  const std::vector<llvm::Value *> &, llvm::Value *par_ptr, std::uint32_t,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    if (order == 0u) {
        auto n0 = taylor_codegen_numparam(s, fp_t, num0, par_ptr, batch_size);
        auto n1 = taylor_codegen_numparam(s, fp_t, num1, par_ptr, batch_size);

        return llvm_fdiv(s, n0, n1);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

// Derivative of variable / variable or numpar / variable. These two cases
// are quite similar, so we handle them together.
template <typename U,
          std::enable_if_t<
              std::disjunction_v<std::is_same<U, number>, std::is_same<U, variable>, std::is_same<U, param>>, int>
          = 0>
llvm::Value *taylor_diff_div_impl(llvm_state &s, llvm::Type *fp_t, const U &nv, const variable &var1,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    // Fetch the index of var1.
    const auto u_idx1 = uname_to_index(var1.name());

    if (order == 0u) {
        // Special casing for zero order.
        auto numerator = [&]() -> llvm::Value * {
            if constexpr (std::is_same_v<U, number> || std::is_same_v<U, param>) {
                return taylor_codegen_numparam(s, fp_t, nv, par_ptr, batch_size);
            } else {
                return taylor_fetch_diff(arr, uname_to_index(nv.name()), 0, n_uvars);
            }
        }();

        return llvm_fdiv(s, numerator, taylor_fetch_diff(arr, u_idx1, 0, n_uvars));
    }

    // NOTE: iteration in the [1, order] range
    // (i.e., order inclusive).
    std::vector<llvm::Value *> sum;
    for (std::uint32_t j = 1; j <= order; ++j) {
        auto *v0 = taylor_fetch_diff(arr, idx, order - j, n_uvars);
        auto *v1 = taylor_fetch_diff(arr, u_idx1, j, n_uvars);

        // Add v0*v1 to the sum.
        sum.push_back(llvm_fmul(s, v0, v1));
    }

    // Init the return value as the result of the sum.
    auto *ret_acc = pairwise_sum(s, sum);

    // Load the divisor for the quotient formula.
    // This is the zero-th order derivative of var1.
    auto *div = taylor_fetch_diff(arr, u_idx1, 0, n_uvars);

    if constexpr (std::is_same_v<U, number> || std::is_same_v<U, param>) {
        // nv is a number/param. Negate the accumulator
        // and divide it by the divisor.
        return llvm_fdiv(s, llvm_fneg(s, ret_acc), div);
    } else {
        // nv is a variable. We need to fetch its
        // derivative of order 'order' from the array of derivatives.
        auto diff_nv_v = taylor_fetch_diff(arr, uname_to_index(nv.name()), order, n_uvars);

        // Produce the result: (diff_nv_v - ret_acc) / div.
        return llvm_fdiv(s, llvm_fsub(s, diff_nv_v, ret_acc), div);
    }
}

// Derivative of variable / numpar.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Value *taylor_diff_div_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, const U &num,
                                  const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, std::uint32_t n_uvars,
                                  std::uint32_t order, std::uint32_t, std::uint32_t batch_size)
{
    auto *ret = taylor_fetch_diff(arr, uname_to_index(var.name()), order, n_uvars);
    auto *div = taylor_codegen_numparam(s, fp_t, num, par_ptr, batch_size);

    return llvm_fdiv(s, ret, div);
}

// All the other cases.
// LCOV_EXCL_START
template <typename V1, typename V2, std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Value *taylor_diff_div_impl(llvm_state &, llvm::Type *, const V1 &, const V2 &,
                                  const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t)
{
    throw std::invalid_argument(
        "An invalid argument type was encountered while trying to build the Taylor derivative of div()");
}
// LCOV_EXCL_STOP

} // namespace

llvm::Value *div_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool) const
{
    assert(args().size() == 2u);

    // LCOV_EXCL_START
    if (!deps.empty()) {
        throw std::invalid_argument(fmt::format("The vector of hidden dependencies in the Taylor diff for a division "
                                                "should be empty, but instead it has a size of {}",
                                                deps.size()));
    }
    // LCOV_EXCL_STOP

    return std::visit(
        [&](const auto &v1, const auto &v2) {
            return taylor_diff_div_impl(s, fp_t, v1, v2, arr, par_ptr, n_uvars, order, idx, batch_size);
        },
        args()[0].value(), args()[1].value());
}

namespace
{

// Derivative of numpar / numpar.
template <typename U, typename V, std::enable_if_t<std::conjunction_v<is_num_param<U>, is_num_param<V>>, int> = 0>
llvm::Function *taylor_c_diff_func_div_impl(llvm_state &s, llvm::Type *fp_t, const U &num0, const V &num1,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return taylor_c_diff_func_numpar(
        s, fp_t, n_uvars, batch_size, "div", 0,
        [&s](const auto &args) {
            // LCOV_EXCL_START
            assert(args.size() == 2u);
            assert(args[0] != nullptr);
            assert(args[1] != nullptr);
            // LCOV_EXCL_STOP

            return llvm_fdiv(s, args[0], args[1]);
        },
        num0, num1);
}

// Derivative of var / numpar.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_div_impl(llvm_state &s, llvm::Type *fp_t, const variable &var, const U &n,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "div", n_uvars, batch_size, {var, n});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto var_idx = f->args().begin() + 5;
        auto num = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Load the derivative.
        auto ret = taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, var_idx);

        // Create the return value.
        builder.CreateRet(llvm_fdiv(s, ret, taylor_c_diff_numparam_codegen(s, fp_t, n, num, par_ptr, batch_size)));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Derivative of numpar / var.
template <typename U, std::enable_if_t<is_num_param_v<U>, int> = 0>
llvm::Function *taylor_c_diff_func_div_impl(llvm_state &s, llvm::Type *fp_t, const U &n, const variable &var,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "div", n_uvars, batch_size, {n, var});
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Fetch the necessary function arguments.
        // NOTE: we don't need the number argument because
        // we only need its derivative of order n >= 1,
        // which is always zero.
        auto ord = f->args().begin();
        auto u_idx = f->args().begin() + 1;
        auto diff_ptr = f->args().begin() + 2;
        auto par_ptr = f->args().begin() + 3;
        auto num = f->args().begin() + 5;
        auto var_idx = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the return value.
        auto *retval = builder.CreateAlloca(val_t);

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);

        llvm_if_then_else(
            s, builder.CreateICmpEQ(ord, builder.getInt32(0)),
            [&]() {
                // For order zero, run the codegen.
                auto num_vec = taylor_c_diff_numparam_codegen(s, fp_t, n, num, par_ptr, batch_size);
                auto ret = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx);

                builder.CreateStore(llvm_fdiv(s, num_vec, ret), retval);
            },
            [&]() {
                // Init the accumulator.
                builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

                // Run the loop.
                llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
                    auto cj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx);
                    auto a_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
                    builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, cj, a_nj)), acc);
                });

                // Negate the loop summation.
                auto *ret = llvm_fneg(s, builder.CreateLoad(val_t, acc));

                // Divide and return.
                builder.CreateStore(
                    llvm_fdiv(s, ret, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx)),
                    retval);
            });

        // Return the result.
        builder.CreateRet(builder.CreateLoad(val_t, retval));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// Derivative of var / var.
llvm::Function *taylor_c_diff_func_div_impl(llvm_state &s, llvm::Type *fp_t, const variable &var0, const variable &var1,
                                            std::uint32_t n_uvars, std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the function name and arguments.
    const auto na_pair = taylor_c_diff_func_name_args(context, fp_t, "div", n_uvars, batch_size, {var0, var1});
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

        // Fetch the necessary function arguments.
        auto *ord = f->args().begin();
        auto *u_idx = f->args().begin() + 1;
        auto *diff_ptr = f->args().begin() + 2;
        auto *var_idx0 = f->args().begin() + 5;
        auto *var_idx1 = f->args().begin() + 6;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Create the accumulator.
        auto *acc = builder.CreateAlloca(val_t);
        builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size), acc);

        // Run the loop.
        llvm_loop_u32(s, builder.getInt32(1), builder.CreateAdd(ord, builder.getInt32(1)), [&](llvm::Value *j) {
            auto *cj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, j, var_idx1);
            auto *a_nj = taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.CreateSub(ord, j), u_idx);
            builder.CreateStore(llvm_fadd(s, builder.CreateLoad(val_t, acc), llvm_fmul(s, cj, a_nj)), acc);
        });

        auto *ret = llvm_fsub(s, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, ord, var_idx0),
                              builder.CreateLoad(val_t, acc));

        // Divide and return.
        builder.CreateRet(
            llvm_fdiv(s, ret, taylor_c_load_diff(s, val_t, diff_ptr, n_uvars, builder.getInt32(0), var_idx1)));

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    }

    return f;
}

// All the other cases.
// LCOV_EXCL_START
template <typename V1, typename V2, std::enable_if_t<!std::conjunction_v<is_num_param<V1>, is_num_param<V2>>, int> = 0>
llvm::Function *taylor_c_diff_func_div_impl(llvm_state &, llvm::Type *, const V1 &, const V2 &, std::uint32_t,
                                            std::uint32_t)
{
    throw std::invalid_argument("An invalid argument type was encountered while trying to build the Taylor derivative "
                                "of div() in compact mode");
}
// LCOV_EXCL_STOP

} // namespace

llvm::Function *div_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                             std::uint32_t batch_size, bool) const
{
    return std::visit([&](const auto &v1,
                          const auto &v2) { return taylor_c_diff_func_div_impl(s, fp_t, v1, v2, n_uvars, batch_size); },
                      args()[0].value(), args()[1].value());
}

expression div(expression a, expression b)
{
    return expression{func{div_impl(std::move(a), std::move(b))}};
}

} // namespace detail

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::div_impl)
