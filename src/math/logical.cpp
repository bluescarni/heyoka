// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/logical.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

logical_and_impl::logical_and_impl() : logical_and_impl({1_dbl}) {}

logical_and_impl::logical_and_impl(std::vector<expression> args) : func_base("logical_and", std::move(args))
{
    assert(!this->args().empty());
}

std::vector<expression> logical_and_impl::gradient() const
{
    return std::vector<expression>(this->args().size(), 0_dbl);
}

namespace
{

llvm::Value *logical_andor_eval_impl(llvm_state &s, const std::vector<llvm::Value *> &args, bool is_and)
{
    assert(!args.empty());

    auto &builder = s.builder();

    // Convert the values in args into booleans.
    std::vector<llvm::Value *> tmp;
    tmp.reserve(args.size());
    std::ranges::transform(args, std::back_inserter(tmp), [&s](auto *v) { return llvm_fnz(s, v); });

    // Run a pairwise AND/OR reduction on the transformed values.
    auto *ret = pairwise_reduce(tmp, [&builder, is_and](auto *a, auto *b) {
        if (is_and) {
            return builder.CreateLogicalAnd(a, b);
        } else {
            return builder.CreateLogicalOr(a, b);
        }
    });

    // Convert back to floating-point.
    return llvm_ui_to_fp(s, ret, args[0]->getType());
}

} // namespace

llvm::Value *logical_and_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                         llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride,
                                         std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper(
        [&s](const std::vector<llvm::Value *> &args, bool) { return logical_andor_eval_impl(s, args, true); }, *this, s,
        fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *logical_and_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                                   bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        "logical_and",
        [&s](const std::vector<llvm::Value *> &args, bool) { return logical_andor_eval_impl(s, args, true); }, *this, s,
        fp_t, batch_size, high_accuracy);
}

llvm::Value *logical_and_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                           const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                           std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                           std::uint32_t batch_size, bool) const
{
    assert(!args().empty());
    assert(deps.empty());

    // NOTE: we need to do something only at differentiation order 0.
    if (order == 0u) {
        std::vector<llvm::Value *> tmp;
        tmp.reserve(static_cast<decltype(tmp.size())>(args().size()));

        for (const auto &cur_arg : args()) {
            tmp.push_back(std::visit(
                [&]<typename T>(const T &v) -> llvm::Value * {
                    if constexpr (std::same_as<T, variable>) {
                        // Variable.
                        return taylor_fetch_diff(arr, uname_to_index(v.name()), 0, n_uvars);
                    } else if constexpr (is_num_param_v<T>) {
                        // Number/param.
                        return taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a logical_and()");
                        // LCOV_EXCL_STOP
                    }
                },
                cur_arg.value()));
        }

        return logical_andor_eval_impl(s, tmp, true);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

namespace
{

llvm::Function *taylor_c_diff_func_logical_andor_impl(const func_base &fb, llvm_state &s, llvm::Type *fp_t,
                                                      std::uint32_t n_uvars, std::uint32_t batch_size, bool is_and)
{
    assert(!fb.args().empty());

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Build the vector of arguments needed to determine the function name.
    std::vector<std::variant<variable, number, param>> nm_args;
    nm_args.reserve(static_cast<decltype(nm_args.size())>(fb.args().size()));
    for (const auto &arg : fb.args()) {
        nm_args.push_back(std::visit(
            []<typename T>(const T &v) -> std::variant<variable, number, param> {
                if constexpr (std::same_as<T, func>) {
                    // LCOV_EXCL_START
                    assert(false);
                    throw;
                    // LCOV_EXCL_STOP
                } else {
                    return v;
                }
            },
            arg.value()));
    }

    // Fetch the function name and arguments.
    const auto [fname, fargs] = taylor_c_diff_func_name_args(context, fp_t, is_and ? "logical_and" : "logical_or",
                                                             n_uvars, batch_size, nm_args);

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f != nullptr) {
        return f;
    }

    // The function was not created before, do it now.

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The return type is val_t.
    auto *ft = llvm::FunctionType::get(val_t, fargs, false);
    // Create the function
    f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
    assert(f != nullptr);

    // Fetch the necessary function arguments.
    auto *order = f->args().begin();
    auto *diff_arr = f->args().begin() + 2;
    auto *par_ptr = f->args().begin() + 3;
    auto *operands = f->args().begin() + 5;

    // Create a new basic block to start insertion into.
    builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

    // Create the return value.
    auto *retval = builder.CreateAlloca(val_t);

    llvm_if_then_else(
        s, builder.CreateICmpEQ(order, builder.getInt32(0)),
        [&]() {
            // For order zero, evaluate the logical operation.
            std::vector<llvm::Value *> vals;
            vals.reserve(2);

            for (decltype(fb.args().size()) i = 0; i < fb.args().size(); ++i) {
                vals.push_back(std::visit(
                    [&]<typename T>(const T &v) -> llvm::Value * {
                        if constexpr (std::same_as<T, variable>) {
                            return taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, operands + i);
                        } else if constexpr (is_num_param_v<T>) {
                            return taylor_c_diff_numparam_codegen(s, fp_t, v, operands + i, par_ptr, batch_size);
                        } else {
                            // LCOV_EXCL_START
                            throw std::invalid_argument(
                                "An invalid argument type was encountered while trying to build the "
                                "Taylor derivative of a logical operation");
                            // LCOV_EXCL_STOP
                        }
                    },
                    fb.args()[i].value()));
            }

            builder.CreateStore(logical_andor_eval_impl(s, vals, is_and), retval);
        },
        [&]() {
            // Otherwise, return zero.
            builder.CreateStore(llvm_constantfp(s, val_t, 0.), retval);
        });

    builder.CreateRet(builder.CreateLoad(val_t, retval));

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

} // namespace

llvm::Function *logical_and_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                     std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_logical_andor_impl(*this, s, fp_t, n_uvars, batch_size, true);
}

logical_or_impl::logical_or_impl() : logical_or_impl({1_dbl}) {}

logical_or_impl::logical_or_impl(std::vector<expression> args) : func_base("logical_or", std::move(args))
{
    assert(!this->args().empty());
}

std::vector<expression> logical_or_impl::gradient() const
{
    return std::vector<expression>(this->args().size(), 0_dbl);
}

llvm::Value *logical_or_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                        llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride,
                                        std::uint32_t batch_size, bool high_accuracy) const
{
    return llvm_eval_helper(
        [&s](const std::vector<llvm::Value *> &args, bool) { return logical_andor_eval_impl(s, args, false); }, *this,
        s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *logical_or_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                                  bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        "logical_or",
        [&s](const std::vector<llvm::Value *> &args, bool) { return logical_andor_eval_impl(s, args, false); }, *this,
        s, fp_t, batch_size, high_accuracy);
}

llvm::Value *logical_or_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                          const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                          std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size, bool) const
{
    assert(!args().empty());
    assert(deps.empty());

    // NOTE: we need to do something only at differentiation order 0.
    if (order == 0u) {
        std::vector<llvm::Value *> tmp;
        tmp.reserve(static_cast<decltype(tmp.size())>(args().size()));

        for (const auto &cur_arg : args()) {
            tmp.push_back(std::visit(
                [&]<typename T>(const T &v) -> llvm::Value * {
                    if constexpr (std::same_as<T, variable>) {
                        // Variable.
                        return taylor_fetch_diff(arr, uname_to_index(v.name()), 0, n_uvars);
                    } else if constexpr (is_num_param_v<T>) {
                        // Number/param.
                        return taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                    } else {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            "An invalid argument type was encountered while trying to build the "
                            "Taylor derivative of a logical_or()");
                        // LCOV_EXCL_STOP
                    }
                },
                cur_arg.value()));
        }

        return logical_andor_eval_impl(s, tmp, false);
    } else {
        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
    }
}

llvm::Function *logical_or_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                    std::uint32_t batch_size, bool) const
{
    return taylor_c_diff_func_logical_andor_impl(*this, s, fp_t, n_uvars, batch_size, false);
}

} // namespace detail

expression logical_and(std::vector<expression> args)
{
    if (args.empty()) {
        return 1_dbl;
    }

    if (args.size() == 1u) {
        return std::move(args[0]);
    }

    return expression{func{detail::logical_and_impl{std::move(args)}}};
}

expression logical_or(std::vector<expression> args)
{
    if (args.empty()) {
        return 0_dbl;
    }

    if (args.size() == 1u) {
        return std::move(args[0]);
    }

    return expression{func{detail::logical_or_impl{std::move(args)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::logical_and_impl)

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::logical_or_impl)
