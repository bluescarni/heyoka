// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
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

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

select_impl::select_impl() : select_impl(0_dbl, 0_dbl, 0_dbl) {}

select_impl::select_impl(expression cond, expression t, expression f)
    : func_base("select", {std::move(cond), std::move(t), std::move(f)})
{
}

std::vector<expression> select_impl::gradient() const
{
    return {0_dbl, select(args()[0], 1_dbl, 0_dbl), select(args()[0], 0_dbl, 1_dbl)};
}

namespace
{

llvm::Value *select_eval_impl(llvm_state &s, const std::vector<llvm::Value *> &args)
{
    assert(args.size() == 3u);

    return s.builder().CreateSelect(llvm_fnz(s, args[0]), args[1], args[2]);
}

} // namespace

llvm::Value *select_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                    llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                    bool high_accuracy) const
{
    return llvm_eval_helper([&s](const std::vector<llvm::Value *> &args, bool) { return select_eval_impl(s, args); },
                            *this, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *select_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                              bool high_accuracy) const
{
    return llvm_c_eval_func_helper(
        "select", [&s](const std::vector<llvm::Value *> &args, bool) { return select_eval_impl(s, args); }, *this, s,
        fp_t, batch_size, high_accuracy);
}

llvm::Value *select_impl::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                                      const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *,
                                      std::uint32_t n_uvars, std::uint32_t order, std::uint32_t,
                                      std::uint32_t batch_size, bool) const
{
    assert(args().size() == 3u);
    assert(deps.empty());

    std::vector<llvm::Value *> tmp;
    tmp.reserve(3);

    // For the condition, we always use the order-0 values.
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
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the "
                                            "Taylor derivative of select()");
                // LCOV_EXCL_STOP
            }
        },
        args()[0].value()));

    // For the branches, we use the order-n derivatives.
    for (decltype(args().size()) i = 1; i < args().size(); ++i) {
        tmp.push_back(std::visit(
            [&]<typename T>(const T &v) -> llvm::Value * {
                if constexpr (std::same_as<T, variable>) {
                    // Variable.
                    return taylor_fetch_diff(arr, uname_to_index(v.name()), order, n_uvars);
                } else if constexpr (is_num_param_v<T>) {
                    // Number/param.
                    if (order == 0u) {
                        return taylor_codegen_numparam(s, fp_t, v, par_ptr, batch_size);
                    } else {
                        return vector_splat(s.builder(), llvm_codegen(s, fp_t, number{0.}), batch_size);
                    }
                } else {
                    // LCOV_EXCL_START
                    throw std::invalid_argument("An invalid argument type was encountered while trying to build the "
                                                "Taylor derivative of select()");
                    // LCOV_EXCL_STOP
                }
            },
            args()[i].value()));
    }

    return select_eval_impl(s, tmp);
}

llvm::Function *select_impl::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                                std::uint32_t batch_size, bool) const
{
    assert(args().size() == 3u);

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Build the vector of arguments needed to determine the function name.
    std::vector<std::variant<variable, number, param>> nm_args;
    nm_args.reserve(static_cast<decltype(nm_args.size())>(args().size()));
    for (const auto &arg : args()) {
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
    const auto [fname, fargs] = taylor_c_diff_func_name_args(context, fp_t, "select", n_uvars, batch_size, nm_args);

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

    std::vector<llvm::Value *> vals;
    vals.reserve(3);

    // For the condition, we always use the order-0 values.
    vals.push_back(std::visit(
        [&]<typename T>(const T &v) -> llvm::Value * {
            if constexpr (std::same_as<T, variable>) {
                return taylor_c_load_diff(s, val_t, diff_arr, n_uvars, builder.getInt32(0), operands);
            } else if constexpr (is_num_param_v<T>) {
                return taylor_c_diff_numparam_codegen(s, fp_t, v, operands, par_ptr, batch_size);
            } else {
                // LCOV_EXCL_START
                throw std::invalid_argument("An invalid argument type was encountered while trying to build the "
                                            "Taylor derivative of select()");
                // LCOV_EXCL_STOP
            }
        },
        args()[0].value()));

    // For the branches, we use the order-n derivatives.
    for (decltype(args().size()) i = 1; i < args().size(); ++i) {
        vals.push_back(std::visit(
            [&]<typename T>(const T &v) -> llvm::Value * {
                if constexpr (std::same_as<T, variable>) {
                    return taylor_c_load_diff(s, val_t, diff_arr, n_uvars, order, operands + i);
                } else if constexpr (is_num_param_v<T>) {
                    // Create the return value.
                    auto *retval = builder.CreateAlloca(val_t);

                    llvm_if_then_else(
                        s, builder.CreateICmpEQ(order, builder.getInt32(0)),
                        [&]() {
                            // If the order is zero, run the codegen.
                            builder.CreateStore(
                                taylor_c_diff_numparam_codegen(s, fp_t, v, operands + i, par_ptr, batch_size), retval);
                        },
                        [&]() {
                            // Otherwise, return zero.
                            builder.CreateStore(vector_splat(builder, llvm_codegen(s, fp_t, number{0.}), batch_size),
                                                retval);
                        });

                    return builder.CreateLoad(val_t, retval);
                } else {
                    // LCOV_EXCL_START
                    throw std::invalid_argument("An invalid argument type was encountered while trying to build the "
                                                "Taylor derivative of select()");
                    // LCOV_EXCL_STOP
                }
            },
            args()[i].value()));
    }

    builder.CreateRet(select_eval_impl(s, vals));

    // Verify.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    return f;
}

} // namespace detail

expression select(expression cond, expression t, expression f)
{
    return expression{func{detail::select_impl{std::move(cond), std::move(t), std::move(f)}}};
}

// NOTE: this macro was copy-pasted from the kepDE() overloads, hence
// the weird (but inconsequential) naming of the function arguments.
#define HEYOKA_DEFINE_SELECT_OVERLOADS(type)                                                                           \
    expression select(expression s0, type c0, type DM)                                                                 \
    {                                                                                                                  \
        return select(std::move(s0), expression{std::move(c0)}, expression{std::move(DM)});                            \
    }                                                                                                                  \
    expression select(type s0, expression c0, type DM)                                                                 \
    {                                                                                                                  \
        return select(expression{std::move(s0)}, std::move(c0), expression{std::move(DM)});                            \
    }                                                                                                                  \
    expression select(type s0, type c0, expression DM)                                                                 \
    {                                                                                                                  \
        return select(expression{std::move(s0)}, expression{std::move(c0)}, std::move(DM));                            \
    }                                                                                                                  \
    expression select(expression s0, expression c0, type DM)                                                           \
    {                                                                                                                  \
        return select(std::move(s0), std::move(c0), expression{std::move(DM)});                                        \
    }                                                                                                                  \
    expression select(expression s0, type c0, expression DM)                                                           \
    {                                                                                                                  \
        return select(std::move(s0), expression{std::move(c0)}, std::move(DM));                                        \
    }                                                                                                                  \
    expression select(type s0, expression c0, expression DM)                                                           \
    {                                                                                                                  \
        return select(expression{std::move(s0)}, std::move(c0), std::move(DM));                                        \
    }

HEYOKA_DEFINE_SELECT_OVERLOADS(float)
HEYOKA_DEFINE_SELECT_OVERLOADS(double)
HEYOKA_DEFINE_SELECT_OVERLOADS(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DEFINE_SELECT_OVERLOADS(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DEFINE_SELECT_OVERLOADS(mppp::real);

#endif

#undef HEYOKA_DEFINE_SELECT_OVERLOADS

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::select_impl)
