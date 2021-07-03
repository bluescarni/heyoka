// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
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
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

sum_impl::sum_impl() : sum_impl(std::vector<expression>{}) {}

sum_impl::sum_impl(std::vector<expression> v) : func_base("sum", std::move(v))
{
    if (std::any_of(args().begin(), args().end(), [](const expression &ex) {
            return std::holds_alternative<param>(ex.value()) || std::holds_alternative<number>(ex.value());
        })) {
        throw std::invalid_argument("The sum() function cannot accept numbers or parameters as input arguments");
    }
}

llvm::Value *sum_impl::taylor_diff_dbl(llvm_state &, const std::vector<std::uint32_t> &,
                                       const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                       std::uint32_t, std::uint32_t, std::uint32_t) const
{
    throw;
}

llvm::Value *sum_impl::taylor_diff_ldbl(llvm_state &, const std::vector<std::uint32_t> &,
                                        const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                        std::uint32_t, std::uint32_t, std::uint32_t) const
{
    throw;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *sum_impl::taylor_diff_f128(llvm_state &, const std::vector<std::uint32_t> &,
                                        const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                                        std::uint32_t, std::uint32_t, std::uint32_t) const
{
    throw;
}

#endif

namespace
{

template <typename T>
llvm::Function *sum_taylor_c_diff_func_impl(llvm_state &s, const sum_impl &sf, std::uint32_t n_uvars,
                                            std::uint32_t batch_size)
{
    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the floating-point type.
    auto val_t = to_llvm_vector_type<T>(context, batch_size);

    // Get the vector of indices of the u variables.
    std::vector<std::uint32_t> v_idx;
    for (const auto &ex : sf.args()) {
        v_idx.push_back(uname_to_index(std::get<variable>(ex.value()).name()));
    }

    // Determine if the vector is in the form:
    // [start, start + stride, start + 2*stride, ...]
    std::uint32_t stride = 0;
    bool is_consecutive = true;

    if (v_idx.size() >= 2u && v_idx[1] > v_idx[0]) {
        // Infer the stride.
        stride = v_idx[1] - v_idx[0];

        for (decltype(v_idx.size()) i = 2; i < v_idx.size(); ++i) {
            if (v_idx[i] <= v_idx[i - 1u] || v_idx[i] - v_idx[i - 1u] != stride) {
                is_consecutive = false;
                break;
            }
        }
    } else {
        is_consecutive = false;
    }

    std::cout << "Consecutive: {}, start={}, stride={}\n"_format(is_consecutive, v_idx[0], stride);

    // Get the function name.
    const auto fname = is_consecutive ? "heyoka_taylor_diff_sum_{}_consec_{}_{}_{}_n_uvars_{}"_format(
                           taylor_mangle_suffix(val_t), v_idx[0], stride, v_idx.back(), n_uvars)
                                      : "heyoka_taylor_diff_sum_{}_{}_n_uvars_{}"_format(taylor_mangle_suffix(val_t),
                                                                                         sf.args().size(), n_uvars);

    // The function arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array,
    // - par ptr,
    // - time ptr,
    // - indices of the variable arguments.
    std::vector<llvm::Type *> fargs{
        llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context), llvm::PointerType::getUnqual(val_t),
        llvm::PointerType::getUnqual(to_llvm_type<T>(context)), llvm::PointerType::getUnqual(to_llvm_type<T>(context))};
    fargs.insert(fargs.end(), boost::numeric_cast<decltype(fargs.size())>(sf.args().size()),
                 llvm::Type::getInt32Ty(context));

    // Try to see if we already created the function.
    auto f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);
        f->addFnAttr(llvm::Attribute::AlwaysInline);

        // Fetch the necessary function arguments.
        auto order = f->args().begin();
        auto diff_arr = f->args().begin() + 2;
        auto vars = f->args().begin() + 5;

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        if (is_consecutive) {
            // Create the return value.
            auto ret = builder.CreateAlloca(val_t);
            builder.CreateStore(vector_splat(builder, codegen<T>(s, number{0.}), batch_size), ret);

            llvm_loop_u32(
                s, builder.getInt32(v_idx[0]), builder.getInt32(v_idx.back()),
                [&](llvm::Value *cur_idx) {
                    auto old_val = builder.CreateLoad(ret);
                    auto new_val
                        = builder.CreateFAdd(old_val, taylor_c_load_diff(s, diff_arr, n_uvars, order, cur_idx));

                    builder.CreateStore(new_val, ret);
                },
                [&](llvm::Value *cur_idx) { return builder.CreateAdd(cur_idx, builder.getInt32(stride)); });

            builder.CreateRet(builder.CreateLoad(ret));
        } else {
            // Create the return value.
            auto ret = vector_splat(builder, codegen<T>(s, number{0.}), batch_size);

            for (decltype(sf.args().size()) i = 0; i < sf.args().size(); ++i) {
                ret = builder.CreateFAdd(ret, taylor_c_load_diff(s, diff_arr, n_uvars, order, vars + i));
            }

            builder.CreateRet(ret);
        }

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
            throw std::invalid_argument(
                "Inconsistent function signature for the Taylor derivative of sum() in compact mode detected");
        }
    }

    return f;
}

} // namespace

llvm::Function *sum_impl::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return sum_taylor_c_diff_func_impl<double>(s, *this, n_uvars, batch_size);
}

llvm::Function *sum_impl::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return sum_taylor_c_diff_func_impl<long double>(s, *this, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *sum_impl::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    return sum_taylor_c_diff_func_impl<mppp::real128>(s, *this, n_uvars, batch_size);
}

#endif

} // namespace detail

expression sum(std::vector<expression> args)
{
    return expression{func{detail::sum_impl{std::move(args)}}};
}

} // namespace heyoka
