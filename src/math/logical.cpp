// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/logical.hpp>
#include <heyoka/s11n.hpp>

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
