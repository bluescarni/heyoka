// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
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
#include <heyoka/math/select.hpp>
#include <heyoka/s11n.hpp>

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

} // namespace detail

expression select(expression cond, expression t, expression f)
{
    return expression{func{detail::select_impl{std::move(cond), std::move(t), std::move(f)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::select_impl)
