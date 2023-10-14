// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

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

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

kepF_impl::kepF_impl() : kepF_impl(0_dbl, 0_dbl, 0_dbl) {}

kepF_impl::kepF_impl(expression h, expression k, expression lam)
    : func_base("kepF", std::vector{std::move(h), std::move(k), std::move(lam)})
{
}

template <typename Archive>
void kepF_impl::serialize(Archive &ar, unsigned)
{
    ar &boost::serialization::base_object<func_base>(*this);
}

template <typename T>
expression kepF_impl::diff_impl(funcptr_map<expression> &func_map, const T &s) const
{
    assert(args().size() == 3u);

    const auto &h = args()[0];
    const auto &k = args()[1];
    const auto &lam = args()[2];

    const expression F{func{*this}};

    return (detail::diff(func_map, k, s) * sin(F) - detail::diff(func_map, h, s) * cos(F)
            + detail::diff(func_map, lam, s))
           / (1_dbl - h * sin(F) - k * cos(F));
}

expression kepF_impl::diff(funcptr_map<expression> &func_map, const std::string &s) const
{
    return diff_impl(func_map, s);
}

expression kepF_impl::diff(funcptr_map<expression> &func_map, const param &p) const
{
    return diff_impl(func_map, p);
}

namespace
{

llvm::Value *kepF_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                 const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                                 std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_eval_helper(
        [&s, fp_t, batch_size](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepF_func = llvm_add_inv_kep_F(s, fp_t, batch_size);

            return s.builder().CreateCall(kepF_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *kepF_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                  llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                  bool high_accuracy) const
{
    return kepF_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *kepF_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                               std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "kepF",
        [&s, batch_size, fp_t](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepF_func = llvm_add_inv_kep_F(s, fp_t, batch_size);

            return s.builder().CreateCall(kepF_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, batch_size, high_accuracy);
}

} // namespace

llvm::Function *kepF_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                            bool high_accuracy) const
{
    return kepF_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

} // namespace detail

// NOTE: constant folding here would need a JIT-compiled version of kepF().
// Perhaps store the function pointer in a thread_local variable and keep around
// a cache of llvm states to fetch the pointer from?
expression kepF(expression h, expression k, expression lam)
{
    return expression{func{detail::kepF_impl{std::move(h), std::move(k), std::move(lam)}}};
}

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::kepF_impl)
