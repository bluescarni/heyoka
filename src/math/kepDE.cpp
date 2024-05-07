// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <initializer_list>
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

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/taylor_common.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepDE.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

kepDE_impl::kepDE_impl() : kepDE_impl(0_dbl, 0_dbl, 0_dbl) {}

kepDE_impl::kepDE_impl(expression s0, expression c0, expression DM)
    : func_base("kepDE", std::vector{std::move(s0), std::move(c0), std::move(DM)})
{
}

template <typename Archive>
void kepDE_impl::serialize(Archive &ar, unsigned)
{
    ar &boost::serialization::base_object<func_base>(*this);
}

std::vector<expression> kepDE_impl::gradient() const
{
    assert(args().size() == 3u);

    const auto &s0 = args()[0];
    const auto &c0 = args()[1];

    const expression DE{func{*this}};

    const auto den = pow(1_dbl + s0 * sin(DE) - c0 * cos(DE), -1_dbl);

    return {(cos(DE) - 1_dbl) * den, sin(DE) * den, den};
}

namespace
{

llvm::Value *kepDE_llvm_eval_impl(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                  const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                                  std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_eval_helper(
        [&s, fp_t, batch_size](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepDE_func = llvm_add_inv_kep_DE(s, fp_t, batch_size);

            return s.builder().CreateCall(kepDE_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

} // namespace

llvm::Value *kepDE_impl::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                                   llvm::Value *par_ptr, llvm::Value *, llvm::Value *stride, std::uint32_t batch_size,
                                   bool high_accuracy) const
{
    return kepDE_llvm_eval_impl(s, fp_t, *this, eval_arr, par_ptr, stride, batch_size, high_accuracy);
}

namespace
{

[[nodiscard]] llvm::Function *kepDE_llvm_c_eval(llvm_state &s, llvm::Type *fp_t, const func_base &fb,
                                                std::uint32_t batch_size, bool high_accuracy)
{
    return llvm_c_eval_func_helper(
        "kepDE",
        [&s, batch_size, fp_t](const std::vector<llvm::Value *> &args, bool) -> llvm::Value * {
            auto *kepDE_func = llvm_add_inv_kep_DE(s, fp_t, batch_size);

            return s.builder().CreateCall(kepDE_func, {args[0], args[1], args[2]});
        },
        fb, s, fp_t, batch_size, high_accuracy);
}

} // namespace

llvm::Function *kepDE_impl::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                             bool high_accuracy) const
{
    return kepDE_llvm_c_eval(s, fp_t, *this, batch_size, high_accuracy);
}

} // namespace detail

// NOTE: constant folding here would need a JIT-compiled version of kepDE().
// Perhaps store the function pointer in a thread_local variable and keep around
// a cache of llvm states to fetch the pointer from?
expression kepDE(expression s0, expression c0, expression DM)
{
    return expression{func{detail::kepDE_impl{std::move(s0), std::move(c0), std::move(DM)}}};
}

#define HEYOKA_DEFINE_KEPDE_OVERLOADS(type)                                                                            \
    expression kepDE(expression s0, type c0, type DM)                                                                  \
    {                                                                                                                  \
        return kepDE(std::move(s0), expression{std::move(c0)}, expression{std::move(DM)});                             \
    }                                                                                                                  \
    expression kepDE(type s0, expression c0, type DM)                                                                  \
    {                                                                                                                  \
        return kepDE(expression{std::move(s0)}, std::move(c0), expression{std::move(DM)});                             \
    }                                                                                                                  \
    expression kepDE(type s0, type c0, expression DM)                                                                  \
    {                                                                                                                  \
        return kepDE(expression{std::move(s0)}, expression{std::move(c0)}, std::move(DM));                             \
    }                                                                                                                  \
    expression kepDE(expression s0, expression c0, type DM)                                                            \
    {                                                                                                                  \
        return kepDE(std::move(s0), std::move(c0), expression{std::move(DM)});                                         \
    }                                                                                                                  \
    expression kepDE(expression s0, type c0, expression DM)                                                            \
    {                                                                                                                  \
        return kepDE(std::move(s0), expression{std::move(c0)}, std::move(DM));                                         \
    }                                                                                                                  \
    expression kepDE(type s0, expression c0, expression DM)                                                            \
    {                                                                                                                  \
        return kepDE(expression{std::move(s0)}, std::move(c0), std::move(DM));                                         \
    }

HEYOKA_DEFINE_KEPDE_OVERLOADS(float)
HEYOKA_DEFINE_KEPDE_OVERLOADS(double)
HEYOKA_DEFINE_KEPDE_OVERLOADS(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DEFINE_KEPDE_OVERLOADS(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DEFINE_KEPDE_OVERLOADS(mppp::real);

#endif

#undef HEYOKA_DEFINE_KEPDE_OVERLOADS

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::kepDE_impl)
