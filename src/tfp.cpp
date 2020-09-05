// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

struct fm_disabler {
    llvm_state &m_s;
    llvm::FastMathFlags m_orig_fmf;

    explicit fm_disabler(llvm_state &s) : m_s(s), m_orig_fmf(m_s.builder().getFastMathFlags())
    {
        llvm::FastMathFlags new_fmf;
        m_s.builder().setFastMathFlags(new_fmf);
    }
    ~fm_disabler()
    {
        m_s.builder().setFastMathFlags(m_orig_fmf);
    }
};

} // namespace

} // namespace detail

tfp tfp_add(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFAdd(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                detail::fm_disabler fmd(s);

                auto x = builder.CreateFAdd(a.first, b.first);
                auto z = builder.CreateFSub(x, a.first);
                auto y = builder.CreateFAdd(builder.CreateFSub(a.first, builder.CreateFSub(x, z)),
                                            builder.CreateFSub(b.first, z));

                return std::pair{x, builder.CreateFAdd(y, builder.CreateFAdd(a.second, b.second))};
            } else {
                throw std::invalid_argument(
                    "Invalid combination of argument in tfp_add(): the input tfp variants must contain the same types");
            }
        },
        x, y);
}

tfp tfp_neg(llvm_state &s, const tfp &x)
{
    return std::visit(
        [&s](const auto &a) -> tfp {
            auto &builder = s.builder();

            if constexpr (std::is_same_v<detail::uncvref_t<decltype(a)>, llvm::Value *>) {
                return builder.CreateFNeg(a);
            } else {
                detail::fm_disabler fmd(s);

                return std::pair{builder.CreateFNeg(a.first), builder.CreateFNeg(a.second)};
            }
        },
        x);
}

tfp tfp_sub(llvm_state &s, const tfp &x, const tfp &y)
{
    return tfp_add(s, x, tfp_neg(s, y));
}

namespace detail
{

namespace
{

llvm::Value *tfp_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    assert(x->getType() == y->getType());
    assert(x->getType() == z->getType());

    const auto intrinsic_ID = llvm::Function::lookupIntrinsicID("llvm.fma");
    if (intrinsic_ID == 0) {
        throw std::invalid_argument("Cannot fetch the ID of the intrinsic 'llvm.fma'");
    }

    // NOTE: for generic intrinsics to work, we need to specify
    // the desired argument types. See:
    // https://stackoverflow.com/questions/11985247/llvm-insert-intrinsic-function-cos
    // And the docs of the getDeclaration() function.
    const std::vector<llvm::Type *> arg_types(1u, x->getType());

    auto callee_f = llvm::Intrinsic::getDeclaration(&s.module(), intrinsic_ID, arg_types);

    if (!callee_f) {
        throw std::invalid_argument("Error getting the declaration of the intrinsic 'llvm.fma'");
    }

    if (!callee_f->isDeclaration()) {
        // It does not make sense to have a definition of a builtin.
        throw std::invalid_argument("The intrinsic 'llvm.fma' must be only declared, not defined");
    }

    // Create the function call.
    auto r = s.builder().CreateCall(callee_f, {x, y, z});
    assert(r != nullptr);

    return r;
}

// NOTE: this assumes fast math has already been disabled.
auto tfp_eft_prod(llvm_state &s, llvm::Value *a, llvm::Value *b)
{
    auto &builder = s.builder();

    auto x = builder.CreateFMul(a, b);
    auto y = detail::tfp_fma(s, a, b, builder.CreateFNeg(x));

    return std::pair{x, y};
}

} // namespace

} // namespace detail

tfp tfp_mul(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFMul(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                detail::fm_disabler fmd(s);

                auto [x, y] = detail::tfp_eft_prod(s, a.first, b.first);

                return std::pair{x, detail::tfp_fma(s, b.first, a.second,
                                                    detail::tfp_fma(s, a.first, b.second,
                                                                    detail::tfp_fma(s, a.second, b.second, y)))};
            } else {
                throw std::invalid_argument(
                    "Invalid combination of argument in tfp_mul(): the input tfp variants must contain the same types");
            }
        },
        x, y);
}

tfp tfp_div(llvm_state &s, const tfp &x, const tfp &y)
{
    return std::visit(
        [&s](const auto &a, const auto &b) -> tfp {
            using t1 = detail::uncvref_t<decltype(a)>;
            using t2 = detail::uncvref_t<decltype(b)>;

            auto &builder = s.builder();

            if constexpr (std::is_same_v<t1, llvm::Value *> && std::is_same_v<t2, llvm::Value *>) {
                return builder.CreateFDiv(a, b);
            } else if constexpr (std::is_same_v<
                                     t1,
                                     std::pair<llvm::Value *,
                                               llvm::Value
                                                   *>> && std::is_same_v<t2, std::pair<llvm::Value *, llvm::Value *>>) {
                detail::fm_disabler fmd(s);

                auto c = builder.CreateFDiv(a.first, b.first);
                auto [u, uu] = detail::tfp_eft_prod(s, c, b.first);
                auto cc = builder.CreateFDiv(
                    builder.CreateFSub(builder.CreateFAdd(builder.CreateFAdd(builder.CreateFNeg(uu), a.second),
                                                          detail::tfp_fma(s, builder.CreateFNeg(c), b.second, a.first)),
                                       u),
                    b.first);

                return std::pair{c, cc};
            } else {
                throw std::invalid_argument(
                    "Invalid combination of argument in tfp_div(): the input tfp variants must contain the same types");
            }
        },
        x, y);
}

} // namespace heyoka
