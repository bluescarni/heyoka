// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/tfp.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// RAII helper to temporarily disable all fast math flags that might
// be set in an LLVM builder. On destruction, the original fast math
// flags will be restored.
struct fm_disabler {
    llvm_state &m_s;
    llvm::FastMathFlags m_orig_fmf;

    explicit fm_disabler(llvm_state &s) : m_s(s), m_orig_fmf(m_s.builder().getFastMathFlags())
    {
        // Set the new flags (all fast math options are disabled).
        m_s.builder().setFastMathFlags(llvm::FastMathFlags{});
    }
    ~fm_disabler()
    {
        // Restore the original flags.
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

                // Knuth's TwoSum algorithm.
                auto x = builder.CreateFAdd(a.first, b.first);
                auto z = builder.CreateFSub(x, a.first);
                auto y = builder.CreateFAdd(builder.CreateFSub(a.first, builder.CreateFSub(x, z)),
                                            builder.CreateFSub(b.first, z));

                // Double-length addition without normalisation.
                return std::pair{x, builder.CreateFAdd(y, builder.CreateFAdd(a.second, b.second))};
            } else {
                throw std::invalid_argument(
                    "Invalid combination of argument in tfp_add(): the input tfp variants must contain the same types");
            }
        },
        x, y);
}

namespace detail
{

namespace
{

// Helper to negate a tfp.
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

} // namespace

} // namespace detail

// x + y = x + (-y).
tfp tfp_sub(llvm_state &s, const tfp &x, const tfp &y)
{
    return tfp_add(s, x, detail::tfp_neg(s, y));
}

namespace detail
{

namespace
{

// Helper to invoke the fma builtin used in the implementation
// of tfp_mul().
llvm::Value *tfp_fma(llvm_state &s, llvm::Value *x, llvm::Value *y, llvm::Value *z)
{
    assert(x->getType() == y->getType());
    assert(x->getType() == z->getType());

    return llvm_invoke_intrinsic(s, "llvm.fma", {x->getType()}, {x, y, z});
}

// TwoProductFMA algorithm of Ogita et al.
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

                // mul2 algorithm of Dekker without normalisation.
                // TODO check if this can be simplified, Dekker actually
                // does not do the multiplication of the errors of a and b.
                // But maybe in our case it's needed because we don't normalise?
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

                // div2 algorithm of Dekker without normalisation.
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
