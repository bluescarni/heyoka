// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_RELATIONAL_HPP
#define HEYOKA_MATH_RELATIONAL_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <sstream>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

enum class rel_op { eq, neq, lt, gt, lte, gte };

class HEYOKA_DLL_PUBLIC rel_impl : public func_base
{
    rel_op m_op = rel_op::eq;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
        ar & m_op;
    }

public:
    rel_impl();
    explicit rel_impl(rel_op, expression, expression);

    [[nodiscard]] rel_op get_op() const noexcept;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                           const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                           std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool) const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression eq(expression, expression);
HEYOKA_DLL_PUBLIC expression neq(expression, expression);
HEYOKA_DLL_PUBLIC expression lt(expression, expression);
HEYOKA_DLL_PUBLIC expression gt(expression, expression);
HEYOKA_DLL_PUBLIC expression lte(expression, expression);
HEYOKA_DLL_PUBLIC expression gte(expression, expression);

#define HEYOKA_DECLARE_REL_OVERLOADS(type)                                                                             \
    HEYOKA_DLL_PUBLIC expression eq(expression, type);                                                                 \
    HEYOKA_DLL_PUBLIC expression eq(type, expression);                                                                 \
    HEYOKA_DLL_PUBLIC expression neq(expression, type);                                                                \
    HEYOKA_DLL_PUBLIC expression neq(type, expression);                                                                \
    HEYOKA_DLL_PUBLIC expression lt(expression, type);                                                                 \
    HEYOKA_DLL_PUBLIC expression lt(type, expression);                                                                 \
    HEYOKA_DLL_PUBLIC expression gt(expression, type);                                                                 \
    HEYOKA_DLL_PUBLIC expression gt(type, expression);                                                                 \
    HEYOKA_DLL_PUBLIC expression lte(expression, type);                                                                \
    HEYOKA_DLL_PUBLIC expression lte(type, expression);                                                                \
    HEYOKA_DLL_PUBLIC expression gte(expression, type);                                                                \
    HEYOKA_DLL_PUBLIC expression gte(type, expression);

HEYOKA_DECLARE_REL_OVERLOADS(float);
HEYOKA_DECLARE_REL_OVERLOADS(double);
HEYOKA_DECLARE_REL_OVERLOADS(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DECLARE_REL_OVERLOADS(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DECLARE_REL_OVERLOADS(mppp::real);

#endif

#undef HEYOKA_DECLARE_REL_OVERLOADS

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::rel_impl)

#endif
