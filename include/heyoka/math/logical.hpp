// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_LOGICAL_HPP
#define HEYOKA_MATH_LOGICAL_HPP

#include <cstdint>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

class HEYOKA_DLL_PUBLIC logical_and_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    logical_and_impl();
    explicit logical_and_impl(std::vector<expression>);

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

class HEYOKA_DLL_PUBLIC logical_or_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    logical_or_impl();
    explicit logical_or_impl(std::vector<expression>);

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

HEYOKA_DLL_PUBLIC expression logical_and(std::vector<expression>);
HEYOKA_DLL_PUBLIC expression logical_or(std::vector<expression>);

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::logical_and_impl)
HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::logical_or_impl)

#endif
