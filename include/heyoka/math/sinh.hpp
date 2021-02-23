// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_SINH_HPP
#define HEYOKA_MATH_SINH_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC sinh_impl : public func_base
{
public:
    sinh_impl();
    explicit sinh_impl(expression);

    expression diff(const std::string &) const;

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const;
    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const;
#endif

    std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
    taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &) &&;
    llvm::Value *taylor_diff_dbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                 llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                 std::uint32_t) const;
    llvm::Value *taylor_diff_ldbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *taylor_diff_f128(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t) const;
#endif
    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t) const;
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t) const;
#endif
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression sinh(expression);

} // namespace heyoka

#endif
