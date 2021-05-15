// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_KEPE_HPP
#define HEYOKA_MATH_KEPE_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC kepE_impl : public func_base
{
public:
    kepE_impl();
    explicit kepE_impl(expression, expression);

    expression diff(const std::string &) const;

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

HEYOKA_DLL_PUBLIC expression kepE(expression, expression);

HEYOKA_DLL_PUBLIC expression kepE(expression, double);
HEYOKA_DLL_PUBLIC expression kepE(expression, long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC expression kepE(expression, mppp::real128);

#endif

HEYOKA_DLL_PUBLIC expression kepE(double, expression);
HEYOKA_DLL_PUBLIC expression kepE(long double, expression);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC expression kepE(mppp::real128, expression);

#endif

} // namespace heyoka

#endif
