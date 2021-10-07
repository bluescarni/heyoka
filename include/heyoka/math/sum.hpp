// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_SUM_HPP
#define HEYOKA_MATH_SUM_HPP

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

class HEYOKA_DLL_PUBLIC sum_impl : public func_base
{
public:
    sum_impl();
    explicit sum_impl(std::vector<expression>);

    llvm::Value *taylor_diff_dbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                 llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                 std::uint32_t, bool) const;
    llvm::Value *taylor_diff_ldbl(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *taylor_diff_f128(llvm_state &, const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                  llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, bool) const;
#endif
    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
#endif
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression sum(std::vector<expression>);

template <typename... Args>
inline auto sum(Args &&...args) -> decltype(sum(std::vector<expression>{expression(std::forward<Args>(args))...}))
{
    return sum(std::vector<expression>{expression(std::forward<Args>(args))...});
}

} // namespace heyoka

#endif
