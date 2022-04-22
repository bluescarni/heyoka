// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_SUM_SQ_HPP
#define HEYOKA_MATH_SUM_SQ_HPP

#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC sum_sq_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

    template <typename T>
    HEYOKA_DLL_LOCAL expression diff_impl(std::unordered_map<const void *, expression> &, const T &) const;

public:
    sum_sq_impl();
    explicit sum_sq_impl(std::vector<expression>);

    void to_stream(std::ostream &) const;

    expression diff(std::unordered_map<const void *, expression> &, const std::string &) const;
    expression diff(std::unordered_map<const void *, expression> &, const param &) const;

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

    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t, bool, std::uint32_t) const;
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t, bool, std::uint32_t) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t, bool, std::uint32_t) const;
#endif
};

// NOTE: the default split value is a power of two so that the
// internal pairwise sums are rounded up exactly.
inline constexpr std::uint32_t default_sum_sq_split = 64;

} // namespace detail

HEYOKA_DLL_PUBLIC expression sum_sq(std::vector<expression>, std::uint32_t = detail::default_sum_sq_split);

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::sum_sq_impl)

#endif
