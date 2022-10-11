// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_ATAN_HPP
#define HEYOKA_MATH_ATAN_HPP

#include <cstdint>
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

class HEYOKA_DLL_PUBLIC atan_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    atan_impl();
    explicit atan_impl(expression);

    expression diff(std::unordered_map<const void *, expression> &, const std::string &) const;
    expression diff(std::unordered_map<const void *, expression> &, const param &) const;

    double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    long double eval_ldbl(const std::unordered_map<std::string, long double> &, const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                            const std::vector<mppp::real128> &) const;
#endif

    [[nodiscard]] llvm::Value *llvm_eval_dbl(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *,
                                             llvm::Value *, std::uint32_t, bool) const;
    [[nodiscard]] llvm::Value *llvm_eval_ldbl(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *,
                                              llvm::Value *, std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] llvm::Value *llvm_eval_f128(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *,
                                              llvm::Value *, std::uint32_t, bool) const;
#endif

    [[nodiscard]] llvm::Function *llvm_c_eval_func_dbl(llvm_state &, std::uint32_t, bool) const;
    [[nodiscard]] llvm::Function *llvm_c_eval_func_ldbl(llvm_state &, std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] llvm::Function *llvm_c_eval_func_f128(llvm_state &, std::uint32_t, bool) const;
#endif

    taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) &&;

    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t, bool) const;
#endif
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression atan(expression);

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::atan_impl)

#endif
