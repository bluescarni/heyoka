// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_EXP_HPP
#define HEYOKA_MATH_EXP_HPP

#include <cstdint>
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

class HEYOKA_DLL_PUBLIC exp_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    exp_impl();
    explicit exp_impl(expression);

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const;
    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const;
#endif

    std::vector<expression> gradient() const;

    double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    long double eval_ldbl(const std::unordered_map<std::string, long double> &, const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                            const std::vector<mppp::real128> &) const;
#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;
    double eval_num_dbl(const std::vector<double> &) const;
    double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const;

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

HEYOKA_DLL_PUBLIC expression exp(expression);

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::exp_impl)

#endif
