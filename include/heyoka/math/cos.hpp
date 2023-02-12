// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_COS_HPP
#define HEYOKA_MATH_COS_HPP

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

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

class HEYOKA_DLL_PUBLIC cos_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    cos_impl();
    explicit cos_impl(expression);

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    [[nodiscard]] long double eval_ldbl(const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    [[nodiscard]] mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &) const;
#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;
    [[nodiscard]] double eval_num_dbl(const std::vector<double> &) const;
    [[nodiscard]] double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) &&;

    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression cos(expression);

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::cos_impl)

#endif
