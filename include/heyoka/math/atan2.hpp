// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_ATAN2_HPP
#define HEYOKA_MATH_ATAN2_HPP

#include <heyoka/config.hpp>

#include <cstdint>
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

class HEYOKA_DLL_PUBLIC atan2_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    atan2_impl();
    explicit atan2_impl(expression, expression);

    [[nodiscard]] std::vector<expression> gradient() const;

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

HEYOKA_DLL_PUBLIC expression atan2(expression, expression);

HEYOKA_DLL_PUBLIC expression atan2(expression, float);
HEYOKA_DLL_PUBLIC expression atan2(expression, double);
HEYOKA_DLL_PUBLIC expression atan2(expression, long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC expression atan2(expression, mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DLL_PUBLIC expression atan2(expression, mppp::real);

#endif

HEYOKA_DLL_PUBLIC expression atan2(float, expression);
HEYOKA_DLL_PUBLIC expression atan2(double, expression);
HEYOKA_DLL_PUBLIC expression atan2(long double, expression);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC expression atan2(mppp::real128, expression);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DLL_PUBLIC expression atan2(mppp::real, expression);

#endif

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::atan2_impl)

#endif
