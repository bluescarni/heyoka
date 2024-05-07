// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_KEPDE_HPP
#define HEYOKA_MATH_KEPDE_HPP

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

// NOTE: this is the trivariate function DE=DE(s0, c0, DM) implicitly defined
// by the trascendental equation:
//
// DM = DE + s0 * (1 - cos(DE)) - c0 * sin(DE)
//
// Where s0**2 + c0**2 < 1.
class HEYOKA_DLL_PUBLIC kepDE_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    HEYOKA_DLL_LOCAL void serialize(Archive &, unsigned);

public:
    kepDE_impl();
    explicit kepDE_impl(expression, expression, expression);

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression kepDE(expression, expression, expression);

#define HEYOKA_DECLARE_KEPDE_OVERLOADS(type)                                                                           \
    HEYOKA_DLL_PUBLIC expression kepDE(expression, type, type);                                                        \
    HEYOKA_DLL_PUBLIC expression kepDE(type, expression, type);                                                        \
    HEYOKA_DLL_PUBLIC expression kepDE(type, type, expression);                                                        \
    HEYOKA_DLL_PUBLIC expression kepDE(expression, expression, type);                                                  \
    HEYOKA_DLL_PUBLIC expression kepDE(expression, type, expression);                                                  \
    HEYOKA_DLL_PUBLIC expression kepDE(type, expression, expression)

HEYOKA_DECLARE_KEPDE_OVERLOADS(float);
HEYOKA_DECLARE_KEPDE_OVERLOADS(double);
HEYOKA_DECLARE_KEPDE_OVERLOADS(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DECLARE_KEPDE_OVERLOADS(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DECLARE_KEPDE_OVERLOADS(mppp::real);

#endif

#undef HEYOKA_DECLARE_KEPDE_OVERLOADS

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::kepDE_impl)

#endif
