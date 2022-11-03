// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EVENT_DETECTION_HPP
#define HEYOKA_DETAIL_EVENT_DETECTION_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <type_traits>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka::detail
{

template <typename T>
inline T taylor_deduce_cooldown(T, T)
{
    static_assert(always_false_v<T>, "Unhandled type");
}

template <>
double taylor_deduce_cooldown(double, double);

template <>
long double taylor_deduce_cooldown(long double, long double);

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 taylor_deduce_cooldown(mppp::real128, mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
mppp::real taylor_deduce_cooldown(mppp::real, mppp::real);

#endif

// Machinery to add a fast event exclusion check function to an llvm_state.
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_fex_check(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool = false);

HEYOKA_DLL_PUBLIC llvm::Function *add_poly_translator_1(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t);

// Machinery to add a function that, given an input polynomial of order n represented
// as an array of coefficients:
// - reverses it,
// - translates it by 1,
// - counts the sign changes in the coefficients
//   of the resulting polynomial.
llvm::Function *llvm_add_poly_rtscc(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t);

} // namespace heyoka::detail

#endif
