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

// Machinery to add a fast event exclusion check function to an llvm_state.
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_fex_check_dbl(llvm_state &, std::uint32_t, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_fex_check_ldbl(llvm_state &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_fex_check_f128(llvm_state &, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Function *llvm_add_fex_check(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return llvm_add_fex_check_dbl(s, n, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return llvm_add_fex_check_ldbl(s, n, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return llvm_add_fex_check_f128(s, n, batch_size);
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }
}

// Machinery to add a function that, given an input polynomial of order n represented
// as an array of coefficients:
// - reverses it,
// - translates it by 1,
// - counts the sign changes in the coefficients
//   of the resulting polynomial.
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_poly_rtscc_dbl(llvm_state &, std::uint32_t, std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_poly_rtscc_ldbl(llvm_state &, std::uint32_t, std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_poly_rtscc_f128(llvm_state &, std::uint32_t, std::uint32_t);

#endif

template <typename T>
inline llvm::Function *llvm_add_poly_rtscc(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return llvm_add_poly_rtscc_dbl(s, n, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return llvm_add_poly_rtscc_ldbl(s, n, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return llvm_add_poly_rtscc_f128(s, n, batch_size);
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka::detail

#endif
