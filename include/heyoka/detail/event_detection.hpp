// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <string>
#include <utility>
#include <vector>

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

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
inline T taylor_deduce_cooldown(T, T)
{
    static_assert(always_false_v<T>, "Unhandled type");
}

template <>
float taylor_deduce_cooldown(float, float);

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

// Helper to add a fast event exclusion check function to an llvm_state.
HEYOKA_DLL_PUBLIC llvm::Function *llvm_ed_add_fex_check(llvm_state &, const std::string &, llvm::Type *, std::uint32_t,
                                                        std::uint32_t);

// Helper implementing polynomial Taylor shifting by 1 in interval arithmetic.
[[nodiscard]] HEYOKA_DLL_PUBLIC std::vector<llvm::Value *>
llvm_ed_ival_poly_shift_1(const llvm_ival_guard &, const std::vector<llvm::Value *> &);

// Helper to produce an interval [lo, hi] of floating-point type T guaranteed to include the binomial coefficient of the
// inputs.
template <typename T>
std::pair<T, T> ed_make_ival_binomial_cf_like(std::uint32_t, std::uint32_t, const T &);

HEYOKA_DLL_PUBLIC llvm::Function *add_poly_translator_1(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t);

// Machinery to add a function that, given an input polynomial of order n represented
// as an array of coefficients:
// - reverses it,
// - translates it by 1,
// - counts the sign changes in the coefficients
//   of the resulting polynomial.
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_poly_rtscc(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
