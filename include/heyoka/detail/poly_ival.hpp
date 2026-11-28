// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_POLY_IVAL_HPP
#define HEYOKA_DETAIL_POLY_IVAL_HPP

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to add a function to exclude the presence of a root for a polynomial over the range [0, h] (resp., [h, 0] when
// h < 0).
HEYOKA_DLL_PUBLIC llvm::Function *llvm_add_poly_root_exclusion_check(llvm_state &, const std::string &, llvm::Type *,
                                                                     std::uint32_t, std::uint32_t);

// Helper implementing polynomial Taylor shifting by 1 in interval arithmetic.
[[nodiscard]] HEYOKA_DLL_PUBLIC std::vector<llvm::Value *> llvm_ival_poly_shift_1(const llvm_ival_guard &,
                                                                                  const std::vector<llvm::Value *> &);

// Helper implementing polynomial rescaling in interval arithmetic.
[[nodiscard]] HEYOKA_DLL_PUBLIC std::vector<llvm::Value *>
llvm_ival_poly_rescale(const llvm_ival_guard &, const std::vector<llvm::Value *> &, llvm::Value *);

// Helper to produce an interval [lo, hi] of floating-point type T guaranteed to include the binomial coefficient of the
// inputs.
template <typename T>
[[nodiscard]] std::pair<T, T> make_ival_binomial_cf_like(std::uint32_t, std::uint32_t, const T &);

// Helper to convert an input polynomial from the monomial basis into the interval Bernstein form.
[[nodiscard]] HEYOKA_DLL_PUBLIC std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ival_poly_bernstein(const llvm_ival_guard &, const std::vector<llvm::Value *> &, llvm::Value *);

// Helper to count the number of sign changes in a list of interval coefficients.
template <typename T>
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> ival_poly_count_sign_changes(std::span<const T>);

// Helper to split the input Bernstein polynomial at the midpoint of its interval.
[[nodiscard]] HEYOKA_DLL_PUBLIC std::pair<std::vector<llvm::Value *>, std::vector<llvm::Value *>>
llvm_ival_poly_de_casteljau_half(const llvm_ival_guard &, const std::vector<llvm::Value *> &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
