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
#include <string>
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

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
