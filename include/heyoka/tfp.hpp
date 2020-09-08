// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TFP_HPP
#define HEYOKA_TFP_HPP

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/IR/Value.h>

#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

// LLVM floating-point type for use in the Taylor machinery.
// It represents either a single floating-point operand (normal mode),
// or a pair of floating-point operands (high accuracy mode) for
// the implementation of double-length arithmetics.
// NOTE: the high accuracy algorithms require fast math to be disabled.
using tfp = std::variant<llvm::Value *, std::pair<llvm::Value *, llvm::Value *>>;

HEYOKA_DLL_PUBLIC tfp tfp_add(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_sub(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_mul(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_div(llvm_state &, const tfp &, const tfp &);

HEYOKA_DLL_PUBLIC tfp tfp_neg(llvm_state &, const tfp &);

HEYOKA_DLL_PUBLIC llvm::Value *tfp_cast(llvm_state &, const tfp &);

namespace detail
{

HEYOKA_DLL_PUBLIC tfp taylor_load_derivative(const std::vector<tfp> &, std::uint32_t, std::uint32_t, std::uint32_t);
HEYOKA_DLL_PUBLIC tfp tfp_pairwise_sum(llvm_state &, std::vector<tfp> &);

} // namespace detail

} // namespace heyoka

#endif
