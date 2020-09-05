// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TFP_HPP
#define HEYOKA_TFP_HPP

#include <utility>
#include <variant>

#include <llvm/IR/Value.h>

#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

// LLVM floating-point type for use in the Taylor machinery.
using tfp = std::variant<llvm::Value *, std::pair<llvm::Value *, llvm::Value *>>;

HEYOKA_DLL_PUBLIC tfp tfp_add(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_neg(llvm_state &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_sub(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_mul(llvm_state &, const tfp &, const tfp &);
HEYOKA_DLL_PUBLIC tfp tfp_div(llvm_state &, const tfp &, const tfp &);

} // namespace heyoka

#endif
