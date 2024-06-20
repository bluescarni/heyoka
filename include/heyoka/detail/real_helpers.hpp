// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_REAL_HELPERS_HPP
#define HEYOKA_DETAIL_REAL_HELPERS_HPP

#include <string>
#include <type_traits>
#include <utility>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>

#include <mp++/real.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// The integral type corresponding to the mpfr_rnd_t enum.
using real_rnd_t = std::underlying_type_t<mpfr_rnd_t>;

mpfr_prec_t llvm_is_real(llvm::Type *);

llvm::Value *llvm_real_fneg(llvm_state &, llvm ::Value *);
llvm::Function *real_nary_op(llvm_state &, llvm::Type *, const std::string &, unsigned);
std::pair<llvm::Value *, llvm::Value *> llvm_real_sincos(llvm_state &, llvm::Value *);
llvm::Value *llvm_real_fcmp_ult(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_uge(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_oge(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_ole(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_olt(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_ogt(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_oeq(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fcmp_one(llvm_state &, llvm::Value *, llvm::Value *);
llvm::Value *llvm_real_fnz(llvm_state &, llvm::Value *);
llvm::Value *llvm_real_ui_to_fp(llvm_state &, llvm::Value *, llvm::Type *);
llvm::Value *llvm_real_sgn(llvm_state &, llvm::Value *);

HEYOKA_DLL_PUBLIC mppp::real eps_from_prec(mpfr_prec_t);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
