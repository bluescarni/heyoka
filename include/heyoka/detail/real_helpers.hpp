// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>

#include <mp++/real.hpp>

namespace heyoka::detail
{

// Handy typedefs.
using real_prec_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_prec);
using real_sign_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_sign);
using real_exp_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_exp);
using real_limb_t = std::remove_pointer_t<decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_exp)>;

// NOTE: mpfr_rnd_t is part of the MPFR API, so technically this introduces
// a direct dependency on MPFR. On the other hand, perhaps we can guarantee that
// including real.hpp includes transitively mpfr.h and leave it at that?
using real_rnd_t = std::underlying_type_t<mpfr_rnd_t>;

real_prec_t llvm_is_real(llvm::Type *);

llvm::Function *real_binary_op(llvm_state &, llvm::Type *, const std::string &, const std::string &);

} // namespace heyoka::detail

#endif
