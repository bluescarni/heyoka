// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_MPFR_HELPERS_HPP
#define HEYOKA_DETAIL_MPFR_HELPERS_HPP

#include <type_traits>
#include <utility>

#include <mp++/real.hpp>

namespace heyoka::detail
{

// Handy typedefs.
using real_prec_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_prec);
using real_sign_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_sign);
using real_exp_t = decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_exp);
using real_limb_t = std::remove_pointer_t<decltype(std::declval<mppp::mpfr_struct_t>()._mpfr_exp)>;

} // namespace heyoka::detail

#endif
