// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#if defined(HEYOKA_HAVE_REAL)

#include <type_traits>

#include <mp++/real.hpp>

#include <heyoka/detail/mpfr_helpers.hpp>

namespace heyoka::detail
{

static_assert(sizeof(mppp::real) == sizeof(mppp::mpfr_struct_t));
static_assert(alignof(mppp::real) == alignof(mppp::mpfr_struct_t));
static_assert(mppp::real_prec_min() > 0);
static_assert(std::is_signed_v<real_sign_t>);
static_assert(std::is_signed_v<real_exp_t>);
static_assert(std::is_signed_v<real_exp_t>);
static_assert(std::is_signed_v<real_rnd_t>);

} // namespace heyoka::detail

#endif
