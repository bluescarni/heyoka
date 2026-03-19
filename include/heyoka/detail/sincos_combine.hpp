// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_SINCOS_COMBINE_HPP
#define HEYOKA_DETAIL_SINCOS_COMBINE_HPP

#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

void sincos_combine_cfunc(std::vector<expression> &);

void sincos_combine_taylor(taylor_dc_t &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
