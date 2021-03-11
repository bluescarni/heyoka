// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_FWD_DECL_HPP
#define HEYOKA_DETAIL_FWD_DECL_HPP

namespace heyoka
{

// Fwd declaration of heyoka classes.
class expression;
class variable;
class number;
class binary_operator;
class func;
class param;

class llvm_state;

namespace detail
{

template <typename>
class taylor_adaptive_impl;

template <typename>
struct nt_event;

} // namespace detail

} // namespace heyoka

#endif
