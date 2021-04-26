// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_FWD_DECL_HPP
#define HEYOKA_DETAIL_FWD_DECL_HPP

#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

// Fwd declaration of heyoka classes.
class HEYOKA_DLL_PUBLIC expression;
class HEYOKA_DLL_PUBLIC variable;
class HEYOKA_DLL_PUBLIC number;
class HEYOKA_DLL_PUBLIC binary_operator;
class HEYOKA_DLL_PUBLIC func;
class HEYOKA_DLL_PUBLIC param;

class HEYOKA_DLL_PUBLIC llvm_state;

namespace detail
{

template <typename>
class HEYOKA_DLL_PUBLIC taylor_adaptive_impl;

template <typename>
class HEYOKA_DLL_PUBLIC taylor_adaptive_batch_impl;

template <typename>
class HEYOKA_DLL_PUBLIC nt_event_impl;

template <typename>
class HEYOKA_DLL_PUBLIC t_event_impl;

} // namespace detail

// Enum to represent the direction of an event.
// NOTE: put it here because this is currently shared between
// taylor.hpp and event_detection.hpp.
enum class event_direction { negative = -1, any = 0, positive = 1 };

} // namespace heyoka

#endif
