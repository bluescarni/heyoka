// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_TYPE_TRAITS_HPP
#define HEYOKA_DETAIL_TYPE_TRAITS_HPP

#include <type_traits>

namespace heyoka::detail
{

template <typename T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename, typename...>
inline constexpr bool always_false_v = false;

} // namespace heyoka::detail

#endif
