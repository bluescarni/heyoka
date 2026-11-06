// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_SUPPORTED_FP_TYPE_HPP
#define HEYOKA_SUPPORTED_FP_TYPE_HPP

#include <concepts>

#include <mp++/fwd.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/type_traits.hpp>

HEYOKA_BEGIN_NAMESPACE

// NOTE: concepts to detect the floating-point types supported in scalar and batch computations.
template <typename T>
concept supported_fp_type = detail::is_supported_fp<T>::value;

template <typename T>
concept supported_batch_fp_type = supported_fp_type<T> && !std::same_as<T, mppp::real>;

HEYOKA_END_NAMESPACE

#endif
