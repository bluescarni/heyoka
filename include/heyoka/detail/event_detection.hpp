// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EVENT_DETECTION_HPP
#define HEYOKA_DETAIL_EVENT_DETECTION_HPP

#include <cstddef>
#include <optional>
#include <span>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
T taylor_deduce_cooldown(T, T);

template <typename T>
[[nodiscard]] std::optional<int> ed_poly_eval_sign_ival(std::span<const T>, const T &, std::size_t offset = 0,
                                                        std::size_t stride = 1);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
