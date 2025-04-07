// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_EGM2008_HPP
#define HEYOKA_DETAIL_EGM2008_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr std::uint32_t egm2008_max_degree = 80;

extern const double egm2008_CS[3318][2];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
