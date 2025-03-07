// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_TIME_CONVERSIONS_HPP
#define HEYOKA_MODEL_TIME_CONVERSIONS_HPP

#include <array>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> fk5j2000_icrs(const std::array<expression, 3> &);
[[nodiscard]] HEYOKA_DLL_PUBLIC std::array<expression, 3> icrs_fk5j2000(const std::array<expression, 3> &);

} // namespace model

HEYOKA_END_NAMESPACE

#endif
