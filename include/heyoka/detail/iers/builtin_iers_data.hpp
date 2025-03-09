// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP
#define HEYOKA_DETAIL_IERS_BUILTIN_IERS_DATA_HPP

#include <heyoka/config.hpp>
#include <heyoka/model/iers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const iers_data_row init_iers_data[19480];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
