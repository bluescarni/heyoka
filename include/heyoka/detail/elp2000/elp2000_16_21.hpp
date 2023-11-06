// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ELP2000_ELP_16_21_HPP
#define HEYOKA_DETAIL_ELP2000_ELP_16_21_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const std::int8_t elp2000_idx_16[170][11];
extern const double elp2000_phi_A_16[170][2];

extern const std::int8_t elp2000_idx_17[150][11];
extern const double elp2000_phi_A_17[150][2];

extern const std::int8_t elp2000_idx_18[114][11];
extern const double elp2000_phi_A_18[114][2];

extern const std::int8_t elp2000_idx_19[226][11];
extern const double elp2000_phi_A_19[226][2];

extern const std::int8_t elp2000_idx_20[188][11];
extern const double elp2000_phi_A_20[188][2];

extern const std::int8_t elp2000_idx_21[169][11];
extern const double elp2000_phi_A_21[169][2];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
