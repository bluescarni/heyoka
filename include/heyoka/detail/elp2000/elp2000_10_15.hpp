// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ELP2000_ELP_10_15_HPP
#define HEYOKA_DETAIL_ELP2000_ELP_10_15_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const std::int8_t elp2000_idx_10[14328][11];
extern const double elp2000_phi_A_10[14328][2];

extern const std::int8_t elp2000_idx_11[5233][11];
extern const double elp2000_phi_A_11[5233][2];

extern const std::int8_t elp2000_idx_12[6631][11];
extern const double elp2000_phi_A_12[6631][2];

extern const std::int8_t elp2000_idx_13[4384][11];
extern const double elp2000_phi_A_13[4384][2];

extern const std::int8_t elp2000_idx_14[833][11];
extern const double elp2000_phi_A_14[833][2];

extern const std::int8_t elp2000_idx_15[1715][11];
extern const double elp2000_phi_A_15[1715][2];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
