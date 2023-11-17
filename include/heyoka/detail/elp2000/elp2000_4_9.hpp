// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ELP2000_ELP_4_9_HPP
#define HEYOKA_DETAIL_ELP2000_ELP_4_9_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const std::int8_t elp2000_idx_4[347][5];
extern const double elp2000_phi_A_4[347][2];

extern const std::int8_t elp2000_idx_5[316][5];
extern const double elp2000_phi_A_5[316][2];

extern const std::int8_t elp2000_idx_6[237][5];
extern const double elp2000_phi_A_6[237][2];

extern const std::int8_t elp2000_idx_7[14][5];
extern const double elp2000_phi_A_7[14][2];

extern const std::int8_t elp2000_idx_8[11][5];
extern const double elp2000_phi_A_8[11][2];

extern const std::int8_t elp2000_idx_9[8][5];
extern const double elp2000_phi_A_9[8][2];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
