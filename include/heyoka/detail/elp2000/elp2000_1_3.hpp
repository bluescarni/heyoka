// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ELP2000_ELP_1_3_HPP
#define HEYOKA_DETAIL_ELP2000_ELP_1_3_HPP

#include <cstdint>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const std::int8_t elp2000_idx_1[1023][4];
extern const double elp2000_A_B_1[1023][6];

extern const std::int8_t elp2000_idx_2[918][4];
extern const double elp2000_A_B_2[918][6];

extern const std::int8_t elp2000_idx_3[704][4];
extern const double elp2000_A_B_3[704][6];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
