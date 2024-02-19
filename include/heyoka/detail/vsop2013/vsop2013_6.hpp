// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_6_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_6_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr unsigned long vsop2013_6_1_sizes[21]
    = {32628ul, 24830ul, 15900ul, 9096ul, 4829ul, 2365ul, 1119ul, 529ul, 227ul, 79ul, 15ul,
       12ul,    12ul,    12ul,    12ul,   12ul,   12ul,   12ul,   12ul,  12ul,  12ul};
inline constexpr unsigned long vsop2013_6_2_sizes[19]
    = {29199ul, 21382ul, 13845ul, 8134ul, 4400ul, 2239ul, 1076ul, 498ul, 213ul, 87ul,
       28ul,    16ul,    13ul,    12ul,   12ul,   12ul,   11ul,   7ul,   4ul};
inline constexpr unsigned long vsop2013_6_3_sizes[11]
    = {27989ul, 19656ul, 12379ul, 7214ul, 3815ul, 1835ul, 876ul, 398ul, 167ul, 53ul, 5ul};
inline constexpr unsigned long vsop2013_6_4_sizes[11]
    = {27622ul, 19702ul, 12513ul, 7262ul, 3848ul, 1846ul, 877ul, 401ul, 167ul, 51ul, 3ul};
inline constexpr unsigned long vsop2013_6_5_sizes[9] = {6331ul, 3920ul, 2125ul, 1096ul, 570ul, 268ul, 113ul, 35ul, 2ul};
inline constexpr unsigned long vsop2013_6_6_sizes[9] = {6262ul, 3948ul, 2159ul, 1110ul, 564ul, 271ul, 114ul, 32ul, 1ul};

extern const double *const vsop2013_6_1_data[21];
extern const double *const vsop2013_6_2_data[19];
extern const double *const vsop2013_6_3_data[11];
extern const double *const vsop2013_6_4_data[11];
extern const double *const vsop2013_6_5_data[9];
extern const double *const vsop2013_6_6_data[9];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
