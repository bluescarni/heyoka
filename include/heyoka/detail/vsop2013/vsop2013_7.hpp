// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_7_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_7_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr unsigned long vsop2013_7_1_sizes[11]
    = {32758ul, 23029ul, 13612ul, 7391ul, 3732ul, 1685ul, 701ul, 230ul, 114ul, 79ul, 61ul};
inline constexpr unsigned long vsop2013_7_2_sizes[11]
    = {28917ul, 19449ul, 11953ul, 6698ul, 3543ul, 1605ul, 683ul, 185ul, 59ul, 26ul, 16ul};
inline constexpr unsigned long vsop2013_7_3_sizes[11]
    = {29609ul, 19128ul, 11526ul, 6371ul, 3243ul, 1417ul, 543ul, 134ul, 35ul, 10ul, 9ul};
inline constexpr unsigned long vsop2013_7_4_sizes[11]
    = {29734ul, 19252ul, 11660ul, 6450ul, 3285ul, 1434ul, 563ul, 145ul, 37ul, 10ul, 9ul};
inline constexpr unsigned long vsop2013_7_5_sizes[7] = {7053ul, 4099ul, 2151ul, 973ul, 339ul, 108ul, 12ul};
inline constexpr unsigned long vsop2013_7_6_sizes[8] = {6983ul, 4107ul, 2163ul, 979ul, 355ul, 111ul, 16ul, 2ul};

extern const double *const vsop2013_7_1_data[11];
extern const double *const vsop2013_7_2_data[11];
extern const double *const vsop2013_7_3_data[11];
extern const double *const vsop2013_7_4_data[11];
extern const double *const vsop2013_7_5_data[7];
extern const double *const vsop2013_7_6_data[8];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
