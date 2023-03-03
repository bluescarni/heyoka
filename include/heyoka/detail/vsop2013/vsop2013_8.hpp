// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_8_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_8_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

inline constexpr unsigned long vsop2013_8_1_sizes[12]
    = {32767ul, 23195ul, 15045ul, 8945ul, 5079ul, 2532ul, 1187ul, 450ul, 190ul, 84ul, 52ul, 1ul};
inline constexpr unsigned long vsop2013_8_2_sizes[11]
    = {24498ul, 16513ul, 10787ul, 6449ul, 3752ul, 1903ul, 928ul, 332ul, 124ul, 37ul, 14ul};
inline constexpr unsigned long vsop2013_8_3_sizes[11]
    = {26949ul, 17860ul, 11265ul, 6699ul, 3862ul, 1969ul, 928ul, 338ul, 134ul, 46ul, 14ul};
inline constexpr unsigned long vsop2013_8_4_sizes[11]
    = {27003ul, 17979ul, 11380ul, 6786ul, 3919ul, 1990ul, 941ul, 341ul, 132ul, 46ul, 13ul};
inline constexpr unsigned long vsop2013_8_5_sizes[9] = {6409ul, 3589ul, 1970ul, 966ul, 393ul, 149ul, 55ul, 19ul, 2ul};
inline constexpr unsigned long vsop2013_8_6_sizes[11]
    = {6391ul, 3612ul, 1981ul, 965ul, 391ul, 142ul, 52ul, 20ul, 4ul, 2ul, 2ul};

extern const double *const vsop2013_8_1_data[12];
extern const double *const vsop2013_8_2_data[11];
extern const double *const vsop2013_8_3_data[11];
extern const double *const vsop2013_8_4_data[11];
extern const double *const vsop2013_8_5_data[9];
extern const double *const vsop2013_8_6_data[11];

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
