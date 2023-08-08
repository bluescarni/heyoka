// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_9_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_9_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr unsigned long vsop2013_9_1_sizes[13]
    = {7774ul, 6473ul, 5400ul, 4114ul, 3277ul, 2372ul, 1889ul, 1354ul, 1021ul, 905ul, 888ul, 681ul, 283ul};
inline constexpr unsigned long vsop2013_9_2_sizes[13]
    = {4007ul, 3530ul, 2716ul, 1920ul, 1671ul, 1503ul, 1294ul, 850ul, 560ul, 419ul, 390ul, 345ul, 188ul};
inline constexpr unsigned long vsop2013_9_3_sizes[13]
    = {4372ul, 3725ul, 2884ul, 2038ul, 1763ul, 1591ul, 1382ul, 936ul, 589ul, 408ul, 375ul, 330ul, 182ul};
inline constexpr unsigned long vsop2013_9_4_sizes[13]
    = {4381ul, 3740ul, 2903ul, 2051ul, 1783ul, 1607ul, 1421ul, 958ul, 612ul, 444ul, 410ul, 363ul, 209ul};
inline constexpr unsigned long vsop2013_9_5_sizes[13]
    = {2021ul, 1752ul, 1359ul, 843ul, 641ul, 515ul, 401ul, 237ul, 122ul, 55ul, 43ul, 37ul, 13ul};
inline constexpr unsigned long vsop2013_9_6_sizes[13]
    = {2053ul, 1838ul, 1455ul, 948ul, 741ul, 616ul, 486ul, 276ul, 146ul, 71ul, 63ul, 56ul, 19ul};

extern const double *const vsop2013_9_1_data[13];
extern const double *const vsop2013_9_2_data[13];
extern const double *const vsop2013_9_3_data[13];
extern const double *const vsop2013_9_4_data[13];
extern const double *const vsop2013_9_5_data[13];
extern const double *const vsop2013_9_6_data[13];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
