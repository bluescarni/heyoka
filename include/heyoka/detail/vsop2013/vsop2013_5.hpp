// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_5_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_5_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr unsigned long vsop2013_5_1_sizes[21]
    = {32466ul, 24467ul, 15345ul, 8814ul, 4621ul, 2242ul, 1102ul, 573ul, 287ul, 94ul, 22ul,
       12ul,    12ul,    12ul,    12ul,   12ul,   12ul,   12ul,   12ul,  12ul,  12ul};
inline constexpr unsigned long vsop2013_5_2_sizes[19]
    = {29250ul, 20297ul, 12703ul, 7370ul, 3946ul, 1984ul, 1000ul, 497ul, 242ul, 89ul,
       24ul,    13ul,    12ul,    12ul,   12ul,   12ul,   10ul,   7ul,   2ul};
inline constexpr unsigned long vsop2013_5_3_sizes[11]
    = {26026ul, 17281ul, 10467ul, 5951ul, 3036ul, 1506ul, 789ul, 402ul, 174ul, 54ul, 3ul};
inline constexpr unsigned long vsop2013_5_4_sizes[11]
    = {26233ul, 17363ul, 10525ul, 5954ul, 3089ul, 1498ul, 774ul, 408ul, 181ul, 52ul, 4ul};
inline constexpr unsigned long vsop2013_5_5_sizes[9] = {5805ul, 3291ul, 1658ul, 878ul, 487ul, 287ul, 143ul, 38ul, 2ul};
inline constexpr unsigned long vsop2013_5_6_sizes[9] = {5697ul, 3358ul, 1693ul, 894ul, 503ul, 294ul, 142ul, 32ul, 1ul};

extern const double *const vsop2013_5_1_data[21];
extern const double *const vsop2013_5_2_data[19];
extern const double *const vsop2013_5_3_data[11];
extern const double *const vsop2013_5_4_data[11];
extern const double *const vsop2013_5_5_data[9];
extern const double *const vsop2013_5_6_data[9];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
