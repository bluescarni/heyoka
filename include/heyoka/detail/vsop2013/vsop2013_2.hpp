// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_2_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_2_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

inline constexpr unsigned long vsop2013_2_1_sizes[10]
    = {32412ul, 20500ul, 11381ul, 5319ul, 2142ul, 948ul, 420ul, 140ul, 53ul, 17ul};
inline constexpr unsigned long vsop2013_2_2_sizes[11]
    = {29979ul, 18284ul, 10091ul, 4784ul, 2111ul, 965ul, 424ul, 146ul, 50ul, 16ul, 1ul};
inline constexpr unsigned long vsop2013_2_3_sizes[11]
    = {27030ul, 16132ul, 8555ul, 3850ul, 1792ul, 818ul, 361ul, 99ul, 33ul, 9ul, 1ul};
inline constexpr unsigned long vsop2013_2_4_sizes[10]
    = {26984ul, 16192ul, 8600ul, 3858ul, 1797ul, 831ul, 345ul, 94ul, 32ul, 8ul};
inline constexpr unsigned long vsop2013_2_5_sizes[9] = {8102ul, 4417ul, 2149ul, 1001ul, 453ul, 163ul, 53ul, 9ul, 1ul};
inline constexpr unsigned long vsop2013_2_6_sizes[9] = {7628ul, 4319ul, 2051ul, 999ul, 463ul, 171ul, 53ul, 9ul, 2ul};

extern const double *const vsop2013_2_1_data[10];
extern const double *const vsop2013_2_2_data[11];
extern const double *const vsop2013_2_3_data[11];
extern const double *const vsop2013_2_4_data[10];
extern const double *const vsop2013_2_5_data[9];
extern const double *const vsop2013_2_6_data[9];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
