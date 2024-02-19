// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_4_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_4_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE
namespace model::detail
{

inline constexpr unsigned long vsop2013_4_1_sizes[9]
    = {32150ul, 22578ul, 14113ul, 7186ul, 3045ul, 1075ul, 439ul, 126ul, 14ul};
inline constexpr unsigned long vsop2013_4_2_sizes[10]
    = {32418ul, 21400ul, 12119ul, 5781ul, 2681ul, 1121ul, 420ul, 130ul, 22ul, 1ul};
inline constexpr unsigned long vsop2013_4_3_sizes[9]
    = {29425ul, 17390ul, 10073ul, 4792ul, 2118ul, 923ul, 359ul, 80ul, 8ul};
inline constexpr unsigned long vsop2013_4_4_sizes[9]
    = {28828ul, 16819ul, 9824ul, 4668ul, 2062ul, 905ul, 350ul, 75ul, 8ul};
inline constexpr unsigned long vsop2013_4_5_sizes[8] = {5914ul, 3156ul, 1688ul, 820ul, 360ul, 120ul, 11ul, 1ul};
inline constexpr unsigned long vsop2013_4_6_sizes[7] = {5638ul, 2973ul, 1646ul, 806ul, 356ul, 113ul, 12ul};

extern const double *const vsop2013_4_1_data[9];
extern const double *const vsop2013_4_2_data[10];
extern const double *const vsop2013_4_3_data[9];
extern const double *const vsop2013_4_4_data[9];
extern const double *const vsop2013_4_5_data[8];
extern const double *const vsop2013_4_6_data[7];

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
