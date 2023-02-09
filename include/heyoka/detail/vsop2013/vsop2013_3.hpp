// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_3_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_3_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

inline constexpr unsigned long vsop2013_3_1_sizes[9]
    = {32658ul, 20621ul, 11778ul, 5606ul, 2380ul, 1010ul, 434ul, 110ul, 18ul};
inline constexpr unsigned long vsop2013_3_2_sizes[11]
    = {31440ul, 18896ul, 10525ul, 5102ul, 2309ul, 1019ul, 431ul, 120ul, 28ul, 1ul, 1ul};
inline constexpr unsigned long vsop2013_3_3_sizes[10]
    = {28260ul, 16450ul, 8937ul, 4246ul, 1978ul, 897ul, 350ul, 78ul, 11ul, 2ul};
inline constexpr unsigned long vsop2013_3_4_sizes[10]
    = {27989ul, 16305ul, 8790ul, 4150ul, 1961ul, 882ul, 340ul, 75ul, 11ul, 1ul};
inline constexpr unsigned long vsop2013_3_5_sizes[8] = {7293ul, 3681ul, 1964ul, 953ul, 446ul, 126ul, 19ul, 1ul};
inline constexpr unsigned long vsop2013_3_6_sizes[8] = {6766ul, 3593ul, 1877ul, 935ul, 438ul, 115ul, 18ul, 1ul};

extern const double *const vsop2013_3_1_data[9];
extern const double *const vsop2013_3_2_data[11];
extern const double *const vsop2013_3_3_data[10];
extern const double *const vsop2013_3_4_data[10];
extern const double *const vsop2013_3_5_data[8];
extern const double *const vsop2013_3_6_data[8];

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
