// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_VSOP2013_VSOP2013_1_HPP
#define HEYOKA_DETAIL_VSOP2013_VSOP2013_1_HPP

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

inline constexpr unsigned long vsop2013_1_1_sizes[11]
    = {32240ul, 20592ul, 11166ul, 5013ul, 2139ul, 1016ul, 473ul, 170ul, 75ul, 23ul, 1ul};
inline constexpr unsigned long vsop2013_1_2_sizes[11]
    = {28251ul, 17448ul, 9533ul, 4429ul, 2044ul, 1005ul, 501ul, 191ul, 89ul, 24ul, 2ul};
inline constexpr unsigned long vsop2013_1_3_sizes[11]
    = {23686ul, 14395ul, 7602ul, 3564ul, 1712ul, 835ul, 415ul, 157ul, 58ul, 22ul, 1ul};
inline constexpr unsigned long vsop2013_1_4_sizes[11]
    = {23695ul, 14602ul, 7712ul, 3598ul, 1729ul, 839ul, 406ul, 159ul, 61ul, 23ul, 1ul};
inline constexpr unsigned long vsop2013_1_5_sizes[9] = {6948ul, 3811ul, 1793ul, 909ul, 446ul, 163ul, 67ul, 19ul, 3ul};
inline constexpr unsigned long vsop2013_1_6_sizes[9] = {8064ul, 4404ul, 2155ul, 1041ul, 513ul, 208ul, 86ul, 27ul, 6ul};

extern const double *const vsop2013_1_1_data[11];
extern const double *const vsop2013_1_2_data[11];
extern const double *const vsop2013_1_3_data[11];
extern const double *const vsop2013_1_4_data[11];
extern const double *const vsop2013_1_5_data[9];
extern const double *const vsop2013_1_6_data[9];

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
