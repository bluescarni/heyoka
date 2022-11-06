// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_NUM_UTILS_HPP
#define HEYOKA_DETAIL_NUM_UTILS_HPP

// This header contains function that allow to treat number-like
// entities in a uniform way, even when the number-like entity might have
// extra properties in addition to its numerical value (e.g., the precision
// in case of mppp::real).

namespace heyoka::detail
{

template <typename T>
T num_zero_like(const T &);

template <typename T>
T num_eps_like(const T &);

template <typename T>
T num_inf_like(const T &);

} // namespace heyoka::detail

#endif
