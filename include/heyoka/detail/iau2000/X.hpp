// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_IAU2000_X_HPP
#define HEYOKA_DETAIL_IAU2000_X_HPP

#include <cstddef>
#include <cstdint>

#include <heyoka/config.hpp>
#include <heyoka/mdspan.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

extern const mdspan<const std::int8_t, extents<std::size_t, 1306, 14>> iau2000_X_args_idxs_0;
extern const mdspan<const double, extents<std::size_t, 1306, 2>> iau2000_X_cfs_0;

extern const mdspan<const std::int8_t, extents<std::size_t, 253, 14>> iau2000_X_args_idxs_1;
extern const mdspan<const double, extents<std::size_t, 253, 2>> iau2000_X_cfs_1;

extern const mdspan<const std::int8_t, extents<std::size_t, 36, 14>> iau2000_X_args_idxs_2;
extern const mdspan<const double, extents<std::size_t, 36, 2>> iau2000_X_cfs_2;

extern const mdspan<const std::int8_t, extents<std::size_t, 4, 14>> iau2000_X_args_idxs_3;
extern const mdspan<const double, extents<std::size_t, 4, 2>> iau2000_X_cfs_3;

extern const mdspan<const std::int8_t, extents<std::size_t, 1, 14>> iau2000_X_args_idxs_4;
extern const mdspan<const double, extents<std::size_t, 1, 2>> iau2000_X_cfs_4;

} // namespace model::detail

HEYOKA_END_NAMESPACE

#endif
