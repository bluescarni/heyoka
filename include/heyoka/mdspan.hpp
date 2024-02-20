// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MDSPAN_HPP
#define HEYOKA_MDSPAN_HPP

#include <heyoka/config.hpp>

#include <cstddef>

#define MDSPAN_USE_PAREN_OPERATOR 1

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#endif

#include <heyoka/detail/mdspan/mdspan>

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#undef MDSPAN_USE_PAREN_OPERATOR

HEYOKA_BEGIN_NAMESPACE

template <typename T, typename Extents, typename LayoutPolicy = std::experimental::layout_right,
          typename AccessorPolicy = std::experimental::default_accessor<T> >
using mdspan = std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;

template <typename IndexType, std::size_t... Extents>
using extents = std::experimental::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = std::experimental::dextents<IndexType, Rank>;

HEYOKA_END_NAMESPACE

#endif
