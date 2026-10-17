// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

// NOTE: forcefully enable both the square and round brackets versions of the indexing operator. We are assuming C++23
// in any case. Setting these explicitly allows us to both support older code which may still be using the round
// brackets operator and to work around any compiler feature detection issue that might appear in Kokkos' mdspan
// implementation (e.g., around the clang vs apple clang versioning mess).
#define MDSPAN_USE_PAREN_OPERATOR 1
#define MDSPAN_USE_BRACKET_OPERATOR 1

// NOTE: in debug mode, activate additional mdspan debugging assertions.
#if !defined(NDEBUG)
#define MDSPAN_DEBUG
#endif

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"

#if __GNUC__ >= 15

#pragma GCC diagnostic ignored "-Wtrailing-whitespace="

#endif

#endif

#include <heyoka/detail/mdspan/mdspan/mdspan.hpp>

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#if !defined(NDEBUG)
#undef MDSPAN_DEBUG
#endif

#undef MDSPAN_USE_BRACKET_OPERATOR
#undef MDSPAN_USE_PAREN_OPERATOR

HEYOKA_BEGIN_NAMESPACE

// NOTE: the idea here is that we lift some of the public names from C++23/C++26's std::mdspan into the heyoka
// namespace.

template <typename IndexType, std::size_t... Extents>
using extents = Kokkos::extents<IndexType, Extents...>;

template <typename IndexType, std::size_t Rank>
using dextents = Kokkos::dextents<IndexType, Rank>;

using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::layout_stride;

template <typename ElementType>
using default_accessor = Kokkos::default_accessor<ElementType>;

template <typename T, typename Extents, typename LayoutPolicy = layout_right,
          typename AccessorPolicy = default_accessor<T> >
using mdspan = Kokkos::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;

inline constexpr Kokkos::full_extent_t full_extent{};

using Kokkos::submdspan;

HEYOKA_END_NAMESPACE

#endif
