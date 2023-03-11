// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this header contains typedefs for sets and maps used
// in several functions of the expression API which internally
// cache intermediate results for repeated subexpressions.
// We make use of the fast set and map classes from Boost>=1.81
// if available.

#ifndef HEYOKA_DETAIL_FUNC_CACHE_HPP
#define HEYOKA_DETAIL_FUNC_CACHE_HPP

#include <heyoka/config.hpp>

#include <boost/version.hpp>

#if (BOOST_VERSION / 100000 > 1) || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 81)

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

using funcptr_set = boost::unordered_flat_set<const void *>;

template <typename T>
using funcptr_map = boost::unordered_flat_map<const void *, T>;

} // namespace detail

HEYOKA_END_NAMESPACE

#else

#include <unordered_map>
#include <unordered_set>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

using funcptr_set = std::unordered_set<const void *>;

template <typename T>
using funcptr_map = std::unordered_map<const void *, T>;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif

#endif
