// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_FAST_UNORDERED_HPP
#define HEYOKA_DETAIL_FAST_UNORDERED_HPP

#include <heyoka/config.hpp>

#include <boost/version.hpp>

#if (BOOST_VERSION / 100000 > 1) || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 81)

#define HEYOKA_HAVE_BOOST_UNORDERED_FLAT

#endif

#if defined(HEYOKA_HAVE_BOOST_UNORDERED_FLAT)

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#else

#include <unordered_map>
#include <unordered_set>

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename... Args>
using fast_uset =
#if defined(HEYOKA_HAVE_BOOST_UNORDERED_FLAT)
    boost::unordered_flat_set<Args...>
#else
    std::unordered_set<Args...>
#endif
    ;

template <typename... Args>
using fast_umap =
#if defined(HEYOKA_HAVE_BOOST_UNORDERED_FLAT)
    boost::unordered_flat_map<Args...>
#else
    std::unordered_map<Args...>
#endif
    ;

} // namespace detail

HEYOKA_END_NAMESPACE

#if defined(HEYOKA_HAVE_BOOST_UNORDERED_FLAT)

#undef HEYOKA_HAVE_BOOST_UNORDERED_FLAT

#endif

#endif
