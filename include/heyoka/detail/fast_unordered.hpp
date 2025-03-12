// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_FAST_UNORDERED_HPP
#define HEYOKA_DETAIL_FAST_UNORDERED_HPP

#include <heyoka/config.hpp>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename... Args>
using fast_uset = boost::unordered_flat_set<Args...>;

template <typename... Args>
using fast_umap = boost::unordered_flat_map<Args...>;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
