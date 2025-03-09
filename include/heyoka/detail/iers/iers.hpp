// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_IERS_IERS_HPP
#define HEYOKA_DETAIL_IERS_IERS_HPP

#include <boost/smart_ptr/atomic_shared_ptr.hpp>

#include <heyoka/config.hpp>
#include <heyoka/model/iers.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern boost::atomic_shared_ptr<const model::iers_data_t> cur_iers_data;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
