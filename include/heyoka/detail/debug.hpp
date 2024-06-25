// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_DEBUG_HPP
#define HEYOKA_DETAIL_DEBUG_HPP

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

struct HEYOKA_DLL_PUBLIC edb_disabler {
    edb_disabler();
    ~edb_disabler();

    edb_disabler(const edb_disabler &) = delete;
    edb_disabler(edb_disabler &&) = delete;
    edb_disabler &operator=(const edb_disabler &) = delete;
    edb_disabler &operator=(edb_disabler &&) = delete;
};

[[nodiscard]] HEYOKA_DLL_PUBLIC bool edb_enabled();

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
