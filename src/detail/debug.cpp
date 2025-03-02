// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>

#include <heyoka/config.hpp>
#include <heyoka/detail/debug.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Global flag to signal whether or not to enable
// expensive debug checks. This is just used to avoid
// excessive runtime in certain unit tests.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool edb_is_enabled = true;

} // namespace

edb_disabler::edb_disabler()
{
    assert(edb_is_enabled);
    edb_is_enabled = false;
}

edb_disabler::~edb_disabler()
{
    assert(!edb_is_enabled);
    edb_is_enabled = true;
}

bool edb_enabled()
{
    return edb_is_enabled;
}

} // namespace detail

HEYOKA_END_NAMESPACE
