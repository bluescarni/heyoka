// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EXCEPTIONS_HPP
#define HEYOKA_EXCEPTIONS_HPP

#include <stdexcept>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

struct HEYOKA_DLL_PUBLIC_INLINE_CLASS not_implemented_error final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

HEYOKA_END_NAMESPACE

#endif
