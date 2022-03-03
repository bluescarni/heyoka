// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EXCEPTIONS_HPP
#define HEYOKA_EXCEPTIONS_HPP

#include <stdexcept>

#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

struct HEYOKA_DLL_PUBLIC_INLINE_CLASS not_implemented_error final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Exception to signal division by zero.
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS zero_division_error final : std::domain_error {
    using std::domain_error::domain_error;
};

} // namespace heyoka

#endif
