// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_STRING_CONV_HPP
#define HEYOKA_DETAIL_STRING_CONV_HPP

#include <cstdint>
#include <string>

#include <heyoka/detail/visibility.hpp>

namespace heyoka::detail
{

// Small helper to compute an index from the name
// of a u variable. E.g., for s = "u_123" this
// will return 123.
HEYOKA_DLL_PUBLIC std::uint32_t uname_to_index(const std::string &);

} // namespace heyoka::detail

#endif
