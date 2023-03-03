// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE
namespace detail
{

// Small helper to compute an index from the name
// of a u variable. E.g., for s = "u_123" this
// will return 123.
HEYOKA_DLL_PUBLIC std::uint32_t uname_to_index(const std::string &);

// Small helper to convert an input floating-point value to string.
// There are no guarantees on the output format - in fact this
// exists only for logging purposes as a workaround until mp++
// supports fmt formatting. Once happens, we can directly
// format mppp::real128 and the need for this helper disappears.
template <typename T>
HEYOKA_DLL_PUBLIC std::string fp_to_string(const T &);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
