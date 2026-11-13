// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fmt/format.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/string_conv.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

std::uint32_t uname_to_index(const std::string &s)
{
    if (!s.starts_with("u_")) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Invalid string '{}' passed to uname_to_index(): the string does not begin with 'u_'", s));
        // LCOV_EXCL_STOP
    }

    std::uint32_t value = 0;
    const auto ret = std::from_chars(s.data() + 2, s.data() + s.size(), value);

    if (ret.ec != std::errc{}) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format(
            "Invalid string '{}' passed to uname_to_index(): could not parse an integer after the initial 'u_' prefix",
            s));
        // LCOV_EXCL_STOP
    }

    return value;
}

} // namespace detail

HEYOKA_END_NAMESPACE
