// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fmt/format.h>

#include <heyoka/detail/string_conv.hpp>

namespace heyoka::detail
{

std::uint32_t uname_to_index(const std::string &s)
{
    using namespace fmt::literals;

    assert(s.rfind("u_", 0) == 0);

    std::uint32_t value;
    auto ret = std::from_chars(s.data() + 2, s.data() + s.size(), value);

    if (ret.ec != std::errc{}) {
        throw std::invalid_argument("Cannot extract a u variable index from the string '{}'"_format(s));
    }

    return value;
}

} // namespace heyoka::detail
