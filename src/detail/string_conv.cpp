// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <charconv>
#include <cstdint>
#include <string>
#include <system_error>
#include <type_traits>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

std::uint32_t uname_to_index(const std::string &s)
{
    assert(s.rfind("u_", 0) == 0);

    std::uint32_t value = 0;
    [[maybe_unused]] auto ret = std::from_chars(s.data() + 2, s.data() + s.size(), value);
    assert(ret.ec == std::errc{});

    return value;
}

template <typename T>
std::string fp_to_string(const T &x)
{
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>) {
        return fmt::format("{}", x);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return x.to_string();
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if constexpr (std::is_same_v<T, mppp::real>) {
        return x.to_string();
#endif
    } else {
        static_assert(always_false_v<T>, "Unhandled type.");
    }
}

// Explicit instantiations.
template HEYOKA_DLL_PUBLIC std::string fp_to_string<double>(const double &);
template HEYOKA_DLL_PUBLIC std::string fp_to_string<long double>(const long double &);

#if defined(HEYOKA_HAVE_REAL128)

template HEYOKA_DLL_PUBLIC std::string fp_to_string<mppp::real128>(const mppp::real128 &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template HEYOKA_DLL_PUBLIC std::string fp_to_string<mppp::real>(const mppp::real &);

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
