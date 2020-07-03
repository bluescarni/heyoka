// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_XOROSHIRO128_PLUS_HPP
#define HEYOKA_DETAIL_XOROSHIRO128_PLUS_HPP

#include <cstdint>
#include <limits>

namespace heyoka::detail
{

// NOTE: constexpr implementation, thus usable at compile-time.
struct splitmix64 {
    // Constructor from seed state (64-bit values).
    constexpr explicit splitmix64(const ::std::uint64_t &s) : m_state{s} {}

    // Compute the next 64-bit value in the sequence.
    constexpr ::std::uint64_t next()
    {
        ::std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    // Provide also an interface compatible with the UniformRandomBitGenerator concept:
    // https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator
    using result_type = ::std::uint64_t;
    static constexpr result_type min()
    {
        return 0;
    }
    static constexpr result_type max()
    {
        return ::std::numeric_limits<result_type>::max();
    }
    constexpr result_type operator()()
    {
        return next();
    }

    ::std::uint64_t m_state;
};

} // namespace heyoka::detail

#endif