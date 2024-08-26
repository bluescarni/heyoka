// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ALIGNED_BUFFER_HPP
#define HEYOKA_DETAIL_ALIGNED_BUFFER_HPP

#include <cstddef>
#include <memory>

#include <heyoka/config.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Utilities to create and destroy tape arrays for compiled functions
// and/or Taylor integrators in compact mode. These may have custom alignment requirements due
// to the use of SIMD instructions, hence we need to use aligned new/delete
// and a custom deleter for the unique ptr.
struct aligned_buffer_deleter {
    std::align_val_t al{};
    void operator()(void *ptr) const noexcept;
};

using aligned_buffer_t = std::unique_ptr<std::byte[], aligned_buffer_deleter>;

aligned_buffer_t make_aligned_buffer(std::size_t, std::size_t);

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
