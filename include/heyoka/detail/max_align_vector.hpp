// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_MAX_ALIGN_VECTOR_HPP
#define HEYOKA_DETAIL_MAX_ALIGN_VECTOR_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

extern const std::size_t max_alignment;

HEYOKA_DLL_PUBLIC void *max_align_allocate(std::size_t, std::size_t);
HEYOKA_DLL_PUBLIC void max_align_deallocate(void *, std::size_t, std::size_t);

// NOTE: see https://eel.is/c++draft/default.allocator.
template <class T>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct max_align_allocator : std::allocator<T> {
    constexpr max_align_allocator() noexcept = default;
    constexpr max_align_allocator(const max_align_allocator &) noexcept = default;
    template <class U>
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    constexpr max_align_allocator(const max_align_allocator<U> &) noexcept {};
    constexpr ~max_align_allocator() = default;
    constexpr max_align_allocator &operator=(const max_align_allocator &) = default;

    T *allocate(std::size_t n)
    {
        return static_cast<T *>(max_align_allocate(n, sizeof(T)));
    }
    void deallocate(T *p, std::size_t n)
    {
        max_align_deallocate(p, n, sizeof(T));
    }
};

template <typename T>
using max_align_vector = std::vector<T, max_align_allocator<T>>;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
