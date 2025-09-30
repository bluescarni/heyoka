// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ALIGNED_VECTOR_HPP
#define HEYOKA_DETAIL_ALIGNED_VECTOR_HPP

#include <cstddef>
#include <type_traits>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

HEYOKA_DLL_PUBLIC void *aligned_allocate(std::size_t, std::size_t, std::size_t);
HEYOKA_DLL_PUBLIC void aligned_deallocate(void *, std::size_t, std::size_t, std::size_t);
HEYOKA_DLL_PUBLIC void check_alignment(std::size_t);

template <class T>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
class aligned_allocator
{
    std::size_t m_al;

    template <typename U>
    friend class aligned_allocator;

public:
    // NOTE: we use the same typedefs as std::allocator. See:
    //
    // https://eel.is/c++draft/default.allocator
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // NOTE: these indicate that the allocator must "follow" the data when a container is copy/move assigned or swapped.
    // See https://en.cppreference.com/w/cpp/container/vector/operator=.html (and the corresponding page for swap()) for
    // a detailed explanation.
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    explicit aligned_allocator(const std::size_t al) : m_al(al)
    {
        check_alignment(m_al);
    }
    // NOTE: no need to implement move semantics here, copy semantics is sufficient.
    aligned_allocator(const aligned_allocator &) noexcept = default;
    template <class U>
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    aligned_allocator(const aligned_allocator<U> &other) noexcept : m_al(other.m_al){};
    ~aligned_allocator() = default;
    aligned_allocator &operator=(const aligned_allocator &) noexcept = default;

    T *allocate(std::size_t n)
    {
        return static_cast<T *>(aligned_allocate(n, sizeof(T), m_al));
    }
    void deallocate(T *p, std::size_t n)
    {
        aligned_deallocate(p, n, sizeof(T), m_al);
    }

    [[nodiscard]] std::size_t get_alignment() const noexcept
    {
        return m_al;
    }

    // NOTE: these equality comparisons will indicate that memory allocated by a1 can be deallocated by a2 only if the
    // alignment values match.
    friend bool operator==(const aligned_allocator &a1, const aligned_allocator &a2) noexcept
    {
        return a1.m_al == a2.m_al;
    }
    friend bool operator!=(const aligned_allocator &a1, const aligned_allocator &a2) noexcept
    {
        return !(a1 == a2);
    }
};

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
