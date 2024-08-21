// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <new>

#include <heyoka/config.hpp>
#include <heyoka/detail/aligned_buffer.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

void aligned_buffer_deleter::operator()(void *ptr) const noexcept
{
    // NOTE: here we are using directly the delete operator (which does not invoke destructors),
    // rather than a delete expression (which would also invoke destructors). However, because
    // ptr points to a bytes array, we do not need to explicitly call the destructor here, deallocation will be
    // sufficient.
    ::operator delete[](ptr, al);
}

aligned_buffer_t make_aligned_buffer(std::size_t sz, std::size_t al)
{
    assert(al > 0u);
    assert((al & (al - 1u)) == 0u);

    if (sz == 0u) {
        return {};
    } else {
#if defined(_MSC_VER)
        // MSVC workaround for this issue:
        // https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320

        // Allocate the raw memory.
        auto *buf = ::operator new[](sz, std::align_val_t{al});

        // Formally construct the bytes array.
        auto *ptr = ::new (buf) std::byte[sz];

        // Construct and return the unique ptr.
        return aligned_buffer_t{ptr, {.al = std::align_val_t{al}}};
#else
        return aligned_buffer_t{::new (std::align_val_t{al}) std::byte[sz], {.al = std::align_val_t{al}}};
#endif
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
