// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <new>

#include <boost/predef.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/max_align_vector.hpp>
#include <heyoka/detail/safe_integer.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// NOTE: the idea here is that we want to account for SIMD types, which are not necessarily accounted for by
// max_align_t.
const std::size_t max_alignment =
#if defined(BOOST_ARCH_X86)
    64
#elif defined(BOOST_ARCH_ARM) || defined(BOOST_ARCH_PPC)
    16
#else
    alignof(std::max_align_t)
#endif
    ;

void *max_align_allocate(const std::size_t n, const std::size_t sz)
{
    return ::operator new(boost::safe_numerics::safe<std::size_t>(n) * sz, std::align_val_t{max_alignment});
}

void max_align_deallocate(void *const ptr, const std::size_t n, const std::size_t sz)
{
    // NOTE: no need to check the n * sz multiplication here, as we assume that this is called only on a pointer
    // returned by a successful max_align_allocate() invocation with the same n/sz values.
    ::operator delete(ptr, n * sz, std::align_val_t{max_alignment});
}

} // namespace detail

HEYOKA_END_NAMESPACE
