// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this is a workaround for a compilation issue on OSX, where clang complains that malloc()/free() declarations
// (used somewhere inside fmt) are not available. See:
// https://github.com/fmtlib/fmt/pull/4477
#include <cstdlib>

#include <cstddef>
#include <new>
#include <stdexcept>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/aligned_vector.hpp>
#include <heyoka/detail/safe_integer.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

void *aligned_allocate(const std::size_t n, const std::size_t sz, const std::size_t al)
{
    return ::operator new(boost::safe_numerics::safe<std::size_t>(n) * sz, std::align_val_t{al});
}

void aligned_deallocate(void *const ptr, const std::size_t n, const std::size_t sz, const std::size_t al)
{
    // NOTE: no need to check the n * sz multiplication here, as we assume that this is called only on a pointer
    // returned by a successful aligned_allocate() invocation with the same n/sz/al values.
    ::operator delete(ptr, n * sz, std::align_val_t{al});
}

void check_alignment(const std::size_t al)
{
    if (al == 0u) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument("The alignment value in an aligned_allocator cannot be zero");
        // LCOV_EXCL_STOP
    }

    if ((al & (al - 1u)) != 0u) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format(
            "An invalid alignment of {} was specified in an aligned_allocator: the alignment must be a power of 2",
            al));
        // LCOV_EXCL_STOP
    }
}

} // namespace detail

HEYOKA_END_NAMESPACE
