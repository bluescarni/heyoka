// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_SLEEF_WRAPPERS_HELPERS_HPP
#define HEYOKA_DETAIL_SLEEF_WRAPPERS_HELPERS_HPP

#include <sleef.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Small wrapper to invoke combined SLEEF functions returning 2 values. The values are written into 'out'.
template <typename V, typename F>
void sleef_pair_wrapper(V *const out, const V a, F func) noexcept
{
    const auto tmp = func(a);

    out[0] = tmp.x;
    out[1] = tmp.y;
}

} // namespace detail

HEYOKA_END_NAMESPACE

#define HEYOKA_SLEEF_PAIR_WRAPPER(sleef_func, vec_type)                                                                \
    extern "C" HEYOKA_DLL_PUBLIC void heyoka_##sleef_func(vec_type *const out, const vec_type a) noexcept              \
    {                                                                                                                  \
        heyoka::detail::sleef_pair_wrapper(out, a, sleef_func);                                                        \
    }

#endif
