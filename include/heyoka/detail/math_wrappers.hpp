// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_MATH_WRAPPERS_HPP
#define HEYOKA_DETAIL_MATH_WRAPPERS_HPP

#include <heyoka/config.hpp>

#include <cmath>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/visibility.hpp>

namespace heyoka::detail
{

template <typename T>
inline bool isfinite(T x)
{
    return std::isfinite(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline bool isfinite<mppp::real128>(mppp::real128 x)
{
    return mppp::finite(x);
}

#endif

} // namespace heyoka::detail

#if defined(HEYOKA_HAVE_REAL128)

// NOTE: these wrappers are needed as a replacement
// for the LLVM builtins, which seem to have issues
// when invoked with __float128 arguments. This may be
// related to incompatibilities between GCC
// and LLVM in the implementation of __float128,
// and needs to be investigated more.
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_pow128(__float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_log128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sin128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_cos128(__float128);

#endif

#endif
