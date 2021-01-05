// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>
#include <heyoka/detail/visibility.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL128)

// Make sure that the size and alignment of __float128
// and mppp::real128 coincide. This is required if we
// want to be able to use mppp::real128 as an alias
// for __float128.
static_assert(sizeof(__float128) == sizeof(mppp::real128));
static_assert(alignof(__float128) == alignof(mppp::real128));

// NOTE: these wrappers are needed as a replacement
// for the LLVM builtins, which seem to have issues
// when invoked with __float128 arguments. This may be
// related to incompatibilities between GCC
// and LLVM in the implementation of __float128,
// and needs to be investigated more.
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_pow128(__float128 x, __float128 y)
{
    return mppp::pow(mppp::real128{x}, mppp::real128{y}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_log128(__float128 x)
{
    return mppp::log(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_exp128(__float128 x)
{
    return mppp::exp(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sin128(__float128 x)
{
    return mppp::sin(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_cos128(__float128 x)
{
    return mppp::cos(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sqrt128(__float128 x)
{
    return mppp::sqrt(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_fma128(__float128 x, __float128 y, __float128 z)
{
    return mppp::fma(mppp::real128{x}, mppp::real128{y}, mppp::real128{z}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_maxabs128(__float128 x, __float128 y)
{
    return mppp::fmax(mppp::real128{x}, mppp::abs(mppp::real128{y})).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minabs128(__float128 x, __float128 y)
{
    return mppp::fmin(mppp::real128{x}, mppp::abs(mppp::real128{y})).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minnum128(__float128 x, __float128 y)
{
    return mppp::fmin(mppp::real128{x}, mppp::real128{y}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_abs128(__float128 x)
{
    return mppp::abs(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_tan128(__float128 x)
{
    return mppp::tan(mppp::real128{x}).m_value;
}

#endif

#if defined(_MSC_VER)

#include <cmath>

// NOTE: there seems issues when trying to invoke the tanl()
// function on MSVC (LLVM complaining about missing symbol).
// Let's create an ad-hoc wrapper.
extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanl(long double x)
{
    return std::tan(x);
}

#endif
