// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_max128(__float128 x, __float128 y) noexcept
{
    return mppp::fmax(mppp::real128{x}, mppp::real128{y}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_maxabs128(__float128 x, __float128 y) noexcept
{
    return mppp::fmax(mppp::real128{x}, mppp::abs(mppp::real128{y})).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minabs128(__float128 x, __float128 y) noexcept
{
    return mppp::fmin(mppp::real128{x}, mppp::abs(mppp::real128{y})).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minnum128(__float128 x, __float128 y) noexcept
{
    return mppp::fmin(mppp::real128{x}, mppp::real128{y}).m_value;
}

#endif

#if defined(_MSC_VER)

#include <cmath>

// NOTE: there seems to be issues when trying to invoke long double
// math functions on MSVC (LLVM complaining about missing symbols).
// Let's create ad-hoc wrappers.
extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanl(long double x) noexcept
{
    return std::tan(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_asinl(long double x) noexcept
{
    return std::asin(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_acosl(long double x) noexcept
{
    return std::acos(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_atanl(long double x) noexcept
{
    return std::atan(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_coshl(long double x) noexcept
{
    return std::cosh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_sinhl(long double x) noexcept
{
    return std::sinh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanhl(long double x) noexcept
{
    return std::tanh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_asinhl(long double x) noexcept
{
    return std::asinh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_acoshl(long double x) noexcept
{
    return std::acosh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_atanhl(long double x) noexcept
{
    return std::atanh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_erfl(long double x) noexcept
{
    return std::erf(x);
}

#endif
