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

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_max128(__float128 x, __float128 y)
{
    return mppp::fmax(mppp::real128{x}, mppp::real128{y}).m_value;
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

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_asin128(__float128 x)
{
    return mppp::asin(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_acos128(__float128 x)
{
    return mppp::acos(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_atan128(__float128 x)
{
    return mppp::atan(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_cosh128(__float128 x)
{
    return mppp::cosh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sinh128(__float128 x)
{
    return mppp::sinh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_tanh128(__float128 x)
{
    return mppp::tanh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_asinh128(__float128 x)
{
    return mppp::asinh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_acosh128(__float128 x)
{
    return mppp::acosh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_atanh128(__float128 x)
{
    return mppp::atanh(mppp::real128{x}).m_value;
}

extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_erf128(__float128 x)
{
    return mppp::erf(mppp::real128{x}).m_value;
}

extern "C" {

using heyoka_f128_pair = struct {
    __float128 x;
    __float128 y;
};

HEYOKA_DLL_PUBLIC heyoka_f128_pair heyoka_sincos128(__float128 x)
{
    mppp::real128 s, c;

    mppp::sincos(mppp::real128{x}, &s, &c);

    return heyoka_f128_pair{s.m_value, c.m_value};
}
}

#endif

#if defined(_MSC_VER)

#include <cmath>

// NOTE: there seems to be issues when trying to invoke long double
// math functions on MSVC (LLVM complaining about missing symbols).
// Let's create ad-hoc wrappers.
extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanl(long double x)
{
    return std::tan(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_asinl(long double x)
{
    return std::asin(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_acosl(long double x)
{
    return std::acos(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_atanl(long double x)
{
    return std::atan(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_coshl(long double x)
{
    return std::cosh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_sinhl(long double x)
{
    return std::sinh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanhl(long double x)
{
    return std::tanh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_asinhl(long double x)
{
    return std::asinh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_acoshl(long double x)
{
    return std::acosh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_atanhl(long double x)
{
    return std::atanh(x);
}

extern "C" HEYOKA_DLL_PUBLIC long double heyoka_erfl(long double x)
{
    return std::erf(x);
}

#endif
