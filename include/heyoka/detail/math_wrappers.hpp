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

inline float fma(float x, float y, float z)
{
#if defined(FP_FAST_FMAF)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

inline double fma(double x, double y, double z)
{
#if defined(FP_FAST_FMA)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

inline long double fma(long double x, long double y, long double z)
{
#if defined(FP_FAST_FMAL)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

#if defined(HEYOKA_HAVE_REAL128)

inline mppp::real128 fma(mppp::real128 x, mppp::real128 y, mppp::real128 z)
{
    return mppp::fma(x, y, z);
}

#endif

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

template <typename T>
inline bool isnan(T x)
{
    return std::isnan(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline bool isnan<mppp::real128>(mppp::real128 x)
{
    return mppp::isnan(x);
}

#endif

template <typename T>
inline T abs(T x)
{
    return std::abs(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline mppp::real128 abs<mppp::real128>(mppp::real128 x)
{
    return mppp::abs(x);
}

#endif

template <typename T>
inline T log(T x)
{
    return std::log(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline mppp::real128 log<mppp::real128>(mppp::real128 x)
{
    return mppp::log(x);
}

#endif

template <typename T>
inline T exp(T x)
{
    return std::exp(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline mppp::real128 exp<mppp::real128>(mppp::real128 x)
{
    return mppp::exp(x);
}

#endif

template <typename T>
inline T ceil(T x)
{
    return std::ceil(x);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline mppp::real128 ceil<mppp::real128>(mppp::real128 x)
{
    return mppp::ceil(x);
}

#endif

template <typename T>
inline T pow(T x, T y)
{
    return std::pow(x, y);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
inline mppp::real128 pow<mppp::real128>(mppp::real128 x, mppp::real128 y)
{
    return mppp::pow(x, y);
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
