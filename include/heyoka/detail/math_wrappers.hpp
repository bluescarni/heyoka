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
#include <heyoka/detail/visibility.hpp>

#if defined(HEYOKA_HAVE_REAL128)

// NOTE: these wrappers are needed as a replacement
// for the LLVM builtins, which seem to have issues
// when invoked with __float128 arguments. This may be
// related to incompatibilities between GCC
// and LLVM in the implementation of __float128,
// and needs to be investigated more.
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_pow128(__float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_log128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_exp128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sin128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_cos128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_sqrt128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_fma128(__float128, __float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_maxabs128(__float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minabs128(__float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_minnum128(__float128, __float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_abs128(__float128);
extern "C" HEYOKA_DLL_PUBLIC __float128 heyoka_tan128(__float128);

#endif

#if defined(_MSC_VER)

// NOTE: there seems issues when trying to invoke the tanl()
// function on MSVC (LLVM complaining about missing symbol).
// Let's create an ad-hoc wrapper.
extern "C" HEYOKA_DLL_PUBLIC long double heyoka_tanl(long double);

#endif

#endif
