// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: MSVC does not define __SSE2__ even though SSE2 is the baseline on x64. SLEEF's header guards its SSE4
// declarations behind #ifdef __SSE2__, so we need to define it manually.
#if defined(_MSC_VER) && !defined(__SSE2__)
#define __SSE2__
#endif

#include "sleef_wrappers_helpers.hpp"

// Double-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u10sse4, __m128d)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd4_u10avx, __m256d)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u35sse4, __m128d)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd4_u35avx, __m256d)

// Single-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u10sse4, __m128)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf8_u10avx, __m256)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u35sse4, __m128)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf8_u35avx, __m256)
