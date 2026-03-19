// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sleef_wrappers_helpers.hpp"

// LCOV_EXCL_START

// Double-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u10advsimd, float64x2_t)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u35advsimd, float64x2_t)

// Single-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u10advsimd, float32x4_t)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u35advsimd, float32x4_t)

// LCOV_EXCL_STOP
