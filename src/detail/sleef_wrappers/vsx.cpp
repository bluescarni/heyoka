// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sleef_wrappers_helpers.hpp"

// NOTE: on PPC64, the VSX vector types are "__vector double" and "__vector float".

// Double-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u10vsx, __vector double)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosd2_u35vsx, __vector double)

// Single-precision.
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u10vsx, __vector float)
HEYOKA_SLEEF_PAIR_WRAPPER(Sleef_sincosf4_u35vsx, __vector float)
