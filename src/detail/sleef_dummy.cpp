// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sleef.h>

extern "C" {

// NOTE: this function is here only to introduce a fake
// dependency of heyoka on libsleef. This is needed
// in certain setups that would otherwise unlink libsleef
// from heyoka after detecting that no symbol from libsleef
// is being used (even though we need libsleef to be linked
// in for use in the JIT machinery).
#if defined(_WIN32) || defined(__CYGWIN__)
__declspec(dllexport)
#elif defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)
__attribute__((visibility("default")))
#endif
    double heyoka_sleef_dummy_f(double x)
{
    return ::Sleef_sin_u10(x);
}
}
