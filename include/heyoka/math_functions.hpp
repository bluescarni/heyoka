// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_FUNCTIONS_HPP
#define HEYOKA_MATH_FUNCTIONS_HPP

#include <heyoka/config.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>

namespace heyoka
{

HEYOKA_DLL_PUBLIC expression sin(expression);
HEYOKA_DLL_PUBLIC expression cos(expression);
HEYOKA_DLL_PUBLIC expression log(expression);
HEYOKA_DLL_PUBLIC expression exp(expression);
HEYOKA_DLL_PUBLIC expression sqrt(expression);

HEYOKA_DLL_PUBLIC expression pow(expression, expression);
HEYOKA_DLL_PUBLIC expression pow(expression, double);
HEYOKA_DLL_PUBLIC expression pow(expression, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression pow(expression, mppp::real128);
#endif

} // namespace heyoka

#endif
