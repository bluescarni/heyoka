// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create multiple symbolic variables in one go.
    auto [x, y] = make_vars("x", "y");

    // Another way of creating symbolic variables.
    auto z = "z"_var;

    // Create and print to screen a mathematical expression.
    std::cout << "The euclidean distance is: " << sqrt(x * x + y * y + z * z) << '\n';

    // Create and print to screen a few constants using
    // different floating-point precisions.

    // Double precision.
    std::cout << 1.1_dbl << '\n'; // Prints '1.1000000000000001'

    // Extended (80-bit) precision.
    std::cout << 1.1_ldbl << '\n'; // Prints '1.10000000000000000002'

#if defined(HEYOKA_HAVE_REAL128)
    // Quadruple precision.
    std::cout << 1.1_f128 << '\n'; // Prints '1.10000000000000000000000000000000008'
#endif

    // An expression involving a few elementary functions.
    std::cout << cos(x + 2_dbl * y) * sqrt(z) - exp(x) << '\n';
}
