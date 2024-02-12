// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>

#include <fmt/ranges.h>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Init the symbolic variables.
    auto [x, y] = make_vars("x", "y");

    // Create the symbolic function.
    auto sym_func = x * x - y * y;

    // Create the compiled function.
    cfunc<double> cf{{sym_func}, {x, y}};

    // Print the compiled function object to screen.
    fmt::println("{}", cf);

    // Prepare the input-output buffers.
    std::array<double, 2> in{1, 2};
    std::array<double, 1> out{};

    // Invoke the compiled function.
    cf(out, in);

    // Print the output.
    fmt::println("{}", out);
}
