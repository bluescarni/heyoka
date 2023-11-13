// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <cmath>
// #include <initializer_list>
// #include <iostream>
// #include <limits>
// #include <stdexcept>
// #include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/elp2000.hpp>
// #include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace heyoka::model;

TEST_CASE("basic")
{
    llvm_state s;

    auto dc = add_cfunc<double>(s, "lon", model::elp2000_spherical(kw::thresh = 1e-3));
    s.compile();

    fmt::println("{}\n\n", fmt::join(dc, "\n"));
}
