// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/gp.hpp>
#include <heyoka/detail/splitmix64.hpp>
#include <random>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

TEST_CASE("basic")
{
    random_expression random_e({"x", "y"}, 3u);
    heyoka::detail::splitmix64 engine(354u);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (auto i = 0u; i< 100; ++i) {
        std::cout << dis(engine) << std::endl;
    }
}
