// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/ffnn.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("impl")
{
    auto linear = [](expression ret) -> expression { return ret; };
    auto [x] = make_vars("x");
    auto my_net = model::ffnn_impl({x}, 2, {2, 2}, {heyoka::tanh, heyoka::tanh, heyoka::tanh},
                                   {1_dbl, 2_dbl, 3_dbl, 4_dbl, 5_dbl, 6_dbl, 7_dbl, 8_dbl, 9_dbl, 0_dbl, 1_dbl, 2_dbl,
                                    3_dbl, 4_dbl, 5_dbl, 6_dbl});
}