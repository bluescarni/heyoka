// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <random>

#include <heyoka/detail/splitmix64.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/gp.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace Catch::literals;

#include <iostream>

TEST_CASE("basic")
{
    std::random_device rd;
    detail::random_engine_type engine(rd());
    expression_generator generator({"x", "y"}, engine());
    auto ex = generator(2, 5);
    std::cout << "Random: " << ex << "\n";
    mutate(ex, generator, 0.1, engine);
    std::cout << "Mutated: " << ex << "\n";
    std::uniform_int_distribution<> node_target(0, count_nodes(ex) - 1u);
    auto ex2 = ex;
    extract_subtree(ex2, ex, node_target(engine));
    std::cout << "Extracted: " << ex2 << "\n";

    inject_subtree(ex, ex2, node_target(engine));
    std::cout << "Injected: " << ex << "\n";
}
