// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

#include <iostream>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("vector expression")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        llvm_state s{""};

        s.add_vec_expression<double>("foo", x + y + z);

        std::cout << s.dump_ir() << '\n';

        s.compile();

        auto f = s.fetch_vec_expression<double>("foo");

        double args[] = {1, 2, 3};

        REQUIRE(f(args) == 6);
    }
    {
        llvm_state s{""};

        s.add_batch_expression<double>("foo", x + y, 4);

        std::cout << s.dump_ir() << '\n';
    }
}
