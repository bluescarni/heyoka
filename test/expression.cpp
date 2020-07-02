// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/binary_operator.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;

#include <iostream>

TEST_CASE("basic")
{
    auto x = "x"_var, y = "y"_var;

    // auto d = taylor_decompose({(1_dbl + x) + (1_dbl + x)});
    // auto d = taylor_decompose({expression{binary_operator{binary_operator::type::add, 1_dbl, 1_dbl}}, x + y *
    // sin(x)});
    auto d = taylor_decompose({x * (1_dbl + x), x * (1_dbl + cos(y))});

    for (const auto &ex : d) {
        std::cout << ex << '\n';
    }

#if 0
    auto ex = sin("x"_var) + 1.1_ldbl;

    llvm_state s{"pippo"};

    s.add_ldbl("f", ex);

    std::cout << s.dump() << '\n';

    s.compile();

    auto f = s.fetch_ldbl<1>("f");

    std::cout << f(5) << '\n';
#endif
}
