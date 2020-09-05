// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    std::cout << llvm_state{kw::mname = "sample state"} << '\n';

    llvm_state s;
    auto [x, y] = make_vars("x", "y");
    taylor_add_jet_dbl(s, "foo", {prime(x) = y, prime(y) = (1_dbl - x * x) * y - x}, 20, 21, true);

    std::cout << s.get_ir() << '\n';
}
