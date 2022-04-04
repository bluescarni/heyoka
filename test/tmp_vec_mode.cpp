// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;

TEST_CASE("foo")
{
    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    taylor_adaptive<double> ta{{prime(x) = x + y, prime(y) = y + z, prime(z) = z + t, prime(t) = t + x},
                               {0., 0., 0., 0.},
                               kw::compact_mode = true,
                               kw::opt_level = 3u};

    std::cout << ta.get_llvm_state().get_ir() << '\n';

    ta.propagate_until(5.);

    std::cout << ta << '\n';
}
