// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

#include "catch.hpp"

using namespace heyoka;

#include <iostream>

TEST_CASE("basic")
{
    std::cout << (45_dbl + "x"_var) / -1_dbl << '\n';
}
