// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/callable.hpp>

#include "catch.hpp"

using namespace heyoka;

void blap() {}

int blop(int n)
{
    return n + 1;
}

TEST_CASE("callable basics")
{
    callable<void()> f0(blap);
    callable<int(int)> f1(blop);
}
