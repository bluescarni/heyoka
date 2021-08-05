// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    auto x = "x"_var;

    auto merc_a = vsop2013_elliptic(1, 1, kw::vsop2013_time = par[0]);

    auto ta = taylor_adaptive<double>{{prime(x) = merc_a}, {0.}, kw::compact_mode = true};

    ta.get_pars_data()[0] = (2447545.0 - 2451545.0) / 365250;

    ta.propagate_until(1);
}
