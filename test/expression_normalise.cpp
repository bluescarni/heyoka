// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <variant>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("normalise")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(normalise(x) == x);

    auto tmp = sin(sin(sin(sin(x))));
    REQUIRE(normalise(subs(tmp, {{x, .1_dbl}})) == sin(sin(sin(sin(.1_dbl)))));

    tmp = fix(x + y) + z;
    REQUIRE(normalise(unfix(tmp)) == x + y + z);

    tmp = subs(z * (x + y), {{z, 2_dbl}});

    auto ret = normalise({sin(tmp), cos(tmp)});

    REQUIRE(ret.size() == 2u);
    REQUIRE(ret[0] == sin(2. * x + 2. * y));
    REQUIRE(ret[1] == cos(2. * x + 2. * y));

    REQUIRE(std::get<func>(std::get<func>(ret[0].value()).args()[0].value()).get_ptr()
            == std::get<func>(std::get<func>(ret[1].value()).args()[0].value()).get_ptr());

    tmp = x + heyoka::time;
    REQUIRE(normalise(tmp) == tmp);
}
