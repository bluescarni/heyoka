// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/model/pendulum.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    {
        auto dyn = model::pendulum();

        REQUIRE(dyn.size() == 2u);

        REQUIRE(dyn[0].first == "x"_var);
        REQUIRE(dyn[0].second == "v"_var);

        REQUIRE(dyn[1].first == "v"_var);
        REQUIRE(dyn[1].second == -sin("x"_var));

        auto E = model::pendulum_energy();

        REQUIRE(E == .5 * pow("v"_var, 2_dbl) + (1. - cos("x"_var)));
    }

    {
        auto dyn = model::pendulum(kw::gconst = .1l, kw::length = .3);

        REQUIRE(dyn.size() == 2u);

        REQUIRE(dyn[0].first == "x"_var);
        REQUIRE(dyn[0].second == "v"_var);

        REQUIRE(dyn[1].first == "v"_var);
        REQUIRE(dyn[1].second == -(.1l / .3) * sin("x"_var));

        auto E = model::pendulum_energy(kw::gconst = .1l, kw::length = .3);

        REQUIRE(E == (.5 * .3 * .3) * pow("v"_var, 2_dbl) + (.1l * .3) * (1. - cos("x"_var)));
    }

    {
        auto dyn = model::pendulum(kw::gconst = .1l, kw::length = par[0]);

        REQUIRE(dyn.size() == 2u);

        REQUIRE(dyn[0].first == "x"_var);
        REQUIRE(dyn[0].second == "v"_var);

        REQUIRE(dyn[1].first == "v"_var);
        REQUIRE(dyn[1].second == (-.1l / par[0]) * sin("x"_var));

        auto E = model::pendulum_energy(kw::gconst = par[0], kw::length = .3);

        REQUIRE(E == (.5 * .3 * .3) * pow("v"_var, 2_dbl) + (par[0] * .3) * (1. - cos("x"_var)));
    }
}
