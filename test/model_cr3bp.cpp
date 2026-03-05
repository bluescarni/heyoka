// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/cr3bp.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    {
        auto dyn = model::cr3bp();

        REQUIRE(dyn.size() == 6u);
        REQUIRE(dyn[0].first == "x"_var);
        REQUIRE(dyn[1].first == "y"_var);
        REQUIRE(dyn[2].first == "z"_var);
        REQUIRE(dyn[3].first == "px"_var);
        REQUIRE(dyn[4].first == "py"_var);
        REQUIRE(dyn[5].first == "pz"_var);
    }

    // Energy conservation.
    {
        auto dyn = model::cr3bp();

        const std::vector init_state = {-0.45, 0.80, 0.00, -0.80, -0.45, 0.58};

        auto ta = taylor_adaptive{dyn, init_state};

        REQUIRE(ta.get_decomposition().size() == 35u);

        ta.propagate_until(20.);

        auto [x, y, z, px, py, pz] = make_vars("x", "y", "z", "px", "py", "pz");

        cfunc<double> cf({model::cr3bp_jacobi()}, {x, y, z, px, py, pz});

        REQUIRE(cf.get_dc().size() == 28u);

        std::vector<double> outs(1u, 0.);
        cf(outs, init_state);
        auto E0 = outs[0];

        cf(outs, ta.get_state());

        REQUIRE(outs[0] == approximately(E0));
    }

    // With custom mu.
    {
        auto dyn = model::cr3bp(kw::mu = 1e-2);

        const std::vector init_state = {-0.45, 0.80, 0.00, -0.80, -0.45, 0.58};

        auto ta = taylor_adaptive{dyn, init_state};

        REQUIRE(ta.get_decomposition().size() == 35u);

        ta.propagate_until(20.);

        auto [x, y, z, px, py, pz] = make_vars("x", "y", "z", "px", "py", "pz");

        cfunc<double> cf({model::cr3bp_jacobi(kw::mu = 1e-2)}, {x, y, z, px, py, pz});

        REQUIRE(cf.get_dc().size() == 28u);

        std::vector<double> outs(1u, 0.);
        cf(outs, init_state);
        auto E0 = outs[0];

        cf(outs, ta.get_state());

        REQUIRE(outs[0] == approximately(E0));
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::cr3bp(kw::mu = -1.), std::invalid_argument,
                           Message("The 'mu' parameter in a CR3BP must be in the range (0, "
                                   "0.5), but a value of -1 was provided instead"));
    REQUIRE_THROWS_MATCHES(model::cr3bp_jacobi(kw::mu = 1.), std::invalid_argument,
                           Message("The 'mu' parameter in a CR3BP must be in the range (0, "
                                   "0.5), but a value of 1 was provided instead"));
}
