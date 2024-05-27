// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/var_ode_sys.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // Input args checking.
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime(x) = v, prime(v) = -x}, var_args::vars, 0), std::invalid_argument,
                           Message("The 'order' argument to the var_ode_sys constructor must be nonzero"));
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime("∂x"_var) = v, prime(v) = -"∂x"_var}, var_args::vars),
                           std::invalid_argument,
                           Message("Invalid state variable '∂x' detected: in a variational ODE system "
                                   "state variable names starting with '∂' are reserved"));
    REQUIRE_THROWS_MATCHES(
        var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector<expression>{}), std::invalid_argument,
        Message("Cannot formulate the variational equations with respect to an empty list of arguments"));
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{"z"_var}), std::invalid_argument,
                           Message("Cannot formulate the variational equations with respect to the "
                                   "initial conditions for the variable 'z', which is not among the state variables "
                                   "of the system"));
    REQUIRE_THROWS_MATCHES(
        var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x + x}), std::invalid_argument,
        Message("Cannot formulate the variational equations with respect to the expression '(x + x)': the "
                "expression is not a variable, not a parameter and not heyoka::time"));
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x, x}), std::invalid_argument,
                           Message("Duplicate entries detected in the list of expressions with respect to which the "
                                   "variational equations are to be formulated: [x, x]"));
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime(x) = v, prime(v) = -x}, var_args{0}), std::invalid_argument,
                           Message("Invalid var_args enumerator detected: the value of the enumerator "
                                   "must be in the [1, 7] range, but a value of 0 was detected instead"));
    REQUIRE_THROWS_MATCHES(var_ode_sys({prime(x) = v, prime(v) = -x}, var_args{8}), std::invalid_argument,
                           Message("Invalid var_args enumerator detected: the value of the enumerator "
                                   "must be in the [1, 7] range, but a value of 8 was detected instead"));

    // Check the deduction of variational args.
    auto vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, var_args::vars);
    REQUIRE(vsys.get_vargs() == std::vector{x, v});
    vsys = var_ode_sys({prime(v) = -x, prime(x) = v}, var_args::vars);
    REQUIRE(vsys.get_vargs() == std::vector{v, x});
    vsys = var_ode_sys({prime(v) = -x, prime(x) = v + par[2]}, var_args::params);
    REQUIRE(vsys.get_vargs() == std::vector{par[2]});
    vsys = var_ode_sys({prime(v) = -x, prime(x) = v + heyoka::time}, var_args::time);
    REQUIRE(vsys.get_vargs() == std::vector{heyoka::time});
    vsys = var_ode_sys({prime(v) = -x, prime(x) = v + heyoka::time}, var_args::vars | var_args::time);
    REQUIRE(vsys.get_vargs() == std::vector{v, x, heyoka::time});
    vsys = var_ode_sys({prime(v) = -x + par[2], prime(x) = v + heyoka::time}, var_args::vars | var_args::params);
    REQUIRE(vsys.get_vargs() == std::vector{v, x, par[2]});
    vsys = var_ode_sys({prime(v) = -x + par[2], prime(x) = v + heyoka::time}, var_args::vars | var_args::time);
    REQUIRE(vsys.get_vargs() == std::vector{v, x, heyoka::time});
    vsys = var_ode_sys({prime(v) = -x + par[2], prime(x) = v + heyoka::time}, var_args::params | var_args::time);
    REQUIRE(vsys.get_vargs() == std::vector{par[2], heyoka::time});
    vsys = var_ode_sys({prime(v) = -x + par[2], prime(x) = v + heyoka::time},
                       var_args::params | var_args::time | var_args::vars);
    REQUIRE(vsys.get_vargs() == std::vector{v, x, par[2], heyoka::time});
    vsys = var_ode_sys({prime(v) = -x + par[2], prime(x) = v + heyoka::time}, var_args::all);
    REQUIRE(vsys.get_vargs() == std::vector{v, x, par[2], heyoka::time});

    // Check explicit specification.
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x});
    REQUIRE(vsys.get_vargs() == std::vector{x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x, v});
    REQUIRE(vsys.get_vargs() == std::vector{x, v});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{v, x});
    REQUIRE(vsys.get_vargs() == std::vector{v, x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{par[2], v, x});
    REQUIRE(vsys.get_vargs() == std::vector{par[2], v, x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{par[2], v, heyoka::time, x});
    REQUIRE(vsys.get_vargs() == std::vector{par[2], v, heyoka::time, x});

    // Copy/move semantics.
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x});
    auto vsys2 = vsys;
    REQUIRE(vsys.get_sys() == vsys2.get_sys());
    REQUIRE(vsys.get_vargs() == vsys2.get_vargs());

    auto vsys3 = std::move(vsys2);
    REQUIRE(vsys.get_sys() == vsys3.get_sys());
    REQUIRE(vsys.get_vargs() == vsys3.get_vargs());

    // Revive vsys2 via copy assignment.
    vsys2 = vsys3;
    REQUIRE(vsys.get_sys() == vsys2.get_sys());
    REQUIRE(vsys.get_vargs() == vsys2.get_vargs());

    auto vsys4 = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x, v});
    vsys4 = std::move(vsys2);
    REQUIRE(vsys.get_sys() == vsys4.get_sys());
    REQUIRE(vsys.get_vargs() == vsys4.get_vargs());

    // Revive vsys2 via move assignment.
    vsys2 = std::move(vsys4);
    REQUIRE(vsys.get_sys() == vsys2.get_sys());
    REQUIRE(vsys.get_vargs() == vsys2.get_vargs());
}
