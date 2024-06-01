// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

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
    REQUIRE(vsys.get_n_orig_sv() == 2u);
    REQUIRE(vsys.get_order() == 1u);
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
    REQUIRE(vsys.get_n_orig_sv() == 2u);
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x, v});
    REQUIRE(vsys.get_vargs() == std::vector{x, v});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{v, x});
    REQUIRE(vsys.get_vargs() == std::vector{v, x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{par[2], v, x});
    REQUIRE(vsys.get_vargs() == std::vector{par[2], v, x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{par[2], v, heyoka::time, x});
    REQUIRE(vsys.get_vargs() == std::vector{par[2], v, heyoka::time, x});
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, {par[2], v, heyoka::time, x});
    REQUIRE(vsys.get_vargs() == std::vector{par[2], v, heyoka::time, x});

    // Copy/move semantics.
    vsys = var_ode_sys({prime(x) = v, prime(v) = -x}, std::vector{x});
    auto vsys2 = vsys;
    REQUIRE(vsys.get_sys() == vsys2.get_sys());
    REQUIRE(vsys.get_vargs() == vsys2.get_vargs());
    REQUIRE(vsys2.get_n_orig_sv() == 2u);

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

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, v] = make_vars("x", "v");

    auto sys = var_ode_sys({prime(x) = v, prime(v) = -x}, var_args::vars);
    auto sys_copy(sys);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << sys;
    }

    sys = var_ode_sys({prime(x) = x}, std::vector{x});

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> sys;
    }

    REQUIRE(sys.get_sys() == sys_copy.get_sys());
    REQUIRE(sys.get_vargs() == sys_copy.get_vargs());
    REQUIRE(sys.get_n_orig_sv() == 2u);
}

TEST_CASE("vareqs")
{
    auto [x, v] = make_vars("x", "v");

    {
        auto orig_sys = std::vector{prime(x) = v + x, prime(v) = -sin(x * v)};
        auto vsys = var_ode_sys(orig_sys, var_args::vars);
        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 6u);
        REQUIRE(vsys.get_order() == 1u);

        auto [x_x0, x_v0, v_x0, v_v0] = make_vars("∂[(0, 1)]x", "∂[(1, 1)]x", "∂[(0, 1)]v", "∂[(1, 1)]v");

        REQUIRE(vsys.get_sys()[2].first == x_x0);
        REQUIRE(vsys.get_sys()[2].second == diff(orig_sys[0].second, v) * v_x0 + diff(orig_sys[0].second, x) * x_x0);

        REQUIRE(vsys.get_sys()[3].first == x_v0);
        REQUIRE(vsys.get_sys()[3].second == diff(orig_sys[0].second, v) * v_v0 + diff(orig_sys[0].second, x) * x_v0);

        REQUIRE(vsys.get_sys()[4].first == v_x0);
        REQUIRE(vsys.get_sys()[4].second == -(((v_x0 * x) + (x_x0 * v)) * cos((x * v))));

        REQUIRE(vsys.get_sys()[5].first == v_v0);
        REQUIRE(vsys.get_sys()[5].second == -(((v_v0 * x) + (x_v0 * v)) * cos((x * v))));

        auto ta = taylor_adaptive<double>{vsys.get_sys(), std::vector<double>(6, 0.), kw::tol = 1e-3};
    }

    {
        auto orig_sys = std::vector{prime(x) = v + x, prime(v) = -sin(x * v)};
        auto vsys = var_ode_sys(orig_sys, std::vector{v});
        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 4u);
        REQUIRE(vsys.get_order() == 1u);

        auto [x_v0, v_v0] = make_vars("∂[(0, 1)]x", "∂[(0, 1)]v");

        REQUIRE(vsys.get_sys()[2].first == x_v0);
        REQUIRE(vsys.get_sys()[2].second == diff(orig_sys[0].second, v) * v_v0 + diff(orig_sys[0].second, x) * x_v0);

        REQUIRE(vsys.get_sys()[3].first == v_v0);
        REQUIRE(vsys.get_sys()[3].second == -(((v_v0 * x) + (x_v0 * v)) * cos((x * v))));

        auto ta = taylor_adaptive<double>{vsys.get_sys(), std::vector<double>(4, 0.), kw::tol = 1e-3};
    }

    {
        auto orig_sys = std::vector{prime(x) = v + x, prime(v) = -sin(x * v)};
        auto vsys = var_ode_sys(orig_sys, var_args::vars, 2);
        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 12u);
        REQUIRE(vsys.get_order() == 2u);

        auto [x_x0, x_v0, v_x0, v_v0] = make_vars("∂[(0, 1)]x", "∂[(1, 1)]x", "∂[(0, 1)]v", "∂[(1, 1)]v");
        auto [x_x0_x0, x_x0_v0] = make_vars("∂[(0, 2)]x", "∂[(0, 1), (1, 1)]x");
        auto x_v0_v0 = make_vars("∂[(1, 2)]x");
        auto [v_x0_x0, v_x0_v0] = make_vars("∂[(0, 2)]v", "∂[(0, 1), (1, 1)]v");
        auto v_v0_v0 = make_vars("∂[(1, 2)]v");

        REQUIRE(vsys.get_sys()[2].first == x_x0);
        REQUIRE(vsys.get_sys()[2].second == diff(orig_sys[0].second, v) * v_x0 + diff(orig_sys[0].second, x) * x_x0);

        REQUIRE(vsys.get_sys()[3].first == x_v0);
        REQUIRE(vsys.get_sys()[3].second == diff(orig_sys[0].second, v) * v_v0 + diff(orig_sys[0].second, x) * x_v0);

        REQUIRE(vsys.get_sys()[4].first == v_x0);
        REQUIRE(vsys.get_sys()[4].second == -(((v_x0 * x) + (x_x0 * v)) * cos((x * v))));

        REQUIRE(vsys.get_sys()[5].first == v_v0);
        REQUIRE(vsys.get_sys()[5].second == -(((v_v0 * x) + (x_v0 * v)) * cos((x * v))));

        REQUIRE(vsys.get_sys()[6].first == x_x0_x0);
        REQUIRE(vsys.get_sys()[6].second == v_x0_x0 + x_x0_x0);

        REQUIRE(vsys.get_sys()[7].first == x_x0_v0);
        REQUIRE(vsys.get_sys()[7].second == v_x0_v0 + x_x0_v0);

        REQUIRE(vsys.get_sys()[8].first == x_v0_v0);
        REQUIRE(vsys.get_sys()[8].second == v_v0_v0 + x_v0_v0);

        REQUIRE(vsys.get_sys()[9].first == v_x0_x0);
        REQUIRE(vsys.get_sys()[9].second
                == -(((((v_x0_x0 * x) + (x_x0 * v_x0)) + ((x_x0_x0 * v) + (v_x0 * x_x0))) * cos((x * v)))
                     + ((((x_x0 * v) + (v_x0 * x)) * -sin((x * v))) * ((v_x0 * x) + (x_x0 * v)))));

        REQUIRE(vsys.get_sys()[10].first == v_x0_v0);
        REQUIRE(vsys.get_sys()[10].second
                == -(((((x_x0 * v) + (v_x0 * x)) * -sin((x * v))) * ((v_v0 * x) + (x_v0 * v)))
                     + ((((v_x0_v0 * x) + (x_x0 * v_v0)) + ((x_x0_v0 * v) + (v_x0 * x_v0))) * cos((x * v)))));

        REQUIRE(vsys.get_sys()[11].first == v_v0_v0);
        REQUIRE(vsys.get_sys()[11].second
                == -(((((x_v0 * v) + (v_v0 * x)) * -sin((x * v))) * ((v_v0 * x) + (x_v0 * v)))
                     + ((((v_v0_v0 * x) + (x_v0 * v_v0)) + ((x_v0_v0 * v) + (v_v0 * x_v0))) * cos((x * v)))));

        auto ta = taylor_adaptive<double>{vsys.get_sys(), std::vector<double>(12, 0.), kw::tol = 1e-3};
    }

    {
        auto orig_sys = std::vector{prime(x) = v + x, prime(v) = -sin(x * v)};
        auto vsys = var_ode_sys(orig_sys, std::vector{v}, 2);
        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 6u);

        auto [x_v0, v_v0] = make_vars("∂[(0, 1)]x", "∂[(0, 1)]v");
        auto [x_v0_v0, v_v0_v0] = make_vars("∂[(0, 2)]x", "∂[(0, 2)]v");

        REQUIRE(vsys.get_sys()[2].first == x_v0);
        REQUIRE(vsys.get_sys()[2].second == diff(orig_sys[0].second, v) * v_v0 + diff(orig_sys[0].second, x) * x_v0);

        REQUIRE(vsys.get_sys()[3].first == v_v0);
        REQUIRE(vsys.get_sys()[3].second == -(((v_v0 * x) + (x_v0 * v)) * cos((x * v))));

        REQUIRE(vsys.get_sys()[4].first == x_v0_v0);
        REQUIRE(vsys.get_sys()[4].second == v_v0_v0 + x_v0_v0);

        REQUIRE(vsys.get_sys()[5].first == v_v0_v0);
        REQUIRE(vsys.get_sys()[5].second
                == -(((((x_v0 * v) + (v_v0 * x)) * -sin((x * v))) * ((v_v0 * x) + (x_v0 * v)))
                     + ((((v_v0_v0 * x) + (x_v0 * v_v0)) + ((x_v0_v0 * v) + (v_v0 * x_v0))) * cos((x * v)))));

        auto ta = taylor_adaptive<double>{vsys.get_sys(), std::vector<double>(6, 0.), kw::tol = 1e-3};
    }

    // Test with params.
    {
        auto orig_sys = std::vector{prime(x) = v + x + par[2], prime(v) = -sin(x * v * par[2])};
        auto vsys = var_ode_sys(orig_sys, std::vector{v, par[2]});

        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 6u);

        auto [x_v0, x_p2, v_v0, v_p2] = make_vars("∂[(0, 1)]x", "∂[(1, 1)]x", "∂[(0, 1)]v", "∂[(1, 1)]v");

        REQUIRE(vsys.get_sys()[2].first == x_v0);
        REQUIRE(vsys.get_sys()[2].second == (v_v0 + x_v0));

        REQUIRE(vsys.get_sys()[3].first == x_p2);
        REQUIRE(vsys.get_sys()[3].second == 1_dbl + (v_p2 + x_p2));

        REQUIRE(vsys.get_sys()[4].first == v_v0);
        REQUIRE(vsys.get_sys()[4].second == -((((v_v0 * x) + (x_v0 * v)) * par[2]) * cos(((x * v) * par[2]))));

        REQUIRE(vsys.get_sys()[5].first == v_p2);
        REQUIRE(vsys.get_sys()[5].second
                == -(((x * v) + (((v_p2 * x) + (x_p2 * v)) * par[2])) * cos(((x * v) * par[2]))));

        auto ta = taylor_adaptive<double>{vsys.get_sys(), std::vector<double>(6, 0.), kw::tol = 1e-3};
    }

    // Test with time.
    {
        auto orig_sys = std::vector{prime(x) = v + x + par[2], prime(v) = -sin(x * v * par[2])};
        auto vsys = var_ode_sys(orig_sys, std::vector{heyoka::time, par[2]});

        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 6u);

        auto [x_t0, x_p2, v_t0, v_p2] = make_vars("∂[(0, 1)]x", "∂[(1, 1)]x", "∂[(0, 1)]v", "∂[(1, 1)]v");

        REQUIRE(vsys.get_sys()[2].first == x_t0);
        REQUIRE(vsys.get_sys()[2].second == (v_t0 + x_t0));

        REQUIRE(vsys.get_sys()[3].first == x_p2);
        REQUIRE(vsys.get_sys()[3].second == 1_dbl + (v_p2 + x_p2));

        REQUIRE(vsys.get_sys()[4].first == v_t0);
        REQUIRE(vsys.get_sys()[4].second == -((((v_t0 * x) + (x_t0 * v)) * par[2]) * cos(((x * v) * par[2]))));

        REQUIRE(vsys.get_sys()[5].first == v_p2);
        REQUIRE(vsys.get_sys()[5].second
                == -(((x * v) + (((v_p2 * x) + (x_p2 * v)) * par[2])) * cos(((x * v) * par[2]))));

        auto ta = taylor_adaptive<double>{
            vsys.get_sys(), {.1, .2, 0., 0., 0., 0.}, kw::pars = {1., 1., .3}, kw::tol = 1e-3};

        ta.step();

        REQUIRE(ta.get_state()[2] == 0.);
        REQUIRE(ta.get_state()[4] == 0.);
    }

    // Test with par not showing up in the dynamics.
    {
        auto orig_sys = std::vector{prime(x) = v + x + par[2], prime(v) = -sin(x * v * par[2])};
        auto vsys = var_ode_sys(orig_sys, std::vector{v, par[3]});

        REQUIRE(
            std::ranges::equal(orig_sys, std::ranges::subrange(vsys.get_sys().begin(), vsys.get_sys().begin() + 2)));
        REQUIRE(vsys.get_sys().size() == 6u);

        auto [x_v0, x_p3, v_v0, v_p3] = make_vars("∂[(0, 1)]x", "∂[(1, 1)]x", "∂[(0, 1)]v", "∂[(1, 1)]v");

        REQUIRE(vsys.get_sys()[2].first == x_v0);
        REQUIRE(vsys.get_sys()[2].second == (v_v0 + x_v0));

        REQUIRE(vsys.get_sys()[3].first == x_p3);
        REQUIRE(vsys.get_sys()[3].second == v_p3 + x_p3);

        REQUIRE(vsys.get_sys()[4].first == v_v0);
        REQUIRE(vsys.get_sys()[4].second == -((((v_v0 * x) + (x_v0 * v)) * par[2]) * cos(((x * v) * par[2]))));

        REQUIRE(vsys.get_sys()[5].first == v_p3);
        REQUIRE(vsys.get_sys()[5].second == -((((v_p3 * x) + (x_p3 * v)) * par[2]) * cos(((x * v) * par[2]))));

        auto ta = taylor_adaptive<double>{
            vsys.get_sys(), {.1, .2, 0., 0., 1., 0.}, kw::pars = {1., 1., .3}, kw::tol = 1e-3};

        ta.step();

        REQUIRE(ta.get_state()[3] == 0.);
        REQUIRE(ta.get_state()[5] == 0.);
    }
}

// Test on a non-autonomous system with parameters.
TEST_CASE("nonauto sys")
{
    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    // The variational ODEs.
    auto vsys = var_ode_sys(orig_sys, var_args::all);

    REQUIRE(vsys.get_sys().size() == 10u);

    const auto ic_x = .2, ic_v = .3, ic_tm = .5, ic_par = .4;

    auto ta = taylor_adaptive<double>{vsys.get_sys(),
                                      {ic_x, ic_v,
                                       // dx/...
                                       1., 0., 0., -ic_v,
                                       // dv/...
                                       0., 1., 0., -(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))},
                                      kw::pars = {ic_par},
                                      kw::time = ic_tm};

    ta.propagate_until(3.);

    // First try: variation on the ics of the state variables.
    const auto delta_x = 1e-8, delta_v = 2e-8;
    auto ta_orig
        = taylor_adaptive<double>{orig_sys, {ic_x + delta_x, ic_v + delta_v}, kw::pars = {ic_par}, kw::time = ic_tm};

    ta_orig.propagate_until(3.);

    REQUIRE(ta_orig.get_state()[0]
            == approximately(ta.get_state()[0] + ta.get_state()[2] * delta_x + ta.get_state()[3] * delta_v));
    REQUIRE(ta_orig.get_state()[1]
            == approximately(ta.get_state()[1] + ta.get_state()[6] * delta_x + ta.get_state()[7] * delta_v));

    // Second try: variation on the ic of x and the parameter.
    const auto delta_par = 3e-8;

    ta_orig
        = taylor_adaptive<double>{orig_sys, {ic_x + delta_x, ic_v}, kw::pars = {ic_par + delta_par}, kw::time = ic_tm};

    ta_orig.propagate_until(3.);

    REQUIRE(ta_orig.get_state()[0]
            == approximately(ta.get_state()[0] + ta.get_state()[2] * delta_x + ta.get_state()[4] * delta_par));
    REQUIRE(ta_orig.get_state()[1]
            == approximately(ta.get_state()[1] + ta.get_state()[6] * delta_x + ta.get_state()[8] * delta_par));

    // Third try: variation on the ic of the parameter and time.
    const auto delta_tm = 4e-8;

    ta_orig
        = taylor_adaptive<double>{orig_sys, {ic_x, ic_v}, kw::pars = {ic_par + delta_par}, kw::time = ic_tm + delta_tm};

    ta_orig.propagate_until(3.);

    REQUIRE(ta_orig.get_state()[0]
            == approximately(ta.get_state()[0] + ta.get_state()[4] * delta_par + ta.get_state()[5] * delta_tm));
    REQUIRE(ta_orig.get_state()[1]
            == approximately(ta.get_state()[1] + ta.get_state()[8] * delta_par + ta.get_state()[9] * delta_tm));
}

// An order-2 test.
TEST_CASE("nonauto sys order 2")
{
    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    // The variational ODEs.
    auto vsys = var_ode_sys(orig_sys, {x, par[0]}, 2);

    REQUIRE(vsys.get_sys().size() == 12u);

    const auto ic_x = .2, ic_v = .3, ic_tm = .5, ic_par = .4;

    auto ta = taylor_adaptive<double>{vsys.get_sys(),
                                      {ic_x, ic_v,
                                       // dx/...
                                       1., 0.,
                                       // dv/...
                                       0., 0.,
                                       // d2x/...
                                       0., 0., 0.,
                                       // d2v/...
                                       0., 0., 0.},
                                      kw::pars = {ic_par},
                                      kw::time = ic_tm};

    ta.propagate_until(3.);

    const auto delta_x = 1e-5;
    const auto delta_par = 2e-5;
    auto ta_orig
        = taylor_adaptive<double>{orig_sys, {ic_x + delta_x, ic_v}, kw::pars = {ic_par + delta_par}, kw::time = ic_tm};

    ta_orig.propagate_until(3.);

    REQUIRE(ta_orig.get_state()[0]
            == approximately(ta.get_state()[0] + ta.get_state()[2] * delta_x + ta.get_state()[3] * delta_par
                                 + 0.5
                                       * (ta.get_state()[6] * delta_x * delta_x
                                          + 2 * ta.get_state()[7] * delta_x * delta_par
                                          + ta.get_state()[8] * delta_par * delta_par),
                             1000.));

    REQUIRE(ta_orig.get_state()[1]
            == approximately(ta.get_state()[1] + ta.get_state()[4] * delta_x + ta.get_state()[5] * delta_par
                                 + 0.5
                                       * (ta.get_state()[9] * delta_x * delta_x
                                          + 2 * ta.get_state()[10] * delta_x * delta_par
                                          + ta.get_state()[11] * delta_par * delta_par),
                             1000.));
}

// An order-2 test wrt initial time.
TEST_CASE("nonauto sys order 2 t0")
{
    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    // The variational ODEs.
    auto vsys = var_ode_sys(orig_sys, {heyoka::time}, 2);

    REQUIRE(vsys.get_sys().size() == 6u);

    const auto ic_x = .2, ic_v = .3, ic_tm = .5, ic_par = .4;

    const auto acc0 = std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x);
    const auto jerk0 = -std::sin(ic_tm) - ic_par * acc0 - std::cos(ic_x) * ic_v;
    const auto dxdt0 = -ic_v;
    const auto dvdt0 = -acc0;

    auto ta = taylor_adaptive<double>{vsys.get_sys(),
                                      {ic_x, ic_v,
                                       // dx/...
                                       dxdt0,
                                       // dv/...
                                       dvdt0,
                                       // d2x/...
                                       -acc0 - 2 * dvdt0,
                                       // d2v/...
                                       -jerk0 - 2 * (-ic_par * dvdt0 - std::cos(ic_x) * dxdt0)},
                                      kw::pars = {ic_par},
                                      kw::time = ic_tm};

    ta.propagate_until(3.);

    const auto delta_t0 = 1e-5;
    auto ta_orig = taylor_adaptive<double>{orig_sys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm + delta_t0};

    ta_orig.propagate_until(3.);

    REQUIRE(ta_orig.get_state()[0]
            == approximately(ta.get_state()[0] + ta.get_state()[2] * delta_t0
                                 + 0.5 * ta.get_state()[4] * delta_t0 * delta_t0,
                             1000.));
    REQUIRE(ta_orig.get_state()[1]
            == approximately(ta.get_state()[1] + ta.get_state()[3] * delta_t0
                                 + 0.5 * ta.get_state()[5] * delta_t0 * delta_t0,
                             1000.));
}
