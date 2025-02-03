// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
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

TEST_CASE("auto ic setup")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    // IC test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::vars, 2);

        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{x, v});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .3);
        // dx/...
        REQUIRE(ta.get_state()[2] == 1);
        REQUIRE(ta.get_state()[3] == 0);
        // dv/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 1);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 6, ta.get_state().end(), [](const auto &val) { return val == 0; }));

        REQUIRE(ta.get_sys().size() == 12u);

        REQUIRE(ta.get_vorder() == 2u);

        std::ostringstream oss;
        oss << ta;
        REQUIRE(boost::contains(oss.str(), "Variational order"));
    }

    // IC test, swap the variables around.
    {
        auto vsys = var_ode_sys(orig_sys, {v, x}, 2);

        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{v, x});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .3);
        // dx/...
        REQUIRE(ta.get_state()[2] == 0);
        REQUIRE(ta.get_state()[3] == 1);
        // dv/...
        REQUIRE(ta.get_state()[4] == 1);
        REQUIRE(ta.get_state()[5] == 0);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 6, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // par test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::params, 2);

        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{par[0]});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .3);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 2, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // par+var test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::params | var_args::vars, 2);

        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::tol = 1e-3};

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .3);
        // dx/...
        REQUIRE(ta.get_state()[2] == 1);
        REQUIRE(ta.get_state()[3] == 0);
        REQUIRE(ta.get_state()[4] == 0);
        // dv/...
        REQUIRE(ta.get_state()[5] == 0);
        REQUIRE(ta.get_state()[6] == 1);
        REQUIRE(ta.get_state()[7] == 0);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 8, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // Single par, single var, mixed up order.
    {
        auto vsys = var_ode_sys(orig_sys, {par[0], v}, 2);

        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::tol = 1e-3};

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .3);
        // dx/...
        REQUIRE(ta.get_state()[2] == 0);
        REQUIRE(ta.get_state()[3] == 0);
        // dv/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 1);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 6, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // Invalid state size passed to the ctor.
    {
        auto vsys = var_ode_sys(orig_sys, {par[0], v}, 2);

        REQUIRE_THROWS_MATCHES(
            taylor_adaptive(vsys, {.2, .3, .4}, kw::tol = 1e-3), std::invalid_argument,
            Message("Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator: the state vector has a dimension of 3, while the total number of equations is 12. "
                    "The size of the state vector must be equal either to the total number of equations, or to the "
                    "number of original (i.e., non-variational) equations, which for this system is 2"));
        REQUIRE_THROWS_MATCHES(
            taylor_adaptive(vsys,
                            {
                                .2,
                            },
                            kw::tol = 1e-3),
            std::invalid_argument,
            Message("Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator: the state vector has a dimension of 1, while the total number of equations is 12. "
                    "The size of the state vector must be equal either to the total number of equations, or to the "
                    "number of original (i.e., non-variational) equations, which for this system is 2"));
    }

#if defined(HEYOKA_HAVE_REAL)

    {
        auto vsys = var_ode_sys(orig_sys, {par[0], v}, 2);

        const auto prec = 13;

        auto ta = taylor_adaptive{vsys, {mppp::real{.2, prec}, mppp::real{.3, prec}}, kw::tol = 1e-3};

        REQUIRE(ta.get_state()[0] == mppp::real{.2, prec});
        REQUIRE(ta.get_state()[1] == mppp::real{.3, prec});
        // dx/...
        REQUIRE(ta.get_state()[2] == 0);
        REQUIRE(ta.get_state()[3] == 0);
        // dv/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 1);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 6, ta.get_state().end(), [](const auto &val) { return val == 0; }));

        REQUIRE(std::ranges::all_of(ta.get_state(), [prec](const auto &r) { return r.get_prec() == prec; }));
    }

#endif

    // A couple of tests for derivatives wrt the initial time.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::time, 1);

        const auto ic_x = .2, ic_v = .3, ic_par = .4, ic_tm = .5;

        auto ta = taylor_adaptive{vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm, kw::tol = 1e-3};

        REQUIRE(ta.get_state().size() == 4u);

        REQUIRE(ta.get_state()[0] == ic_x);
        REQUIRE(ta.get_state()[1] == ic_v);
        // dx/...
        REQUIRE(ta.get_state()[2] == -ic_v);
        // dv/...
        REQUIRE(ta.get_state()[3] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
    }

    // Custom ordering for vars and time derivatives.
    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, x}, 1);

        const auto ic_x = .2, ic_v = .3, ic_par = .4, ic_tm = .5;

        auto ta = taylor_adaptive{vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm, kw::tol = 1e-3};

        REQUIRE(ta.get_state().size() == 8u);

        REQUIRE(ta.get_state()[0] == ic_x);
        REQUIRE(ta.get_state()[1] == ic_v);
        // dx/...
        REQUIRE(ta.get_state()[2] == 0);
        REQUIRE(ta.get_state()[3] == -ic_v);
        REQUIRE(ta.get_state()[4] == 1);
        // dv/...
        REQUIRE(ta.get_state()[5] == 1);
        REQUIRE(ta.get_state()[6] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
        REQUIRE(ta.get_state()[7] == 0);
    }

    // Order greater than 1 not supported.
    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, x}, 2);

        const auto ic_x = .2, ic_v = .3, ic_par = .4, ic_tm = .5;

        REQUIRE_THROWS_MATCHES(
            taylor_adaptive(vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm, kw::tol = 1e-3),
            std::invalid_argument,
            Message(
                "In a variational integrator the automatic setup of the initial conditions for the derivatives with "
                "respect to the initial time are currently supported only at the first order, but an order of 2 was "
                "specified instead"));
    }

#if defined(HEYOKA_HAVE_REAL)

    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, x}, 1);

        const auto prec = 13;

        const auto ic_x = mppp::real{.2, prec}, ic_v = mppp::real{.3, prec}, ic_par = mppp::real{.4, prec},
                   ic_tm = mppp::real{.5, prec};

        auto ta = taylor_adaptive{vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm, kw::tol = 1e-3};

        REQUIRE(ta.get_state().size() == 8u);

        REQUIRE(ta.get_state()[0] == ic_x);
        REQUIRE(ta.get_state()[1] == ic_v);
        // dx/...
        REQUIRE(ta.get_state()[2] == 0);
        REQUIRE(ta.get_state()[3] == -ic_v);
        REQUIRE(ta.get_state()[4] == 1);
        // dv/...
        REQUIRE(ta.get_state()[5] == 1);
        REQUIRE(ta.get_state()[6] == approximately(-(cos(ic_tm) - ic_par * ic_v - sin(ic_x))));
        REQUIRE(ta.get_state()[7] == 0);

        REQUIRE(std::ranges::all_of(ta.get_state(), [prec](const auto &r) { return r.get_prec() == prec; }));
    }

#endif
}

TEST_CASE("auto ic setup batch")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    // IC test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::vars, 2);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{x, v});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .21);
        REQUIRE(ta.get_state()[2] == .3);
        REQUIRE(ta.get_state()[3] == .31);
        // dx/...
        REQUIRE(ta.get_state()[4] == 1);
        REQUIRE(ta.get_state()[5] == 1);
        REQUIRE(ta.get_state()[6] == 0);
        REQUIRE(ta.get_state()[7] == 0);
        // dv/...
        REQUIRE(ta.get_state()[8] == 0);
        REQUIRE(ta.get_state()[9] == 0);
        REQUIRE(ta.get_state()[10] == 1);
        REQUIRE(ta.get_state()[11] == 1);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 12, ta.get_state().end(), [](const auto &val) { return val == 0; }));

        REQUIRE(ta.get_sys().size() == 12u);

        REQUIRE(ta.get_vorder() == 2u);

        std::ostringstream oss;
        oss << ta;
        REQUIRE(boost::contains(oss.str(), "Variational order"));
    }

    // IC test, swap the variables around.
    {
        auto vsys = var_ode_sys(orig_sys, {v, x}, 2);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{v, x});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .21);
        REQUIRE(ta.get_state()[2] == .3);
        REQUIRE(ta.get_state()[3] == .31);
        // dx/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 0);
        REQUIRE(ta.get_state()[6] == 1);
        REQUIRE(ta.get_state()[7] == 1);
        // dv/...
        REQUIRE(ta.get_state()[8] == 1);
        REQUIRE(ta.get_state()[9] == 1);
        REQUIRE(ta.get_state()[10] == 0);
        REQUIRE(ta.get_state()[11] == 0);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 12, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // par test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::params, 2);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::tol = 1e-3};

        REQUIRE(ta.get_vargs() == std::vector{par[0]});

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .21);
        REQUIRE(ta.get_state()[2] == .3);
        REQUIRE(ta.get_state()[3] == .31);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 4, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // par+var test.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::params | var_args::vars, 2);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::tol = 1e-3};

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .21);
        REQUIRE(ta.get_state()[2] == .3);
        REQUIRE(ta.get_state()[3] == .31);
        // dx/...
        REQUIRE(ta.get_state()[4] == 1);
        REQUIRE(ta.get_state()[5] == 1);
        REQUIRE(ta.get_state()[6] == 0);
        REQUIRE(ta.get_state()[7] == 0);
        REQUIRE(ta.get_state()[8] == 0);
        REQUIRE(ta.get_state()[9] == 0);
        // dv/...
        REQUIRE(ta.get_state()[10] == 0);
        REQUIRE(ta.get_state()[11] == 0);
        REQUIRE(ta.get_state()[12] == 1);
        REQUIRE(ta.get_state()[13] == 1);
        REQUIRE(ta.get_state()[14] == 0);
        REQUIRE(ta.get_state()[15] == 0);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 16, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // Single par, single var, mixed up order.
    {
        auto vsys = var_ode_sys(orig_sys, {par[0], v}, 2);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::tol = 1e-3};

        REQUIRE(ta.get_state()[0] == .2);
        REQUIRE(ta.get_state()[1] == .21);
        REQUIRE(ta.get_state()[2] == .3);
        REQUIRE(ta.get_state()[3] == .31);
        // dx/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 0);
        REQUIRE(ta.get_state()[6] == 0);
        REQUIRE(ta.get_state()[7] == 0);
        // dv/...
        REQUIRE(ta.get_state()[8] == 0);
        REQUIRE(ta.get_state()[9] == 0);
        REQUIRE(ta.get_state()[10] == 1);
        REQUIRE(ta.get_state()[11] == 1);
        // All the rest must be zero.
        REQUIRE(
            std::all_of(ta.get_state().begin() + 12, ta.get_state().end(), [](const auto &val) { return val == 0; }));
    }

    // Invalid state size passed to the ctor.
    {
        auto vsys = var_ode_sys(orig_sys, {par[0], v}, 2);

        REQUIRE_THROWS_MATCHES(
            taylor_adaptive_batch(vsys, {.2, .21, .3, .31, .4, .41}, 2, kw::tol = 1e-3), std::invalid_argument,
            Message("Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator in batch mode: the state vector has a dimension of 6 (in batches of 2), while the "
                    "total number of "
                    "equations is 12. "
                    "The size of the state vector must be equal either to the total number of equations times the "
                    "batch size, or to the "
                    "number of original (i.e., non-variational) equations, which for this system is 2, times the batch "
                    "size"));
        REQUIRE_THROWS_MATCHES(
            taylor_adaptive_batch(vsys, {.2, .21}, 2, kw::tol = 1e-3), std::invalid_argument,
            Message("Inconsistent sizes detected in the initialization of a variational adaptive Taylor "
                    "integrator in batch mode: the state vector has a dimension of 2 (in batches of 2), while the "
                    "total number of "
                    "equations is 12. "
                    "The size of the state vector must be equal either to the total number of equations times the "
                    "batch size, or to the "
                    "number of original (i.e., non-variational) equations, which for this system is 2, times the batch "
                    "size"));
    }

    // A couple of tests for derivatives wrt the initial time.
    {
        auto vsys = var_ode_sys(orig_sys, var_args::time, 1);

        const auto ic_x = .2, ic_v = .3, ic_par = .4, ic_tm = .5;

        auto ta = taylor_adaptive_batch{
            vsys, {ic_x, ic_x, ic_v, ic_v}, 2, kw::pars = {ic_par, ic_par}, kw::time = {ic_tm, ic_tm}, kw::tol = 1e-3};

        REQUIRE(ta.get_state().size() == 8u);

        REQUIRE(ta.get_state()[0] == ic_x);
        REQUIRE(ta.get_state()[1] == ic_x);
        REQUIRE(ta.get_state()[2] == ic_v);
        REQUIRE(ta.get_state()[3] == ic_v);
        // dx/...
        REQUIRE(ta.get_state()[4] == -ic_v);
        REQUIRE(ta.get_state()[5] == -ic_v);
        // dv/...
        REQUIRE(ta.get_state()[6] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
        REQUIRE(ta.get_state()[7] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
    }

    // Custom ordering for vars and time derivatives.
    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, x}, 1);

        const auto ic_x = .2, ic_v = .3, ic_par = .4, ic_tm = .5;

        auto ta = taylor_adaptive_batch{
            vsys, {ic_x, ic_x, ic_v, ic_v}, 2, kw::pars = {ic_par, ic_par}, kw::time = {ic_tm, ic_tm}, kw::tol = 1e-3};

        REQUIRE(ta.get_state().size() == 16u);

        REQUIRE(ta.get_state()[0] == ic_x);
        REQUIRE(ta.get_state()[1] == ic_x);
        REQUIRE(ta.get_state()[2] == ic_v);
        REQUIRE(ta.get_state()[3] == ic_v);
        // dx/...
        REQUIRE(ta.get_state()[4] == 0);
        REQUIRE(ta.get_state()[5] == 0);
        REQUIRE(ta.get_state()[6] == -ic_v);
        REQUIRE(ta.get_state()[7] == -ic_v);
        REQUIRE(ta.get_state()[8] == 1);
        REQUIRE(ta.get_state()[9] == 1);
        // dv/...
        REQUIRE(ta.get_state()[10] == 1);
        REQUIRE(ta.get_state()[11] == 1);
        REQUIRE(ta.get_state()[12] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
        REQUIRE(ta.get_state()[13] == approximately(-(std::cos(ic_tm) - ic_par * ic_v - std::sin(ic_x))));
        REQUIRE(ta.get_state()[14] == 0);
        REQUIRE(ta.get_state()[15] == 0);
    }
}

// A comprehensive test with variations wrt everything in the forced damped pendulum.
TEST_CASE("comp test")
{
    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    auto vsys = var_ode_sys(orig_sys, var_args::all, 1);

    const auto ic_x = .2, ic_v = .3, ic_tm = .5, ic_par = .4;

    {
        auto ta = taylor_adaptive{vsys, {ic_x, ic_v}, kw::pars = {ic_par}, kw::time = ic_tm};

        const auto delta_x = 1e-8, delta_v = -2e-8, delta_tm = 3e-8, delta_par = -4e-8;

        auto ta_orig = taylor_adaptive{
            orig_sys, {ic_x + delta_x, ic_v + delta_v}, kw::pars = {ic_par + delta_par}, kw::time = ic_tm + delta_tm};

        REQUIRE(ta.get_sys().size() == 10u);
        REQUIRE(ta_orig.get_sys().size() == 2u);
        REQUIRE(ta.is_variational());
        REQUIRE(!ta_orig.is_variational());
        REQUIRE(ta.get_n_orig_sv() == 2u);
        REQUIRE(ta_orig.get_n_orig_sv() == 2u);

        ta.propagate_until(3.);
        ta_orig.propagate_until(3.);

        REQUIRE(ta_orig.get_state()[0]
                == approximately(ta.get_state()[0] + ta.get_state()[2] * delta_x + ta.get_state()[3] * delta_v
                                     + ta.get_state()[4] * delta_par + ta.get_state()[5] * delta_tm,
                                 1000.));
        REQUIRE(ta_orig.get_state()[1]
                == approximately(ta.get_state()[1] + ta.get_state()[6] * delta_x + ta.get_state()[7] * delta_v
                                     + ta.get_state()[8] * delta_par + ta.get_state()[9] * delta_tm,
                                 1000.));
    }

    {
        auto ta = taylor_adaptive_batch{vsys,
                                        {ic_x, ic_x + 0.01, ic_v, ic_v + 0.02},
                                        2,
                                        kw::pars = {ic_par, ic_par + 0.03},
                                        kw::time = {ic_tm, ic_tm + 0.04}};

        const auto delta_x = 1e-8, delta_v = -2e-8, delta_tm = 3e-8, delta_par = -4e-8;

        auto ta_orig
            = taylor_adaptive_batch{orig_sys,
                                    {ic_x + delta_x, ic_x + 0.01 + delta_x, ic_v + delta_v, ic_v + 0.02 + delta_v},
                                    2,
                                    kw::pars = {ic_par + delta_par, ic_par + 0.03 + delta_par},
                                    kw::time = {ic_tm + delta_tm, ic_tm + 0.04 + delta_tm}};

        REQUIRE(ta.get_sys().size() == 10u);
        REQUIRE(ta_orig.get_sys().size() == 2u);
        REQUIRE(ta.is_variational());
        REQUIRE(!ta_orig.is_variational());
        REQUIRE(ta.get_n_orig_sv() == 2u);
        REQUIRE(ta_orig.get_n_orig_sv() == 2u);

        ta.propagate_until(3.);
        ta_orig.propagate_until(3.);

        REQUIRE(ta_orig.get_state()[0]
                == approximately(ta.get_state()[0] + ta.get_state()[4] * delta_x + ta.get_state()[6] * delta_v
                                     + ta.get_state()[8] * delta_par + ta.get_state()[10] * delta_tm,
                                 1000.));
        REQUIRE(ta_orig.get_state()[1]
                == approximately(ta.get_state()[1] + ta.get_state()[5] * delta_x + ta.get_state()[7] * delta_v
                                     + ta.get_state()[9] * delta_par + ta.get_state()[11] * delta_tm,
                                 1000.));

        REQUIRE(ta_orig.get_state()[2]
                == approximately(ta.get_state()[2] + ta.get_state()[12] * delta_x + ta.get_state()[14] * delta_v
                                     + ta.get_state()[16] * delta_par + ta.get_state()[18] * delta_tm,
                                 1000.));
        REQUIRE(ta_orig.get_state()[3]
                == approximately(ta.get_state()[3] + ta.get_state()[13] * delta_x + ta.get_state()[15] * delta_v
                                     + ta.get_state()[17] * delta_par + ta.get_state()[19] * delta_tm,
                                 1000.));
    }
}

TEST_CASE("taylor map")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    {
        auto vsys = var_ode_sys(orig_sys, var_args::vars, 3);
        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::compact_mode = true};

        const auto dx = 1e-4, dv = 2e-4;
        auto ta_nv = taylor_adaptive{orig_sys, {.2 + dx, .3 + dv}, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0.});

        REQUIRE(ta.get_tstate().size() == 2u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);

        ta.eval_taylor_map({dx, dv});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1]));

        REQUIRE(ta_nv.get_n_orig_sv() == 2u);
        REQUIRE(ta_nv.get_dim() == 2u);

        REQUIRE(ta.get_n_orig_sv() == 2u);
        REQUIRE(ta.get_dim() > 2u);

        // Check error throwing on non-variational integrators.
        REQUIRE_THROWS_MATCHES(ta_nv.get_tstate(), std::invalid_argument,
                               Message("The function 'get_tstate()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(ta_nv.get_vargs(), std::invalid_argument,
                               Message("The function 'get_vargs()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.eval_taylor_map({0., 0.}), std::invalid_argument,
            Message("The function 'eval_taylor_map()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(ta_nv.get_vorder(), std::invalid_argument,
                               Message("The function 'get_vorder()' cannot be invoked on non-variational integrators"));

        // Check error conditions on invalid input to eval_taylor_map().
        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({0.}), std::invalid_argument,
                               Message("Unable to compute the Taylor map: the input range of values has a "
                                       "size of 1, but the number of variational arguments is 2"));
        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({0., 0., 0.}), std::invalid_argument,
                               Message("Unable to compute the Taylor map: the input range of values has a "
                                       "size of 3, but the number of variational arguments is 2"));

        // Check that the Taylor map machinery keeps on working after copy/s11n.
        auto ta_copy = ta;

        // Check that the internal tstate is properly copied.
        REQUIRE(ta_copy.get_tstate() == ta.get_tstate());

        // Test the internal compiled function/LLVM state is properly set up in the copy.
        ta_copy.eval_taylor_map({dx, dv});

        REQUIRE(ta_copy.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta_copy.get_tstate()[1] == approximately(ta_nv.get_state()[1]));

        // s11n test.
        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ta_copy;
        }

        ta_copy = ta_nv;

        REQUIRE(!ta_copy.is_variational());

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ta_copy;
        }

        REQUIRE(ta_copy.is_variational());
        REQUIRE(ta_copy.get_tstate() == ta.get_tstate());
        ta_copy.eval_taylor_map({dx, dv});
        REQUIRE(ta_copy.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta_copy.get_tstate()[1] == approximately(ta_nv.get_state()[1]));
    }

    {
        auto vsys = var_ode_sys(orig_sys, {v, x}, 3);
        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::compact_mode = true};

        const auto dx = 1e-4, dv = 2e-4;
        auto ta_nv = taylor_adaptive{orig_sys, {.2 + dx, .3 + dv}, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0.});

        REQUIRE(ta.get_tstate().size() == 2u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);

        ta.eval_taylor_map({dv, dx});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1]));
    }

    {
        auto vsys = var_ode_sys(orig_sys, var_args::params | var_args::vars, 4);
        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::pars = {.4}, kw::compact_mode = true};

        const auto dx = 1e-4, dv = -2e-4, dp = 3e-4;
        auto ta_nv = taylor_adaptive{orig_sys, {.2 + dx, .3 + dv}, kw::pars = {.4 + dp}, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 2u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);

        ta.eval_taylor_map({dx, dv, dp});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1]));
    }

    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, par[0]}, 1);
        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::pars = {.4}, kw::time = .5, kw::compact_mode = true};

        const auto dv = -2e-8, dp = 3e-8, dt = -4e-8;
        auto ta_nv = taylor_adaptive{
            orig_sys, {.2, .3 + dv}, kw::pars = {.4 + dp}, kw::time = .5 + dt, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 2u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);

        ta.eval_taylor_map({dv, dt, dp});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
    }

#if defined(HEYOKA_HAVE_REAL)

    {
        const auto prec = 14;

        auto vsys = var_ode_sys(orig_sys, var_args::vars, 3);
        auto ta = taylor_adaptive{vsys, {mppp::real{.2, prec}, mppp::real{.3, prec}}, kw::compact_mode = true};

        const auto dx = mppp::real{1e-4, prec}, dv = mppp::real{2e-4, prec};
        auto ta_nv = taylor_adaptive{
            orig_sys, {mppp::real{.2, prec} + dx, mppp::real{.3, prec} + dv}, kw::compact_mode = true};

        ta.propagate_until(mppp::real{3, prec});
        ta_nv.propagate_until(mppp::real{3, prec});

        ta.eval_taylor_map({mppp::real{0., prec}, mppp::real{0., prec}});

        REQUIRE(ta.get_tstate().size() == 2u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);

        ta.eval_taylor_map({dx, dv});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1]));

        REQUIRE_THROWS_MATCHES(
            ta.eval_taylor_map({dx, mppp::real{1e-4, prec + 1}}), std::invalid_argument,
            Message("Unable to compute the Taylor map: the input value at index 1 has a precision of "
                    "15, but the expected precision instead is 14"));

        // Check that changing the precision of a derivative in the state vector
        // correctly triggers an error during the evaluation of a Taylor map.
        ta.get_state_data()[2].prec_round(21);

        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({dx, dv}), std::invalid_argument,
                               Message("A state variable with precision 21 was detected in the state vector: this is "
                                       "incompatible with the integrator precision of 14"));
    }

#endif
}

TEST_CASE("taylor map batch")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    {
        auto vsys = var_ode_sys(orig_sys, var_args::vars, 3);

        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::compact_mode = true};

        const auto dx = 1e-4, dv = 2e-4;
        auto ta_nv = taylor_adaptive_batch{
            orig_sys, {.2 + dx, .21 + 2 * dx, .3 + dv, .31 + 2 * dv}, 2, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 4u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);
        REQUIRE(ta.get_tstate()[2] == ta.get_state()[2]);
        REQUIRE(ta.get_tstate()[3] == ta.get_state()[3]);

        ta.eval_taylor_map({dx, 2 * dx, dv, 2 * dv});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
        REQUIRE(ta.get_tstate()[2] == approximately(ta_nv.get_state()[2], 1000.));
        REQUIRE(ta.get_tstate()[3] == approximately(ta_nv.get_state()[3], 1000.));

        REQUIRE(ta_nv.get_n_orig_sv() == 2u);
        REQUIRE(ta_nv.get_dim() == 2u);

        REQUIRE(ta.get_n_orig_sv() == 2u);
        REQUIRE(ta.get_dim() > 2u);

        // Check error throwing on non-variational integrators.
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_tstate(), std::invalid_argument,
            Message("The function 'get_tstate()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_vargs(), std::invalid_argument,
            Message("The function 'get_vargs()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.eval_taylor_map({0., 0.}), std::invalid_argument,
            Message("The function 'eval_taylor_map()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_vorder(), std::invalid_argument,
            Message("The function 'get_vorder()' cannot be invoked on non-variational batch integrators"));

        // Check error conditions on invalid input to eval_taylor_map().
        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({0.}), std::invalid_argument,
                               Message("Unable to compute the Taylor map: the input range of values has a "
                                       "size of 1, which is not a multiple of the batch size 2"));
        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({0., 0.}), std::invalid_argument,
                               Message("Unable to compute the Taylor map: the input range of values has a "
                                       "size of 1 (in batches of 2), but the number of variational arguments is 2"));
        REQUIRE_THROWS_MATCHES(ta.eval_taylor_map({0., 0., 0., 0., 0., 0.}), std::invalid_argument,
                               Message("Unable to compute the Taylor map: the input range of values has a "
                                       "size of 3 (in batches of 2), but the number of variational arguments is 2"));

        // Check that the Taylor map machinery keeps on working after copy/s11n.
        auto ta_copy = ta;

        // Check that the internal tstate is properly copied.
        REQUIRE(ta_copy.get_tstate() == ta.get_tstate());

        // Test the internal compiled function/LLVM state is properly set up in the copy.
        ta_copy.eval_taylor_map({dx, 2 * dx, dv, 2 * dv});

        REQUIRE(ta_copy.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta_copy.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
        REQUIRE(ta_copy.get_tstate()[2] == approximately(ta_nv.get_state()[2], 1000.));
        REQUIRE(ta_copy.get_tstate()[3] == approximately(ta_nv.get_state()[3], 1000.));

        // s11n test.
        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ta_copy;
        }

        ta_copy = ta_nv;

        REQUIRE(!ta_copy.is_variational());

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ta_copy;
        }

        REQUIRE(ta_copy.is_variational());
        REQUIRE(ta_copy.get_tstate() == ta.get_tstate());
        ta_copy.eval_taylor_map({dx, 2 * dx, dv, 2 * dv});
        REQUIRE(ta_copy.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta_copy.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
        REQUIRE(ta_copy.get_tstate()[2] == approximately(ta_nv.get_state()[2], 1000.));
        REQUIRE(ta_copy.get_tstate()[3] == approximately(ta_nv.get_state()[3], 1000.));
    }

    {
        auto vsys = var_ode_sys(orig_sys, {v, x}, 3);
        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::compact_mode = true};

        const auto dx = 1e-4, dv = 2e-4;
        auto ta_nv = taylor_adaptive_batch{
            orig_sys, {.2 + dx, .21 + 2 * dx, .3 + dv, .31 + 2 * dv}, 2, kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 4u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);
        REQUIRE(ta.get_tstate()[2] == ta.get_state()[2]);
        REQUIRE(ta.get_tstate()[3] == ta.get_state()[3]);

        ta.eval_taylor_map({dv, 2 * dv, dx, 2 * dx});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
        REQUIRE(ta.get_tstate()[2] == approximately(ta_nv.get_state()[2], 1000.));
        REQUIRE(ta.get_tstate()[3] == approximately(ta_nv.get_state()[3], 1000.));
    }

    {
        auto vsys = var_ode_sys(orig_sys, var_args::params | var_args::vars, 4);
        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::pars = {.4, .41}, kw::compact_mode = true};

        const auto dx = 1e-4, dv = -2e-4, dp = 3e-4;
        auto ta_nv = taylor_adaptive_batch{orig_sys,
                                           {.2 + dx, .21 + 2 * dx, .3 + dv, .31 + 2 * dv},
                                           2,
                                           kw::pars = {.4 + dp, .41 + 2 * dp},
                                           kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0., 0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 4u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);
        REQUIRE(ta.get_tstate()[2] == ta.get_state()[2]);
        REQUIRE(ta.get_tstate()[3] == ta.get_state()[3]);

        ta.eval_taylor_map({dx, 2 * dx, dv, 2 * dv, dp, 2 * dp});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0]));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1]));
        REQUIRE(ta.get_tstate()[2] == approximately(ta_nv.get_state()[2]));
        REQUIRE(ta.get_tstate()[3] == approximately(ta_nv.get_state()[3]));
    }

    {
        auto vsys = var_ode_sys(orig_sys, {v, heyoka::time, par[0]}, 1);
        auto ta = taylor_adaptive_batch{
            vsys, {.2, .21, .3, .31}, 2, kw::pars = {.4, .41}, kw::time = {.5, .51}, kw::compact_mode = true};

        const auto dv = -2e-8, dp = 3e-8, dt = -4e-8;
        auto ta_nv = taylor_adaptive_batch{orig_sys,
                                           {.2, .21, .3 + dv, .31 + 2 * dv},
                                           2,
                                           kw::pars = {.4 + dp, .41 + 2 * dp},
                                           kw::time = {.5 + dt, .51 + 2 * dt},
                                           kw::compact_mode = true};

        ta.propagate_until(3.);
        ta_nv.propagate_until(3.);

        ta.eval_taylor_map({0., 0., 0., 0., 0., 0.});

        REQUIRE(ta.get_tstate().size() == 4u);
        REQUIRE(ta.get_tstate()[0] == ta.get_state()[0]);
        REQUIRE(ta.get_tstate()[1] == ta.get_state()[1]);
        REQUIRE(ta.get_tstate()[2] == ta.get_state()[2]);
        REQUIRE(ta.get_tstate()[3] == ta.get_state()[3]);

        ta.eval_taylor_map({dv, 2 * dv, dt, 2 * dt, dp, 2 * dp});

        REQUIRE(ta.get_tstate()[0] == approximately(ta_nv.get_state()[0], 1000.));
        REQUIRE(ta.get_tstate()[1] == approximately(ta_nv.get_state()[1], 1000.));
        REQUIRE(ta.get_tstate()[2] == approximately(ta_nv.get_state()[2], 1000.));
        REQUIRE(ta.get_tstate()[3] == approximately(ta_nv.get_state()[3], 1000.));
    }
}

TEST_CASE("vslice mindex")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    // The original ODEs.
    auto orig_sys = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

    auto vsys = var_ode_sys(orig_sys, var_args::vars, 3);

    {
        auto ta = taylor_adaptive{vsys, {.2, .3}, kw::compact_mode = true};
        auto ta_nv = taylor_adaptive{orig_sys, {.2, .3}, kw::compact_mode = true};

        auto sl = ta.get_vslice(0);
        REQUIRE(sl.first == 0u);
        REQUIRE(sl.second == 2u);

        REQUIRE(ta.get_mindex(0).first == 0u);
        REQUIRE(ta.get_mindex(0).second.empty());

        REQUIRE(ta.get_mindex(1).first == 1u);
        REQUIRE(ta.get_mindex(1).second.empty());

        sl = ta.get_vslice(1);
        REQUIRE(sl.first == 2u);
        REQUIRE(sl.second == 6u);

        REQUIRE(ta.get_mindex(2).first == 0u);
        REQUIRE(ta.get_mindex(2).second.size() == 1u);
        REQUIRE(ta.get_mindex(2).second[0].first == 0);
        REQUIRE(ta.get_mindex(2).second[0].second == 1);

        REQUIRE(ta.get_mindex(3).first == 0u);
        REQUIRE(ta.get_mindex(3).second.size() == 1u);
        REQUIRE(ta.get_mindex(3).second[0].first == 1);
        REQUIRE(ta.get_mindex(3).second[0].second == 1);

        REQUIRE(ta.get_mindex(4).first == 1u);
        REQUIRE(ta.get_mindex(4).second.size() == 1u);
        REQUIRE(ta.get_mindex(4).second[0].first == 0);
        REQUIRE(ta.get_mindex(4).second[0].second == 1);

        REQUIRE(ta.get_mindex(5).first == 1u);
        REQUIRE(ta.get_mindex(5).second.size() == 1u);
        REQUIRE(ta.get_mindex(5).second[0].first == 1);
        REQUIRE(ta.get_mindex(5).second[0].second == 1);

        sl = ta.get_vslice(0, 0);
        REQUIRE(sl.first == 0u);
        REQUIRE(sl.second == 1u);

        sl = ta.get_vslice(1, 0);
        REQUIRE(sl.first == 1u);
        REQUIRE(sl.second == 2u);

        sl = ta.get_vslice(0, 1);
        REQUIRE(sl.first == 2u);
        REQUIRE(sl.second == 4u);

        sl = ta.get_vslice(1, 1);
        REQUIRE(sl.first == 4u);
        REQUIRE(sl.second == 6u);

        sl = ta.get_vslice(10);
        REQUIRE(sl.first == ta.get_state().size());
        REQUIRE(sl.second == ta.get_state().size());

        sl = ta.get_vslice(2, 0);
        REQUIRE(sl.first == ta.get_state().size());
        REQUIRE(sl.second == ta.get_state().size());

        sl = ta.get_vslice(0, 10);
        REQUIRE(sl.first == ta.get_state().size());
        REQUIRE(sl.second == ta.get_state().size());

        // Error throwing.
        REQUIRE_THROWS_MATCHES(ta_nv.get_vslice(0), std::invalid_argument,
                               Message("The function 'get_vslice()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(ta_nv.get_vslice(0, 0), std::invalid_argument,
                               Message("The function 'get_vslice()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(ta_nv.get_mindex(0), std::invalid_argument,
                               Message("The function 'get_mindex()' cannot be invoked on non-variational integrators"));
        REQUIRE_THROWS_MATCHES(ta.get_mindex(100), std::invalid_argument,
                               Message("Cannot fetch the multiindex of the derivative at index 100: the index "
                                       "is not less than the total number of derivatives (20)"));
    }

    {
        auto ta = taylor_adaptive_batch{vsys, {.2, .21, .3, .31}, 2, kw::compact_mode = true};
        auto ta_nv = taylor_adaptive_batch{orig_sys, {.2, .21, .3, .31}, 2, kw::compact_mode = true};

        auto sl = ta.get_vslice(0);
        REQUIRE(sl.first == 0u);
        REQUIRE(sl.second == 2u);

        REQUIRE(ta.get_mindex(0).first == 0u);
        REQUIRE(ta.get_mindex(0).second.empty());

        REQUIRE(ta.get_mindex(1).first == 1u);
        REQUIRE(ta.get_mindex(1).second.empty());

        sl = ta.get_vslice(1);
        REQUIRE(sl.first == 2u);
        REQUIRE(sl.second == 6u);

        REQUIRE(ta.get_mindex(2).first == 0u);
        REQUIRE(ta.get_mindex(2).second.size() == 1u);
        REQUIRE(ta.get_mindex(2).second[0].first == 0);
        REQUIRE(ta.get_mindex(2).second[0].second == 1);

        REQUIRE(ta.get_mindex(3).first == 0u);
        REQUIRE(ta.get_mindex(3).second.size() == 1u);
        REQUIRE(ta.get_mindex(3).second[0].first == 1);
        REQUIRE(ta.get_mindex(3).second[0].second == 1);

        REQUIRE(ta.get_mindex(4).first == 1u);
        REQUIRE(ta.get_mindex(4).second.size() == 1u);
        REQUIRE(ta.get_mindex(4).second[0].first == 0);
        REQUIRE(ta.get_mindex(4).second[0].second == 1);

        REQUIRE(ta.get_mindex(5).first == 1u);
        REQUIRE(ta.get_mindex(5).second.size() == 1u);
        REQUIRE(ta.get_mindex(5).second[0].first == 1);
        REQUIRE(ta.get_mindex(5).second[0].second == 1);

        sl = ta.get_vslice(0, 0);
        REQUIRE(sl.first == 0u);
        REQUIRE(sl.second == 1u);

        sl = ta.get_vslice(1, 0);
        REQUIRE(sl.first == 1u);
        REQUIRE(sl.second == 2u);

        sl = ta.get_vslice(0, 1);
        REQUIRE(sl.first == 2u);
        REQUIRE(sl.second == 4u);

        sl = ta.get_vslice(1, 1);
        REQUIRE(sl.first == 4u);
        REQUIRE(sl.second == 6u);

        sl = ta.get_vslice(10);
        REQUIRE(sl.first == ta.get_state().size() / 2u);
        REQUIRE(sl.second == ta.get_state().size() / 2u);

        sl = ta.get_vslice(2, 0);
        REQUIRE(sl.first == ta.get_state().size() / 2u);
        REQUIRE(sl.second == ta.get_state().size() / 2u);

        sl = ta.get_vslice(0, 10);
        REQUIRE(sl.first == ta.get_state().size() / 2u);
        REQUIRE(sl.second == ta.get_state().size() / 2u);

        // Error throwing.
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_vslice(0), std::invalid_argument,
            Message("The function 'get_vslice()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_vslice(0, 0), std::invalid_argument,
            Message("The function 'get_vslice()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(
            ta_nv.get_mindex(0), std::invalid_argument,
            Message("The function 'get_mindex()' cannot be invoked on non-variational batch integrators"));
        REQUIRE_THROWS_MATCHES(ta.get_mindex(100), std::invalid_argument,
                               Message("Cannot fetch the multiindex of the derivative at index 100: the index "
                                       "is not less than the total number of derivatives (20)"));
    }
}
