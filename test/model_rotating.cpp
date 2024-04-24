// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/rotating.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Wrapper to ease the transition of old test code
// after the removal of sum_sq() from the public API.
auto sum_sq(const std::vector<expression> &args)
{
    std::vector<expression> new_args;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        new_args.push_back(pow(arg, 2_dbl));
    }

    return sum(new_args);
}

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    {
        auto dyn = model::rotating();

        REQUIRE(dyn.size() == 6u);
        REQUIRE(dyn[0].first == "x"_var);
        REQUIRE(dyn[0].second == "vx"_var);
        REQUIRE(dyn[1].first == "y"_var);
        REQUIRE(dyn[1].second == "vy"_var);
        REQUIRE(dyn[2].first == "z"_var);
        REQUIRE(dyn[2].second == "vz"_var);
        REQUIRE(dyn[3].first == "vx"_var);
        REQUIRE(dyn[3].second == 0_dbl);
        REQUIRE(dyn[4].first == "vy"_var);
        REQUIRE(dyn[4].second == 0_dbl);
        REQUIRE(dyn[5].first == "vz"_var);
        REQUIRE(dyn[5].second == 0_dbl);

        auto pot = model::rotating_potential();
        REQUIRE(pot == 0_dbl);
    }

    // Energy conservation.
    {
        auto dyn = model::rotating(kw::omega = {.1, .2, .3});

        const std::vector<double> init_state = {.4, .5, .6, .7, .8, .9};

        auto ta = taylor_adaptive{dyn, init_state, kw::compact_mode = true};

        REQUIRE(ta.get_decomposition().size() == 49u);

        ta.propagate_until(20.);

        auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

        llvm_state s;

        const auto dc = add_cfunc<double>(
            s, "en",
            // NOTE: need to add the kinetic energy per unit of mass.
            {0.5 * (vx * vx + vy * vy + vz * vz) + model::rotating_potential(kw::omega = {.1, .2, .3})},
            {x, y, z, vx, vy, vz});

        REQUIRE(dc.size() == 23u);

        s.compile();

        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en"));
        double E0 = 0;
        cf(&E0, init_state.data(), nullptr, nullptr);

        double E = 0;
        cf(&E, ta.get_state().data(), nullptr, nullptr);

        REQUIRE(E == approximately(E0));

        REQUIRE(0.5 * sum_sq({vx, vy, vz}) + model::rotating_potential(kw::omega = {.1, .2, .3})
                == model::rotating_energy(kw::omega = {.1, .2, .3}));
    }

    {
        const std::vector omega_vals = {.1, .2, .3};

        auto dyn = model::rotating(kw::omega = {par[0], par[1], par[2]});

        const std::vector<double> init_state = {.4, .5, .6, .7, .8, .9};

        auto ta = taylor_adaptive{dyn, init_state, kw::compact_mode = true, kw::pars = omega_vals};

        REQUIRE(ta.get_decomposition().size() == 52u);

        ta.propagate_until(20.);

        auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

        llvm_state s;

        const auto dc = add_cfunc<double>(
            s, "en",
            // NOTE: need to add the kinetic energy per unit of mass.
            {0.5 * (vx * vx + vy * vy + vz * vz) + model::rotating_potential(kw::omega = {par[0], par[1], par[2]})},
            {x, y, z, vx, vy, vz});

        REQUIRE(dc.size() == 24u);

        s.compile();

        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en"));
        double E0 = 0;
        cf(&E0, init_state.data(), omega_vals.data(), nullptr);

        double E = 0;
        cf(&E, ta.get_state().data(), omega_vals.data(), nullptr);

        REQUIRE(E == approximately(E0));

        REQUIRE(0.5 * sum_sq({vx, vy, vz}) + model::rotating_potential(kw::omega = {par[0], par[1], par[2]})
                == model::rotating_energy(kw::omega = {par[0], par[1], par[2]}));
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::rotating(kw::omega = {.1, .2}), std::invalid_argument,
                           Message("In a rotating reference frame model the angular velocity must be a "
                                   "3-dimensional vector, but instead it is a 2-dimensional vector"));
    REQUIRE_THROWS_MATCHES(model::rotating(kw::omega = {.1}), std::invalid_argument,
                           Message("In a rotating reference frame model the angular velocity must be a "
                                   "3-dimensional vector, but instead it is a 1-dimensional vector"));
    REQUIRE_THROWS_MATCHES(model::rotating(kw::omega = {.1, .2, .3, .4}), std::invalid_argument,
                           Message("In a rotating reference frame model the angular velocity must be a "
                                   "3-dimensional vector, but instead it is a 4-dimensional vector"));
    REQUIRE_THROWS_MATCHES(model::rotating_potential(kw::omega = {.1, .2, .3, .4}), std::invalid_argument,
                           Message("In a rotating reference frame model the angular velocity must be a "
                                   "3-dimensional vector, but instead it is a 4-dimensional vector"));
    REQUIRE_THROWS_MATCHES(model::rotating_energy(kw::omega = {.1, .2, .3, .4}), std::invalid_argument,
                           Message("In a rotating reference frame model the angular velocity must be a "
                                   "3-dimensional vector, but instead it is a 4-dimensional vector"));
}
