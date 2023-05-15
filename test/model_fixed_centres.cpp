// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    {
        auto dyn = model::fixed_centres();

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

        auto en = model::fixed_centres_energy();
        REQUIRE(en == 0.5_dbl * sum_sq({"vx"_var, "vy"_var, "vz"_var}));
    }

    // Test equivalence between two-body and fixed centres when the
    // two-body problem has only 1 massive mass.
    {
        auto dyn_fix = model::fixed_centres(kw::Gconst = 1.02, kw::masses = {1.01}, kw::positions = {1., 2., 3.});
        auto dyn_2bp = model::nbody(2, kw::Gconst = 1.02, kw::masses = {1.01});

        const auto init_x = 1. + 1.;
        const auto init_y = 2.;
        const auto init_z = 3.;
        const auto init_vx = .1;
        const auto init_vy = 1.;
        const auto init_vz = .2;

        const auto init_state = std::vector{init_x, init_y, init_z, init_vx, init_vy, init_vz};

        auto ta_fix = taylor_adaptive{dyn_fix, init_state, kw::compact_mode = true};
        auto ta_2bp = taylor_adaptive{dyn_2bp,
                                      {1., 2., 3., 0., 0., 0., init_x, init_y, init_z, init_vx, init_vy, init_vz},
                                      kw::compact_mode = true};

        ta_fix.propagate_until(20.);
        ta_2bp.propagate_until(20.);

        // NOTE: the tolerance here is relatively high because fixed centres
        // dynamic has a slightly different formulation of the dynamics wrt nbody.
        REQUIRE(ta_fix.get_state()[0] == approximately(ta_2bp.get_state()[6], 1000.));
        REQUIRE(ta_fix.get_state()[1] == approximately(ta_2bp.get_state()[7], 1000.));
        REQUIRE(ta_fix.get_state()[2] == approximately(ta_2bp.get_state()[8], 1000.));
        REQUIRE(ta_fix.get_state()[3] == approximately(ta_2bp.get_state()[9], 1000.));
        REQUIRE(ta_fix.get_state()[4] == approximately(ta_2bp.get_state()[10], 1000.));
        REQUIRE(ta_fix.get_state()[5] == approximately(ta_2bp.get_state()[11], 1000.));

        // Energy check.
        llvm_state s;
        add_cfunc<double>(
            s, "en",
            {model::fixed_centres_energy(kw::Gconst = 1.02, kw::masses = {1.01}, kw::positions = {1., 2., 3.})},
            kw::vars = {"x"_var, "y"_var, "z"_var, "vx"_var, "vy"_var, "vz"_var});
        s.optimise();
        s.compile();

        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en"));
        double E0 = 0;
        cf(&E0, init_state.data(), nullptr, nullptr);

        double E = 0;
        cf(&E, ta_fix.get_state().data(), nullptr, nullptr);

        REQUIRE(E == approximately(E0));
    }

    // Randomly generate a fixed centres problem and check energy conservation.
    {
        std::mt19937 rng;
        std::uniform_real_distribution<double> rdist(-1e-2, 1e-2);

        const auto n_masses = 100;
        std::vector<double> pos, masses;
        for (auto i = 0; i < n_masses; ++i) {
            masses.push_back(rdist(rng));

            for (auto j = 0; j < 3; ++j) {
                pos.push_back(rdist(rng));
            }
        }

        auto dyn = model::fixed_centres(kw::masses = masses, kw::positions = pos);
        auto ta = taylor_adaptive{dyn, {1., 0., 0., 0., 1., 0.}, kw::compact_mode = true};

        llvm_state s;
        add_cfunc<double>(s, "en", {model::fixed_centres_energy(kw::masses = masses, kw::positions = pos)},
                          kw::vars = {"x"_var, "y"_var, "z"_var, "vx"_var, "vy"_var, "vz"_var});
        s.optimise();
        s.compile();

        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("en"));

        double E0 = 0;
        cf(&E0, ta.get_state().data(), nullptr, nullptr);

        ta.propagate_until(100.);

        double E = 0;
        cf(&E, ta.get_state().data(), nullptr, nullptr);

        REQUIRE(E == approximately(E0));

        // Test also the fixed_centres_potential implementation.
        auto kin = 0.5_dbl * sum_sq({"vx"_var, "vy"_var, "vz"_var});
        REQUIRE(model::fixed_centres_energy(kw::Gconst = 1.2, kw::masses = masses, kw::positions = pos)
                == kin + model::fixed_centres_potential(kw::masses = masses, kw::positions = pos, kw::Gconst = 1.2));
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres(kw::masses = {1.}, kw::positions = {2.}), std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 1"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3.}), std::invalid_argument,
        Message("In a fixed centres system the number of masses (1) differs from the number of position vectors (2)"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3., 4.}), std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 7"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_energy(kw::masses = {1.}, kw::positions = {2.}), std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 1"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_energy(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3.}), std::invalid_argument,
        Message("In a fixed centres system the number of masses (1) differs from the number of position vectors (2)"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_energy(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3., 4.}),
        std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 7"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_potential(kw::masses = {1.}, kw::positions = {2.}), std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 1"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_potential(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3.}),
        std::invalid_argument,
        Message("In a fixed centres system the number of masses (1) differs from the number of position vectors (2)"));
    REQUIRE_THROWS_MATCHES(
        model::fixed_centres_potential(kw::masses = {1.}, kw::positions = {2., 2., 2., 3., 3., 3., 4.}),
        std::invalid_argument,
        Message("In a fixed centres system the positions vector's size must be a multiple of 3, but instead it is 7"));
}
