// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("nbody")
{
    using Catch::Matchers::Message;

    const auto Gconst = 0.01720209895 * 0.01720209895 * 365 * 365;

    const auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto n_ic = std::vector{// Sun.
                                  -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6,
                                  +6.69048890636161e-6 * 365, -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                                  // Jupiter.
                                  +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2,
                                  -5.59797969310664e-3 * 365, +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                                  // Saturn.
                                  +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1,
                                  -4.17354020307064e-3 * 365, +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                                  // Uranus.
                                  +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1,
                                  -3.25884806151064e-3 * 365, +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                                  // Neptune.
                                  -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1,
                                  -2.17471785045538e-4 * 365, -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                                  // Pluto.
                                  -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0,
                                  -1.76936577252484e-3 * 365, -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};

    {
        auto dyn = model::nbody(6, kw::masses = masses, kw::Gconst = Gconst);
        auto en_ex = model::nbody_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        std::vector new_masses(masses.begin(), masses.begin() + 5);

        auto dyn = model::nbody(6, kw::masses = new_masses, kw::Gconst = Gconst);
        auto en_ex = model::nbody_energy(6, kw::masses = new_masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        auto dyn = model::nbody(6, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]}, kw::Gconst = Gconst);
        auto en_ex = model::nbody_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true, kw::pars = masses};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        auto dyn = model::nbody(6, kw::masses = {par[0], par[1], par[2], par[3], par[4]}, kw::Gconst = Gconst);
        auto en_ex
            = model::nbody_energy(6, kw::masses = std::vector(masses.begin(), masses.begin() + 5), kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 5)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        auto dyn = model::nbody(
            6,
            kw::masses = {par[0], par[1], par[2], expression{masses[3]}, expression{masses[4]}, expression{masses[5]}},
            kw::Gconst = Gconst);
        auto en_ex = model::nbody_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 3)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        auto dyn = model::nbody(
            6, kw::masses = {par[0], par[1], par[2], expression{0.}, expression{masses[4]}, expression{masses[5]}},
            kw::Gconst = Gconst);
        auto new_masses = masses;
        new_masses[3] = 0;
        auto en_ex = model::nbody_energy(6, kw::masses = new_masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 3)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    {
        auto dyn = model::nbody(6);
        auto en_ex = model::nbody_energy(6);

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    // Test with all zero masses.
    {
        const auto zero_masses = std::vector<double>(6u, 0.);

        auto dyn = model::nbody(6, kw::masses = zero_masses);
        auto en_ex = model::nbody_energy(6, kw::masses = zero_masses);

        for (auto i = 0u; i < 6u; ++i) {
            REQUIRE(dyn[i * 6u + 3u].second == 0_dbl);
            REQUIRE(dyn[i * 6u + 4u].second == 0_dbl);
            REQUIRE(dyn[i * 6u + 5u].second == 0_dbl);
        }

        auto ta = heyoka::taylor_adaptive{dyn, n_ic, kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);
        s.optimise();
        s.compile();

        double en_out = 0;
        auto *cf
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cf"));
        cf(&en_out, ta.get_state_data(), nullptr, nullptr);
        const auto orig_en = en_out;

        auto res = ta.propagate_until(100.);

        REQUIRE(std::get<0>(res) == taylor_outcome::time_limit);

        cf(&en_out, ta.get_state_data(), nullptr, nullptr);

        REQUIRE(en_out == approximately(orig_en));
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::nbody(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody(1), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 1: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
}
