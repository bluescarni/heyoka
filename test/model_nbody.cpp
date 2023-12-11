// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/detail/sub.hpp>
#include <heyoka/detail/sum_sq.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
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

        // Check that llvm.pow appears only maximum 3 times: its declaration plus 2 uses
        // for determining the timestep size. Vectorisation may further reduce this number.
        std::vector<boost::iterator_range<std::string::const_iterator>> pow_matches;
        boost::find_all(pow_matches, ta.get_llvm_state().get_ir(), "@llvm.pow");
        REQUIRE(pow_matches.size() <= 3u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        // Check that all sums were replaced by sums of squares, and check the
        // number of subtractions.
        auto n_sums = 0, n_sum_sqs = 0, n_subs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
                n_subs += static_cast<int>(fptr->extract<detail::sub_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(n_subs == 45);
        REQUIRE(ta.get_decomposition().size() == 270u);

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 146u);

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

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 114u);

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

        // Check that all sums were replaced by sums of squares.
        auto n_sums = 0, n_sum_sqs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(ta.get_decomposition().size() == 305u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 146u);

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

        // Check that all sums were replaced by sums of squares.
        auto n_sums = 0, n_sum_sqs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(ta.get_decomposition().size() == 285u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 114u);

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

        // Check that all sums were replaced by sums of squares.
        auto n_sums = 0, n_sum_sqs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(ta.get_decomposition().size() == 287u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 146u);

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

        // Check that all sums were replaced by sums of squares.
        auto n_sums = 0, n_sum_sqs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(ta.get_decomposition().size() == 273u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 114u);

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

        // Check that all sums were replaced by sums of squares.
        auto n_sums = 0, n_sum_sqs = 0;
        for (const auto &[s_ex, _] : ta.get_decomposition()) {
            if (const auto *fptr = std::get_if<func>(&s_ex.value())) {
                n_sums += static_cast<int>(fptr->extract<detail::sum_impl>() != nullptr);
                n_sum_sqs += static_cast<int>(fptr->extract<detail::sum_sq_impl>() != nullptr);
            }
        }

        REQUIRE(n_sum_sqs == 15);
        REQUIRE(n_sums == 18);
        REQUIRE(ta.get_decomposition().size() == 255u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 124u);

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

        REQUIRE(ta.get_decomposition().size() == 72u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 37u);

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
        REQUIRE(en_out == 0.);
    }

    // Direct invocation of the potential top level helper.
    REQUIRE(model::nbody_potential(2, kw::masses = std::vector<double>{}) == 0_dbl);
    REQUIRE(model::nbody_potential(10, kw::masses = std::vector<double>{}) == 0_dbl);

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::nbody(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody_energy(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody_potential(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody(1), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 1: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::nbody(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
    REQUIRE_THROWS_MATCHES(model::nbody_energy(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
    REQUIRE_THROWS_MATCHES(model::nbody_potential(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
}

TEST_CASE("np1body")
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
        auto dyn = model::np1body(6, kw::masses = masses, kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true};

        REQUIRE(ta.get_decomposition().size() == 285u);

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 158u);

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

        auto dyn = model::np1body(6, kw::masses = new_masses, kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = new_masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 123u);

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
        auto dyn
            = model::np1body(6, kw::masses = {par[0], par[1], par[2], par[3], par[4], par[5]}, kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true,
                                          kw::pars = masses};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 158u);

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
        auto dyn = model::np1body(6, kw::masses = {par[0], par[1], par[2], par[3], par[4]}, kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = std::vector(masses.begin(), masses.begin() + 5),
                                           kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 5)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 123u);

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
        auto dyn = model::np1body(
            6,
            kw::masses = {par[0], par[1], par[2], expression{masses[3]}, expression{masses[4]}, expression{masses[5]}},
            kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 3)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 158u);

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
        auto dyn = model::np1body(
            6, kw::masses = {par[0], par[1], par[2], expression{0.}, expression{masses[4]}, expression{masses[5]}},
            kw::Gconst = Gconst);
        auto new_masses = masses;
        new_masses[3] = 0;
        auto en_ex = model::np1body_energy(6, kw::masses = new_masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true,
                                          kw::pars = std::vector(masses.begin(), masses.begin() + 3)};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 123u);

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
        auto dyn = model::np1body(6);
        auto en_ex = model::np1body_energy(6);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 136u);

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

    // Test with zeroth mass equal to zero.
    {
        auto dyn = model::np1body(6, kw::masses = {0., masses[1], masses[2], masses[3], masses[4], masses[5]},
                                  kw::Gconst = Gconst);
        auto new_masses = masses;
        new_masses[0] = 0;
        auto en_ex = model::np1body_energy(6, kw::masses = new_masses, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 141u);

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

    // Test empty mass vector.
    {
        auto dyn = model::np1body(6, kw::masses = std::vector<double>{}, kw::Gconst = Gconst);
        auto en_ex = model::np1body_energy(6, kw::masses = std::vector<double>{}, kw::Gconst = Gconst);

        auto ta = heyoka::taylor_adaptive{dyn, std::vector(n_ic.begin() + 6, n_ic.end()), kw::compact_mode = true};

        llvm_state s;
        std::vector<expression> vars;
        for (const auto &p : dyn) {
            vars.push_back(p.first);
        }

        const auto dc = add_cfunc<double>(s, "cf", {en_ex}, kw::vars = vars);

        REQUIRE(dc.size() == 31u);

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

    // Direct invocation of the potential top level helper.
    REQUIRE(model::np1body_potential(2, kw::masses = std::vector<double>{}) == 0_dbl);
    REQUIRE(model::np1body_potential(10, kw::masses = std::vector<double>{}) == 0_dbl);

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::np1body(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::np1body_energy(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::np1body_potential(0), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 0: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::np1body(1), std::invalid_argument,
                           Message("Cannot construct an N-body system with N == 1: at least 2 bodies are needed"));
    REQUIRE_THROWS_MATCHES(model::np1body(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
    REQUIRE_THROWS_MATCHES(model::np1body_energy(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
    REQUIRE_THROWS_MATCHES(model::np1body_potential(2, kw::masses = {1., 2., 3., 4.}), std::invalid_argument,
                           Message("In an N-body system the number of particles with mass (4) cannot be "
                                   "greater than the total number of particles (2)"));
}

// Integrate the usual outer Solar System setup in both the N-body and (N+1)-body
// formulations, and compare the results.
TEST_CASE("nbody np1body consistency")
{
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

    std::vector<double> np1_ic;
    for (auto i = 0u; i < 5u; ++i) {
        for (auto j = 0u; j < 6u; ++j) {
            np1_ic.push_back(n_ic[(i + 1u) * 6u + j] - n_ic[j]);
        }
    }

    auto nsys = model::nbody(6, kw::masses = masses, kw::Gconst = Gconst);
    auto np1sys = model::np1body(6, kw::masses = masses, kw::Gconst = Gconst);

    taylor_adaptive<double> ta_n{nsys, n_ic, kw::high_accuracy = true, kw::tol = 1e-18, kw::compact_mode = true};
    taylor_adaptive<double> ta_np1{np1sys, np1_ic, kw::high_accuracy = true, kw::tol = 1e-18, kw::compact_mode = true};

    // Correct the state in ta_n to put the barycentre in the origin.
    auto s_array = xt::adapt(ta_n.get_state_data(), {6, 6});
    auto m_array = xt::adapt(masses.data(), {6});

    // Cache the total mass.
    const auto tot_mass = xt::sum(m_array)[0];

    // Helpers to compute the position and velocity of the COM.
    auto get_com = [&s_array, &m_array, &tot_mass]() {
        auto com_x = xt::sum(m_array * xt::view(s_array, xt::all(), 0)) / tot_mass;
        auto com_y = xt::sum(m_array * xt::view(s_array, xt::all(), 1)) / tot_mass;
        auto com_z = xt::sum(m_array * xt::view(s_array, xt::all(), 2)) / tot_mass;

        return vNd<double, 3>{com_x[0], com_y[0], com_z[0]};
    };

    auto get_com_v = [&s_array, &m_array, &tot_mass]() {
        auto com_vx = xt::sum(m_array * xt::view(s_array, xt::all(), 3)) / tot_mass;
        auto com_vy = xt::sum(m_array * xt::view(s_array, xt::all(), 4)) / tot_mass;
        auto com_vz = xt::sum(m_array * xt::view(s_array, xt::all(), 5)) / tot_mass;

        return vNd<double, 3>{com_vx[0], com_vy[0], com_vz[0]};
    };

    // Compute position and velocity of the COM.
    const auto init_com = get_com();
    const auto init_com_v = get_com_v();

    std::cout << "Initial COM         : " << init_com << '\n';
    std::cout << "Initial COM velocity: " << init_com_v << '\n';

    // Offset the existing positions/velocities so that they refer
    // to the COM.
    xt::view(s_array, 0, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 1, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 2, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 3, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 4, xt::range(0, 3)) -= init_com;
    xt::view(s_array, 5, xt::range(0, 3)) -= init_com;

    xt::view(s_array, 0, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 1, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 2, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 3, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 4, xt::range(3, 6)) -= init_com_v;
    xt::view(s_array, 5, xt::range(3, 6)) -= init_com_v;

    std::cout << "New COM         : " << get_com() << '\n';
    std::cout << "New COM velocity: " << get_com_v() << '\n';

    // Integrate for 100 years.
    REQUIRE(std::get<0>(ta_n.propagate_for(100.)) == taylor_outcome::time_limit);
    REQUIRE(std::get<0>(ta_np1.propagate_for(100.)) == taylor_outcome::time_limit);

    // Compare.
    for (auto i = 0u; i < 5u; ++i) {
        for (auto j = 0u; j < 6u; ++j) {
            REQUIRE(ta_n.get_state()[(i + 1u) * 6u + j] - ta_n.get_state()[j]
                    == approximately(ta_np1.get_state()[i * 6u + j], 10000.));
        }
    }
}
