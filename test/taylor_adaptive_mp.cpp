// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include <mp++/real.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Tests to check precision handling on construction.
TEST_CASE("ctors prec")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    const auto prec = 30u;
    const auto sprec = static_cast<int>(prec);

    for (auto cm : {false, true}) {
        // Check the logic around precision handling in the constructor:
        // - prec member is inited correctly,
        // - tolerance and time are default constructed with the correct
        //   precision and values,
        // - pars are padded with the correct precision.
        auto ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm,
                                              kw::opt_level = 0u);

        REQUIRE(ta.get_prec() == prec);
        REQUIRE(ta.get_time() == 0);
        REQUIRE(ta.get_time().get_prec() == sprec);
        // TODO test tolerance value as well.
        REQUIRE(ta.get_tol().get_prec() == sprec);
        REQUIRE(ta.get_pars()[0] == 0);
        REQUIRE(ta.get_pars()[0].get_prec() == sprec);
        REQUIRE(std::all_of(ta.get_tc().begin(), ta.get_tc().end(),
                            [sprec](const auto &v) { return v == 0 && v.get_prec() == sprec; }));
        REQUIRE(std::all_of(ta.get_d_output().begin(), ta.get_d_output().end(),
                            [sprec](const auto &v) { return v == 0 && v.get_prec() == sprec; }));
        REQUIRE(ta.get_last_h() == 0);
        REQUIRE(ta.get_last_h().get_prec() == sprec);

        // Check that tolerance is autocast to the correct precision.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::tol = mppp::real(1e-1));
        REQUIRE(ta.get_tol() == mppp::real(1e-1, sprec));
        REQUIRE(ta.get_tol().get_prec() == sprec);

        // Check with explicit time.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::time = mppp::real(1, sprec));
        REQUIRE(ta.get_time() == 1);
        REQUIRE(ta.get_time().get_prec() == sprec);

        // Check wrong time prec raises an error.
        REQUIRE_THROWS_MATCHES(
            taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                        kw::time = mppp::real{0, sprec + 1}),
            std::invalid_argument,
            Message("Invalid precision detected in the time variable: the precision of the integrator is "
                    "30, but the time variable has a precision of 31 instead"));

        // Wrong precision in the state vector.
        REQUIRE_THROWS_MATCHES(taylor_adaptive<mppp::real>({x + par[0], y + par[1]},
                                                           {mppp::real{1, prec}, mppp::real{1, prec + 1}},
                                                           kw::compact_mode = cm, kw::opt_level = 0u),
                               std::invalid_argument,
                               Message("A state variable with precision 31 was detected in the state "
                                       "vector: this is incompatible with the integrator precision of 30"));

        // Wrong precision in the pars.
        REQUIRE_THROWS_MATCHES(taylor_adaptive<mppp::real>({x + par[0], y + par[1]},
                                                           {mppp::real{1, prec}, mppp::real{1, prec}},
                                                           kw::compact_mode = cm, kw::opt_level = 0u,
                                                           kw::pars = {mppp::real{1, prec}, mppp::real{1, prec + 1}}),
                               std::invalid_argument,
                               Message("A value with precision 31 was detected in the parameter "
                                       "vector: this is incompatible with the integrator precision of 30"));
    }
}

// Odes consisting of trivial right-hand sides.
TEST_CASE("trivial odes")
{
    auto [x, y, z, u] = make_vars("x", "y", "z", "u");

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto prec : {30, 123}) {
                auto ta = taylor_adaptive<mppp::real>(
                    {prime(x) = x, prime(y) = par[0] + par[1], prime(z) = heyoka::time, prime(u) = 1.5_dbl},
                    {mppp::real{1, prec}, mppp::real{2, prec}, mppp::real{3, prec}, mppp::real{4, prec}},
                    kw::compact_mode = cm, kw::opt_level = opt_level,
                    kw::pars = {mppp::real{"1.3", prec}, mppp::real{"2.7", prec}});

                for (auto i = 0; i < 3; ++i) {
                    auto [oc, h] = ta.step();
                    REQUIRE(oc == taylor_outcome::success);
                }
                REQUIRE(ta.get_state()[0] == approximately(exp(ta.get_time())));
                REQUIRE(ta.get_state()[1]
                        == approximately(mppp::real{2, prec} + (ta.get_pars()[0] + ta.get_pars()[1]) * ta.get_time()));
                REQUIRE(ta.get_state()[2]
                        == approximately(mppp::real{3, prec} + ta.get_time() * ta.get_time() / mppp::real{2, prec}));
                REQUIRE(ta.get_state()[3]
                        == approximately(mppp::real{4, prec} + mppp::real{"1.5", prec} * ta.get_time()));
            }
        }
    }
}

// Failure modes in the stepper.
TEST_CASE("step")
{
    using Catch::Matchers::Message;

    auto [x] = make_vars("x");

    const auto prec = 30u;

    auto ta = taylor_adaptive<mppp::real>({prime(x) = x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = true);

    REQUIRE_THROWS_MATCHES(
        ta.step(mppp::real{1, 31}), std::invalid_argument,
        Message("Invalid max_delta_t argument passed to the step() function of an adaptive Taylor "
                "integrator: max_delta_t has a precision of 31, while the integrator's precision is 30"));

    // Take a step, then change state in incompatible way.
    ta.step();
    ta.get_state_data()[0].prec_round(prec + 1);

    auto old_dtime = ta.get_dtime();

    REQUIRE_THROWS_MATCHES(ta.step(), std::invalid_argument,
                           Message("A state variable with precision 31 was detected in the state "
                                   "vector: this is incompatible with the integrator precision of 30"));

    REQUIRE(ta.get_dtime() == old_dtime);

    ta.get_state_data()[0].prec_round(prec);

    // Same with the par.
    ta.step();
    ta.get_pars_data()[0].prec_round(prec + 1);

    old_dtime = ta.get_dtime();

    REQUIRE_THROWS_MATCHES(ta.step(), std::invalid_argument,
                           Message("A value with precision 31 was detected in the parameter "
                                   "vector: this is incompatible with the integrator precision of 30"));

    REQUIRE(ta.get_dtime() == old_dtime);
}

// Failure modes in time setting.
TEST_CASE("time set")
{
    using Catch::Matchers::Message;

    auto [x] = make_vars("x");

    const auto prec = 30u;

    auto ta = taylor_adaptive<mppp::real>({prime(x) = x}, {mppp::real{1, prec}}, kw::compact_mode = true);

    REQUIRE_THROWS_MATCHES(
        ta.set_time(mppp::real{1, 31}), std::invalid_argument,
        Message("Invalid precision detected in the time variable: the precision of the integrator is "
                "30, but the time variable has a precision of 31 instead"));

    REQUIRE_THROWS_MATCHES(
        ta.set_dtime(mppp::real{1, 31}, mppp::real{0, 31}), std::invalid_argument,
        Message("Invalid precision detected in the time variable: the precision of the integrator is "
                "30, but the time variable has a precision of 31 instead"));
    REQUIRE_THROWS_MATCHES(
        ta.set_dtime(mppp::real{1, 30}, mppp::real{0, 31}), std::invalid_argument,
        Message("Mismatched precisions in the components of a dfloat<mppp::real>: the high component has a "
                "precision of 30, while the low component has a precision of 31"));
}

// Test that precision is preserved when copying/moving an integrator object.
TEST_CASE("copy move prec")
{
    auto [x] = make_vars("x");

    const auto prec = 30u;

    auto ta = taylor_adaptive<mppp::real>({prime(x) = x}, {mppp::real{1, prec}}, kw::compact_mode = true);

    // Copy ctor.
    auto ta2 = ta;
    REQUIRE(ta2.get_prec() == prec);

    // Move ctor.
    auto ta3(std::move(ta2));
    REQUIRE(ta3.get_prec() == prec);

    // Copy assignment.
    auto ta4 = taylor_adaptive<mppp::real>({prime(x) = x}, {mppp::real{1, prec + 1u}}, kw::compact_mode = true);
    ta = ta4;
    REQUIRE(ta.get_prec() == prec + 1u);

    // Move assignment.
    ta2 = std::move(ta4);
    REQUIRE(ta2.get_prec() == prec + 1u);
}

// Test failure mode in taylor_add_jet.
TEST_CASE("taylor_add_jet prec")
{
    using Catch::Matchers::Message;

    auto [x] = make_vars("x");

    llvm_state s;

    REQUIRE_THROWS_MATCHES(taylor_add_jet<mppp::real>(s, "jet", {x}, 2, 1, false, false, {}, false),
                           std::invalid_argument,
                           Message(fmt::format("An invalid precision value of 0 was passed to taylor_add_jet() (the "
                                               "value must be in the [{}, {}] range)",
                                               mppp::real_prec_min(), mppp::real_prec_max())));
}

// Dense output.
TEST_CASE("dense out")
{
    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            for (auto cm : {false, true}) {
                auto ta = taylor_adaptive<mppp::real>({prime(x) = v, prime(v) = -x},
                                                      {mppp::real{0, prec}, mppp::real{1, prec}}, kw::compact_mode = cm,
                                                      kw::opt_level = opt_level);

                auto [oc, h] = ta.step(true);

                REQUIRE(oc == taylor_outcome::success);

                ta.update_d_output(mppp::real{0, prec});

                REQUIRE(ta.get_d_output()[0] == 0);
                REQUIRE(ta.get_d_output()[1] == 1);

                ta.update_d_output(h);

                REQUIRE(ta.get_d_output()[0] == approximately(sin(h)));
                REQUIRE(ta.get_d_output()[1] == approximately(cos(h)));

                ta.update_d_output(h / mppp::real{2, prec});

                REQUIRE(ta.get_d_output()[0] == approximately(sin(h / mppp::real{2, prec})));
                REQUIRE(ta.get_d_output()[1] == approximately(cos(h / mppp::real{2, prec})));

                // Failure mode.
                REQUIRE_THROWS_MATCHES(
                    ta.update_d_output(mppp::real{0, prec - 1}), std::invalid_argument,
                    Message(fmt::format(
                        "Invalid time variable passed to update_d_output(): the time variable has a precision of "
                        "{}, while the integrator has a precision of {}",
                        prec - 1, prec)));
            }
        }
    }
}
