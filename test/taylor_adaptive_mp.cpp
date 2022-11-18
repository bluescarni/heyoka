// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <initializer_list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "catch.hpp"
#include "heyoka/kw.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

template <typename Out, typename P, typename T>
auto &horner_eval(Out &ret, const P &p, int order, const T &eval)
{
    ret = xt::view(p, xt::all(), order);

    for (--order; order >= 0; --order) {
        ret = xt::view(p, xt::all(), order) + ret * eval;
    }

    return ret;
}

static std::mt19937 rng;

// Tests to check precision handling on construction.
TEST_CASE("ctors prec")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    const auto prec = 30;

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
        REQUIRE(ta.get_time().get_prec() == prec);
        REQUIRE(ta.get_tol() == detail::eps_from_prec(prec));
        REQUIRE(ta.get_tol().get_prec() == prec);
        REQUIRE(ta.get_pars()[0] == 0);
        REQUIRE(ta.get_pars()[0].get_prec() == prec);
        REQUIRE(std::all_of(ta.get_tc().begin(), ta.get_tc().end(),
                            [prec](const auto &v) { return v == 0 && v.get_prec() == prec; }));
        REQUIRE(std::all_of(ta.get_d_output().begin(), ta.get_d_output().end(),
                            [prec](const auto &v) { return v == 0 && v.get_prec() == prec; }));
        REQUIRE(ta.get_last_h() == 0);
        REQUIRE(ta.get_last_h().get_prec() == prec);

        // Check that tolerance is autocast to the correct precision.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::tol = mppp::real(1e-1));
        REQUIRE(ta.get_tol() == mppp::real(1e-1, prec));
        REQUIRE(ta.get_tol().get_prec() == prec);

        // Check with explicit time.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::time = mppp::real(1, prec));
        REQUIRE(ta.get_time() == 1);
        REQUIRE(ta.get_time().get_prec() == prec);

        // Check wrong time prec is automatically corrected.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::time = mppp::real{0, prec + 1});
        REQUIRE(ta.get_dtime().first.get_prec() == prec);
        REQUIRE(ta.get_dtime().second.get_prec() == prec);

        // Wrong precision in the state vector with automatic deduction.
        REQUIRE_THROWS_MATCHES(
            taylor_adaptive<mppp::real>({x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec + 1}},
                                        kw::compact_mode = cm, kw::opt_level = 0u),
            std::invalid_argument,
            Message("The precision deduced automatically from the initial state vector in a multiprecision "
                    "adaptive Taylor integrator is 30, but values with different precision(s) were "
                    "detected in the state vector"));

        // Check that wrong precision in the pars if automatically corrected.
        ta = taylor_adaptive<mppp::real>({x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec}},
                                         kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::pars = {mppp::real{1, prec}, mppp::real{1, prec + 1}});
        REQUIRE(std::all_of(ta.get_pars().begin(), ta.get_pars().end(),
                            [prec](const auto &val) { return val.get_prec() == prec; }));

        // Checks with precision provided explicitly by the user.
        ta = taylor_adaptive<mppp::real>({x + par[0]}, {mppp::real{1, prec}}, kw::compact_mode = cm, kw::opt_level = 0u,
                                         kw::prec = prec + 1);

        // Check that time and state are automatically adjusted.
        REQUIRE(ta.get_dtime().first.get_prec() == prec + 1);
        REQUIRE(ta.get_dtime().second.get_prec() == prec + 1);
        REQUIRE(std::all_of(ta.get_state().begin(), ta.get_state().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));
        REQUIRE(std::all_of(ta.get_pars().begin(), ta.get_pars().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));

        // Check that it does not matter if the state vector has different precisions.
        ta = taylor_adaptive<mppp::real>({x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec - 1}},
                                         kw::compact_mode = cm, kw::opt_level = 0u, kw::prec = prec + 1);
        REQUIRE(ta.get_dtime().first.get_prec() == prec + 1);
        REQUIRE(ta.get_dtime().second.get_prec() == prec + 1);
        REQUIRE(std::all_of(ta.get_state().begin(), ta.get_state().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));
        REQUIRE(std::all_of(ta.get_pars().begin(), ta.get_pars().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));

        // Try with several different precisions for the data.
        ta = taylor_adaptive<mppp::real>({x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec - 1}},
                                         kw::compact_mode = cm, kw::opt_level = 0u, kw::prec = prec + 1,
                                         kw::pars = {mppp::real{-1, prec - 2}}, kw::time = mppp::real{0, prec + 3},
                                         kw::tol = mppp::real{1e-1, prec + 4});
        REQUIRE(ta.get_tol().get_prec() == prec + 1);
        REQUIRE(ta.get_dtime().first.get_prec() == prec + 1);
        REQUIRE(ta.get_dtime().second.get_prec() == prec + 1);
        REQUIRE(std::all_of(ta.get_state().begin(), ta.get_state().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));
        REQUIRE(std::all_of(ta.get_pars().begin(), ta.get_pars().end(),
                            [prec](const auto &val) { return val.get_prec() == prec + 1; }));

        // Try with bogus precision values.
        REQUIRE_THROWS_AS(taylor_adaptive<mppp::real>({x + par[0], y + par[1]},
                                                      {mppp::real{1, prec}, mppp::real{1, prec - 1}},
                                                      kw::compact_mode = cm, kw::opt_level = 0u, kw::prec = -1),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(taylor_adaptive<mppp::real>(
                              {x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec - 1}},
                              kw::compact_mode = cm, kw::opt_level = 0u, kw::prec = mppp::real_prec_max() + 1),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(taylor_adaptive<mppp::real>(
                              {x + par[0], y + par[1]}, {mppp::real{1, prec}, mppp::real{1, prec - 1}},
                              kw::compact_mode = cm, kw::opt_level = 0u, kw::prec = mppp::real_prec_min() - 1),
                          std::invalid_argument);
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
                    REQUIRE(h.get_prec() == prec);
                    REQUIRE(ta.get_last_h().get_prec() == prec);
                    REQUIRE(ta.get_time().get_prec() == prec);
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

                REQUIRE(h.get_prec() == prec);
                REQUIRE(ta.get_last_h().get_prec() == prec);
                REQUIRE(ta.get_time().get_prec() == prec);

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

TEST_CASE("taylor tc basic")
{
    using fp_t = mppp::real;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto [x, v] = make_vars("x", "v");

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0.05, prec), fp_t(0.025, prec)},
                                            kw::compact_mode = true,
                                            kw::opt_level = opt_level};

            REQUIRE(ta.get_tc().size() == 2u * (ta.get_order() + 1u));

            auto tca = xt::adapt(ta.get_tc().data(), {2u, ta.get_order() + 1u});

            auto [oc, h] = ta.step(true);

            auto ret = xt::eval(xt::zeros<mppp::real>({2}));

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), fp_t(0, prec));
            REQUIRE(ret[0] == approximately(fp_t(0.05, prec), fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(fp_t(0.025, prec), fp_t(10., prec)));

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
            REQUIRE(ret[0] == approximately(ta.get_state()[0], fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(ta.get_state()[1], fp_t(10., prec)));

            auto old_state = ta.get_state();

            std::tie(oc, h) = ta.step_backward(true);

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), fp_t(0, prec));
            REQUIRE(ret[0] == approximately(old_state[0], fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(old_state[1], fp_t(10., prec)));

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
            REQUIRE(ret[0] == approximately(ta.get_state()[0], fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(ta.get_state()[1], fp_t(10., prec)));

            old_state = ta.get_state();

            std::tie(oc, h) = ta.step(fp_t(1e-3, prec), true);

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), fp_t(0, prec));
            REQUIRE(ret[0] == approximately(old_state[0], fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(old_state[1], fp_t(10., prec)));

            horner_eval(ret, tca, static_cast<int>(ta.get_order()), h);
            REQUIRE(ret[0] == approximately(ta.get_state()[0], fp_t(10., prec)));
            REQUIRE(ret[1] == approximately(ta.get_state()[1], fp_t(10., prec)));
        }
    }
}

// A test to make sure the propagate functions deal correctly
// with trivial dynamics.
TEST_CASE("propagate trivial")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = 1_dbl},
                                            {fp_t(0, prec), fp_t(0, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            REQUIRE(std::get<0>(ta.propagate_for(fp_t(1.2, prec))) == taylor_outcome::time_limit);
            REQUIRE(std::get<0>(ta.propagate_until(fp_t(2.3, prec))) == taylor_outcome::time_limit);
            REQUIRE(std::get<0>(
                        ta.propagate_grid({fp_t(3, prec), fp_t(4, prec), fp_t(5, prec), fp_t(6, prec), fp_t(7., prec)}))
                    == taylor_outcome::time_limit);
        }
    }
}

TEST_CASE("propagate for_until")
{
    using Catch::Matchers::Message;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0.05, prec), fp_t(0.025, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};
            auto ta_copy = ta;

            // Error modes.
            REQUIRE_THROWS_MATCHES(
                ta.propagate_until(fp_t(std::numeric_limits<double>::infinity(), prec)), std::invalid_argument,
                Message(
                    "A non-finite time was passed to the propagate_until() function of an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_until(fp_t(1, prec - 1)), std::invalid_argument,
                Message(fmt::format(
                    "Invalid final time argument passed to the propagate_until() function of an adaptive Taylor "
                    "integrator: the time variable has a precision of {}, while the integrator's precision is {}",
                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_until(fp_t(10., prec),
                                   kw::max_delta_t = fp_t(std::numeric_limits<double>::quiet_NaN(), prec)),
                std::invalid_argument,
                Message(
                    "A nan max_delta_t was passed to the propagate_until() function of an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_until(fp_t(10., prec), kw::max_delta_t = fp_t(-1, prec)), std::invalid_argument,
                Message("A non-positive max_delta_t was passed to the propagate_until() function of an "
                        "adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_until(fp_t(10., prec), kw::max_delta_t = fp_t(1, prec + 1)), std::invalid_argument,
                Message(fmt::format(
                    "Invalid max_delta_t argument passed to the propagate_until() function of an adaptive Taylor "
                    "integrator: max_delta_t has a precision of {}, while the integrator's precision is {}",
                    prec + 1, prec)));

            // Propagate forward in time limiting the timestep size and passing in a callback.
            auto counter = 0ul;

            auto oc = std::get<0>(ta.propagate_until(
                fp_t(1., prec), kw::max_delta_t = fp_t(1e-4, prec), kw::callback = [&counter](auto &) {
                    ++counter;
                    return true;
                }));
            auto oc_copy = std::get<0>(ta_copy.propagate_until(fp_t(1., prec)));

            REQUIRE(ta.get_time() == 1.);
            // NOTE: at low precision the step counting
            // results in 10001 rather than 10000.
            if (prec > 30) {
                REQUIRE(counter == 10000ul);
            }
            REQUIRE(oc == taylor_outcome::time_limit);

            REQUIRE(ta_copy.get_time() == 1.);
            REQUIRE(oc_copy == taylor_outcome::time_limit);

            REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], fp_t(1000., prec)));
            REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], fp_t(1000., prec)));

            // Do propagate_for() too.
            oc = std::get<0>(ta.propagate_for(
                fp_t(1., prec), kw::max_delta_t = fp_t(1e-4, prec), kw::callback = [&counter](auto &) {
                    ++counter;
                    return true;
                }));
            oc_copy = std::get<0>(ta_copy.propagate_for(fp_t(1., prec)));

            REQUIRE(ta.get_time() == 2.);
            if (prec > 30) {
                REQUIRE(counter == 20000ul);
            }
            REQUIRE(oc == taylor_outcome::time_limit);

            REQUIRE(ta_copy.get_time() == 2.);
            REQUIRE(oc_copy == taylor_outcome::time_limit);

            REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], fp_t(1000., prec)));
            REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], fp_t(1000., prec)));

            // Do backwards in time too.
            oc = std::get<0>(ta.propagate_for(
                -fp_t(1., prec), kw::max_delta_t = fp_t(1e-4, prec), kw::callback = [&counter](auto &) {
                    ++counter;
                    return true;
                }));
            oc_copy = std::get<0>(ta_copy.propagate_for(-fp_t(1., prec)));

            REQUIRE(ta.get_time() == 1.);
            if (prec > 30) {
                REQUIRE(counter == 30000ul);
            }
            REQUIRE(oc == taylor_outcome::time_limit);

            REQUIRE(ta_copy.get_time() == 1.);
            REQUIRE(oc_copy == taylor_outcome::time_limit);

            REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], fp_t(1000., prec)));
            REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], fp_t(1000., prec)));

            oc = std::get<0>(ta.propagate_until(
                fp_t(0., prec), kw::max_delta_t = fp_t(1e-4, prec), kw::callback = [&counter](auto &) {
                    ++counter;
                    return true;
                }));
            oc_copy = std::get<0>(ta_copy.propagate_until(fp_t(0., prec)));

            REQUIRE(ta.get_time() == 0.);
            if (prec > 30) {
                REQUIRE(counter == 40000ul);
            }
            REQUIRE(oc == taylor_outcome::time_limit);

            REQUIRE(ta_copy.get_time() == 0.);
            REQUIRE(oc_copy == taylor_outcome::time_limit);

            REQUIRE(ta.get_state()[0] == approximately(ta_copy.get_state()[0], fp_t(1000., prec)));
            REQUIRE(ta.get_state()[1] == approximately(ta_copy.get_state()[1], fp_t(1000., prec)));

            // Pass a callback that changes the precision of the internal data.
            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(
                    fp_t(1., prec), kw::callback =
                                        [prec](auto &tint) {
                                            tint.get_state_data()[0].prec_round(prec - 1);
                                            return true;
                                        }),
                std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_for(
                    fp_t(1., prec), kw::callback =
                                        [prec](auto &tint) {
                                            tint.get_state_data()[0].prec_round(prec - 1);
                                            return true;
                                        }),
                std::invalid_argument,
                Message(fmt::format("A state variable with precision {} was detected in the state "
                                    "vector: this is incompatible with the integrator precision of {}",
                                    prec - 1, prec)));
        }
    }
}

TEST_CASE("propagate for_until write_tc")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0.05, prec), fp_t(0.025, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            ta.propagate_until(
                fp_t(10, prec), kw::callback = [&](auto &t) {
                    REQUIRE(
                        std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &val) { return val == 0.; }));
                    REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(),
                                        [prec](const auto &val) { return val.get_prec() == prec; }));
                    return true;
                });

            ta.propagate_until(
                fp_t(20, prec), kw::write_tc = true, kw::callback = [&](auto &t) {
                    REQUIRE(
                        !std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &val) { return val == 0.; }));
                    REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(),
                                        [prec](const auto &val) { return val.get_prec() == prec; }));
                    return true;
                });

            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0.05, prec), fp_t(0.025, prec)},
                                       kw::compact_mode = true};

            ta.propagate_for(
                fp_t(10, prec), kw::callback = [&](auto &t) {
                    REQUIRE(
                        std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &val) { return val == 0.; }));
                    REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(),
                                        [prec](const auto &val) { return val.get_prec() == prec; }));
                    return true;
                });

            ta.propagate_for(
                fp_t(20, prec), kw::write_tc = true, kw::callback = [&](auto &t) {
                    REQUIRE(
                        !std::all_of(t.get_tc().begin(), t.get_tc().end(), [](const auto &val) { return val == 0.; }));
                    REQUIRE(std::all_of(t.get_tc().begin(), t.get_tc().end(),
                                        [prec](const auto &val) { return val.get_prec() == prec; }));
                    return true;
                });
        }
    }
}

TEST_CASE("propagate grid scalar")
{
    using fp_t = mppp::real;

    using Catch::Matchers::Message;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0.05, prec), fp_t(0.025, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({}), std::invalid_argument,
                Message("Cannot invoke propagate_grid() in an adaptive Taylor integrator if the time grid is empty"));

            ta.set_time(fp_t(std::numeric_limits<double>::infinity(), prec));

            REQUIRE_THROWS_MATCHES(ta.propagate_grid({0.}), std::invalid_argument,
                                   Message("Cannot invoke propagate_grid() in an adaptive Taylor integrator if the "
                                           "current time is not finite"));

            ta.set_time(fp_t(0., prec));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(std::numeric_limits<double>::infinity(), prec)}), std::invalid_argument,
                Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(1., prec), fp_t(std::numeric_limits<double>::infinity(), prec)}),
                std::invalid_argument,
                Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(1., prec), fp_t(2., prec), fp_t(1., prec)}), std::invalid_argument,
                Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({-fp_t(1., prec), -fp_t(2., prec), fp_t(1., prec)}), std::invalid_argument,
                Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(0., prec), fp_t(0., prec), fp_t(1., prec)}), std::invalid_argument,
                Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));
            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(0., prec), fp_t(1., prec), fp_t(1., prec)}), std::invalid_argument,
                Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));
            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(0., prec), fp_t(1., prec), fp_t(2., prec), fp_t(2., prec)}),
                std::invalid_argument,
                Message("A non-monotonic time grid was passed to propagate_grid() in an adaptive Taylor integrator"));

            // Set an infinity in the state.
            ta.get_state_data()[0] = fp_t(std::numeric_limits<double>::infinity(), prec);

            auto out = ta.propagate_grid({fp_t(.2, prec)});
            REQUIRE(std::get<0>(out) == taylor_outcome::err_nf_state);
            REQUIRE(std::get<4>(out).empty());

            // Error modes specific to mppp::real.
            ta.set_time(fp_t(0, prec));

            // max_delta_t with wrong precision.
            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(.2, prec)}, kw::max_delta_t = fp_t(.1, prec - 1)), std::invalid_argument,
                Message(fmt::format(
                    "Invalid max_delta_t argument passed to the propagate_grid() function of an adaptive Taylor "
                    "integrator: max_delta_t has a precision of {}, while the integrator's precision is {}",
                    prec - 1, prec)));

            // Wrong precisions in the grid.
            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(.2, prec + 1)}), std::invalid_argument,
                Message(
                    fmt::format("Invalid precision detected in the time grid passed to the propagate_grid() function "
                                "of an adaptive Taylor integrator: a value of precision {} was "
                                "detected in the grid, but the precision of the integrator is {} instead",
                                prec + 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(.2, prec), fp_t(.3, prec - 1)}), std::invalid_argument,
                Message(
                    fmt::format("Invalid precision detected in the time grid passed to the propagate_grid() function "
                                "of an adaptive Taylor integrator: a value of precision {} was "
                                "detected in the grid, but the precision of the integrator is {} instead",
                                prec - 1, prec)));

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(.2, prec), fp_t(.3, prec), fp_t(.4, prec + 2)}), std::invalid_argument,
                Message(
                    fmt::format("Invalid precision detected in the time grid passed to the propagate_grid() function "
                                "of an adaptive Taylor integrator: a value of precision {} was "
                                "detected in the grid, but the precision of the integrator is {} instead",
                                prec + 2, prec)));

            // Reset the integrator.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0.05, prec), fp_t(0.025, prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            // Propagate to the initial time.
            out = ta.propagate_grid({fp_t(0., prec)});
            REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
            REQUIRE(std::get<1>(out) == std::numeric_limits<double>::infinity());
            REQUIRE(std::get<2>(out) == 0);
            REQUIRE(std::get<3>(out) == 0u);
            REQUIRE(std::get<4>(out) == std::vector{fp_t(0.05, prec), fp_t(0.025, prec)});
            REQUIRE(ta.get_time() == 0.);

            REQUIRE_THROWS_MATCHES(
                ta.propagate_grid({fp_t(2., prec), fp_t(std::numeric_limits<double>::infinity(), prec)}),
                std::invalid_argument,
                Message("A non-finite time value was passed to propagate_grid() in an adaptive Taylor integrator"));

            // Switch to the harmonic oscillator.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                       {fp_t(0., prec), fp_t(1., prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            // Integrate forward over a dense grid from 0 to 10.
            std::vector<fp_t> grid;
            for (auto i = 0u; i < 1000u; ++i) {
                grid.emplace_back(i / 100., prec);
            }
            out = ta.propagate_grid(grid);

            REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
            REQUIRE(std::get<4>(out).size() == 2000u);
            REQUIRE(ta.get_time() == grid.back());

            for (auto i = 0u; i < 1000u; ++i) {
                REQUIRE(std::get<4>(out)[2u * i] == approximately(sin(grid[i]), fp_t(10000., prec)));
                REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(cos(grid[i]), fp_t(10000., prec)));
            }

            // Do the same backwards.
            ta.set_time(fp_t(10., prec));
            ta.get_state_data()[0] = sin(fp_t(10., prec));
            ta.get_state_data()[1] = cos(fp_t(10., prec));
            std::reverse(grid.begin(), grid.end());

            out = ta.propagate_grid(grid);

            REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
            REQUIRE(std::get<4>(out).size() == 2000u);
            REQUIRE(ta.get_time() == grid.back());

            for (auto i = 0u; i < 1000u; ++i) {
                REQUIRE(std::get<4>(out)[2u * i] == approximately(sin(grid[i]), fp_t(10000., prec)));
                REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(cos(grid[i]), fp_t(10000., prec)));
            }

            // Random testing.
            ta.set_time(fp_t(0., prec));
            ta.get_state_data()[0] = fp_t(0., prec);
            ta.get_state_data()[1] = fp_t(1., prec);

            std::uniform_real_distribution<double> rdist(0., .1);
            grid[0] = fp_t(0, prec);
            for (auto i = 1u; i < 1000u; ++i) {
                grid[i] = grid[i - 1u] + fp_t(rdist(rng), prec);
            }

            out = ta.propagate_grid(grid);

            REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
            REQUIRE(std::get<4>(out).size() == 2000u);
            REQUIRE(ta.get_time() == grid.back());

            for (auto i = 0u; i < 1000u; ++i) {
                REQUIRE(std::get<4>(out)[2u * i] == approximately(sin(grid[i]), fp_t(100000., prec)));
                REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(cos(grid[i]), fp_t(100000., prec)));
            }

            // Do it also backwards.
            ta.set_time(fp_t(0., prec));
            ta.get_state_data()[0] = fp_t(0., prec);
            ta.get_state_data()[1] = fp_t(1., prec);

            rdist = std::uniform_real_distribution<double>(-.1, 0.);
            grid[0] = fp_t(0, prec);
            for (auto i = 1u; i < 1000u; ++i) {
                grid[i] = grid[i - 1u] + fp_t(rdist(rng), prec);
            }

            out = ta.propagate_grid(grid);

            REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
            REQUIRE(std::get<4>(out).size() == 2000u);
            REQUIRE(ta.get_time() == grid.back());

            for (auto i = 0u; i < 1000u; ++i) {
                REQUIRE(std::get<4>(out)[2u * i] == approximately(sin(grid[i]), fp_t(100000., prec)));
                REQUIRE(std::get<4>(out)[2u * i + 1u] == approximately(cos(grid[i]), fp_t(100000., prec)));
            }

            // A test with a sparse grid.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                       {fp_t(0., prec), fp_t(1., prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            out = ta.propagate_grid({fp_t(.1, prec), fp_t(10., prec), fp_t(100., prec)});

            REQUIRE(std::get<4>(out).size() == 6u);
            REQUIRE(ta.get_time() == 100.);
            REQUIRE(std::get<4>(out)[0] == approximately(sin(fp_t(.1, prec)), fp_t(100., prec)));
            REQUIRE(std::get<4>(out)[1] == approximately(cos(fp_t(.1, prec)), fp_t(100., prec)));
            REQUIRE(std::get<4>(out)[2] == approximately(sin(fp_t(10, prec)), fp_t(100., prec)));
            REQUIRE(std::get<4>(out)[3] == approximately(cos(fp_t(10, prec)), fp_t(100, prec)));
            REQUIRE(std::get<4>(out)[4] == approximately(sin(fp_t(100, prec)), fp_t(1000, prec)));
            REQUIRE(std::get<4>(out)[5] == approximately(cos(fp_t(100, prec)), fp_t(1000, prec)));

            // A case in which the initial propagate_until() to bring the system
            // to grid[0] interrupts the integration.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                       {fp_t(0., prec), fp_t(1., prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true,
                                       kw::t_events = {t_event<fp_t>(v - 0.999)}};
            out = ta.propagate_grid({fp_t(10., prec), fp_t(100., prec)});
            REQUIRE(std::get<0>(out) == taylor_outcome{-1});
            REQUIRE(std::get<4>(out).empty());

            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -x},
                {fp_t(0., prec), fp_t(1., prec)},
                kw::opt_level = opt_level,
                kw::compact_mode = true,
                kw::t_events = {t_event<fp_t>(
                    v - 0.999, kw::callback = [](taylor_adaptive<fp_t> &, bool, int) { return false; })}};
            out = ta.propagate_grid({fp_t(10., prec), fp_t(100., prec)});
            REQUIRE(std::get<0>(out) == taylor_outcome{-1});
            REQUIRE(std::get<4>(out).empty());

            // A case in which we have a callback which never stops and a terminal event
            // which triggers.
            ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -x},
                {fp_t(0., prec), fp_t(1., prec)},
                kw::opt_level = opt_level,
                kw::compact_mode = true,
                kw::t_events = {t_event<fp_t>(
                    v - .1, kw::callback = [](taylor_adaptive<fp_t> &, bool, int) { return false; })}};
            out = ta.propagate_grid(
                {fp_t(10., prec), fp_t(100., prec)}, kw::callback = [](const auto &) { return true; });
            REQUIRE(std::get<0>(out) == taylor_outcome{-1});
        }
    }
}

TEST_CASE("continuous output")
{
    using std::cos;
    using std::sin;

    using Catch::Matchers::Message;

    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            continuous_output<fp_t> co;

            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                            {fp_t(0., prec), fp_t(1., prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            auto [_0, _1, _2, tot_steps, d_out] = ta.propagate_until(fp_t(10., prec), kw::c_output = true);

            REQUIRE(d_out.has_value());
            REQUIRE(d_out->get_output().size() == 2u);
            REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
            REQUIRE(!d_out->get_tcs().empty());
            REQUIRE(!d_out->get_llvm_state().get_ir().empty());
            REQUIRE(tot_steps == d_out->get_n_steps());

            // Reset time/state.
            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(1, prec);
            ta.set_time(fp_t(0, prec));

            // Run a grid propagation.
            auto t_grid = std::vector<fp_t>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
            for (auto &vec : t_grid) {
                vec.prec_round(prec);
            }
            auto grid_out = std::get<4>(ta.propagate_grid(t_grid));

            // Compare the two.
            for (auto i = 0u; i < 11u; ++i) {
                (*d_out)(t_grid[i]);
                REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10, prec)));
                REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10, prec)));
            }

            REQUIRE(d_out->get_bounds().first == 0.);
            REQUIRE(d_out->get_bounds().second == approximately(fp_t(10, prec)));
            REQUIRE(d_out->get_n_steps() > 0u);

            // Try slightly outside the bounds.
            (*d_out)(fp_t(-.01, prec));
            REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-0.01, prec))));
            REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-0.01, prec))));
            (*d_out)(fp_t(10.01, prec));
            REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(10.01, prec))));
            REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(10.01, prec))));

            // Try making a copy too.
            auto co3 = *d_out;
            co3(fp_t(4., prec));
            REQUIRE(co3.get_output()[0] == approximately(grid_out[2u * 4u], fp_t(10, prec)));
            REQUIRE(co3.get_output()[1] == approximately(grid_out[2u * 4u + 1u], fp_t(10, prec)));

            // Limiting case in which not steps are taken.
            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(std::numeric_limits<double>::infinity(), prec);
            ta.set_time(fp_t(0, prec));
            d_out = std::get<4>(ta.propagate_until(fp_t(10., prec), kw::c_output = true));
            REQUIRE(!d_out.has_value());

            // Try with propagate_for() too.

            // Reset time/state.
            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(1, prec);
            ta.set_time(fp_t(0, prec));

            std::tie(_0, _1, _2, tot_steps, d_out) = ta.propagate_for(fp_t(10., prec), kw::c_output = true);

            REQUIRE(d_out.has_value());
            REQUIRE(d_out->get_output().size() == 2u);
            REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
            REQUIRE(!d_out->get_tcs().empty());
            REQUIRE(!d_out->get_llvm_state().get_ir().empty());
            REQUIRE(tot_steps == d_out->get_n_steps());

            // Compare the two.
            for (auto i = 0u; i < 11u; ++i) {
                (*d_out)(t_grid[i]);
                REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10, prec)));
                REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10, prec)));
            }

            // Do it backwards too.
            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(1, prec);
            ta.set_time(fp_t(0, prec));

            d_out = std::get<4>(ta.propagate_until(fp_t(-10., prec), kw::c_output = true));

            REQUIRE(d_out.has_value());
            REQUIRE(d_out->get_times().size() == d_out->get_n_steps() + 1u);
            REQUIRE(!d_out->get_tcs().empty());

            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(1, prec);
            ta.set_time(fp_t(0, prec));

            // Run a grid propagation.
            t_grid = std::vector<fp_t>{0., -1., -2., -3., -4., -5., -6., -7., -8., -9., -10.};
            for (auto &vec : t_grid) {
                vec.prec_round(prec);
            }
            grid_out = std::get<4>(ta.propagate_grid(t_grid));

            // Compare the two.
            for (auto i = 0u; i < 11u; ++i) {
                (*d_out)(t_grid[i]);
                REQUIRE(d_out->get_output()[0] == approximately(grid_out[2u * i], fp_t(10, prec)));
                REQUIRE(d_out->get_output()[1] == approximately(grid_out[2u * i + 1u], fp_t(10, prec)));
            }

            REQUIRE(d_out->get_bounds().first == 0.);
            REQUIRE(d_out->get_bounds().second == approximately(fp_t(-10, prec)));
            REQUIRE(d_out->get_n_steps() > 0u);

            // Try slightly outside the bounds.
            (*d_out)(fp_t(.01, prec));
            REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(0.01, prec))));
            REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(0.01, prec))));
            (*d_out)(fp_t(-10.01, prec));
            REQUIRE(d_out->get_output()[0] == approximately(sin(fp_t(-10.01, prec))));
            REQUIRE(d_out->get_output()[1] == approximately(cos(fp_t(-10.01, prec))));

            // Try making a copy too.
            co = *d_out;
            co(fp_t(-4., prec));
            REQUIRE(co.get_output()[0] == approximately(grid_out[2u * 4u], fp_t(10, prec)));
            REQUIRE(co.get_output()[1] == approximately(grid_out[2u * 4u + 1u], fp_t(10, prec)));

            co = *&co;
            co(fp_t(-5., prec));
            REQUIRE(co.get_output()[0] == approximately(grid_out[2u * 5u], fp_t(10, prec)));
            REQUIRE(co.get_output()[1] == approximately(grid_out[2u * 5u + 1u], fp_t(10, prec)));

            // Limiting case in which not steps are taken.
            ta.get_state_data()[0] = fp_t(0, prec);
            ta.get_state_data()[1] = fp_t(std::numeric_limits<double>::infinity(), prec);
            ta.set_time(fp_t(0, prec));
            d_out = std::get<4>(ta.propagate_until(fp_t(-10., prec), kw::c_output = true));
            REQUIRE(!d_out.has_value());

            // Try with non-finite time.
            REQUIRE_THROWS_AS(co(fp_t(std::numeric_limits<double>::infinity(), prec)), std::invalid_argument);

            // Failure modes specific to mppp::real.
            REQUIRE_THROWS_MATCHES(
                co(fp_t(1, prec - 1)), std::invalid_argument,
                Message(
                    fmt::format("Invalid precision detected for the time argument in a continuous output functor: the "
                                "precision of the time coordinate is {}, but the precision of the internal data is {}",
                                prec - 1, prec)));
        }
    }
}

TEST_CASE("s11n")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                            {fp_t(0., prec), fp_t(0.5, prec)},
                                            kw::compact_mode = true,
                                            kw::opt_level = opt_level};

            REQUIRE(std::get<0>(ta.step(true)) == taylor_outcome::success);

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);
                oa << ta;
            }

            auto ta_copy = ta;
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                       {fp_t(0., prec), fp_t(0.5, prec)},
                                       kw::compact_mode = true,
                                       kw::opt_level = opt_level};

            {
                boost::archive::binary_iarchive ia(ss);
                ia >> ta;
            }

            REQUIRE(ta.get_llvm_state().get_ir() == ta_copy.get_llvm_state().get_ir());
            REQUIRE(ta.get_decomposition() == ta_copy.get_decomposition());
            REQUIRE(ta.get_order() == ta_copy.get_order());
            REQUIRE(ta.get_tol() == ta_copy.get_tol());
            REQUIRE(ta.get_high_accuracy() == ta_copy.get_high_accuracy());
            REQUIRE(ta.get_compact_mode() == ta_copy.get_compact_mode());
            REQUIRE(ta.get_dim() == ta_copy.get_dim());
            REQUIRE(ta.get_time() == ta_copy.get_time());
            REQUIRE(ta.get_state() == ta_copy.get_state());
            REQUIRE(ta.get_pars() == ta_copy.get_pars());
            REQUIRE(ta.get_tc() == ta_copy.get_tc());
            REQUIRE(ta.get_last_h() == ta_copy.get_last_h());
            REQUIRE(ta.get_d_output() == ta_copy.get_d_output());
            REQUIRE(ta.get_prec() == ta_copy.get_prec());

            // Take a step in ta and in ta_copy.
            ta.step(true);
            ta_copy.step(true);

            REQUIRE(ta.get_time() == ta_copy.get_time());
            REQUIRE(ta.get_state() == ta_copy.get_state());
            REQUIRE(ta.get_tc() == ta_copy.get_tc());
            REQUIRE(ta.get_last_h() == ta_copy.get_last_h());

            ta.update_d_output(fp_t(-.1, prec), true);
            ta_copy.update_d_output(fp_t(-.1, prec), true);

            REQUIRE(ta.get_d_output() == ta_copy.get_d_output());
        }
    }
}
