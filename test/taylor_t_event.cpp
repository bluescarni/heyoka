// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Test for an event triggering exactly at the end of a timestep.
TEST_CASE("linear box")
{
    using ev_t = taylor_adaptive<double>::t_event_t;

    auto x = make_vars("x");

    auto counter = 0u;

    auto ta_ev = taylor_adaptive<double>{{prime(x) = 1_dbl},
                                         {0.},
                                         kw::t_events = {ev_t(
                                             x - 1., kw::callback = [&counter](taylor_adaptive<double> &, int) {
                                                 ++counter;
                                                 return true;
                                             })}};

    // Check that the event triggers at the beginning of the second step.
    auto [oc, h] = ta_ev.step(1.);

    REQUIRE(oc == taylor_outcome::time_limit);
    REQUIRE(h == 1.);
    REQUIRE(counter == 0u);
    REQUIRE(ta_ev.get_state()[0] == 1.);

    std::tie(oc, h) = ta_ev.step(1.);
    REQUIRE(oc == taylor_outcome{0});
    REQUIRE(h == approximately(0.));
    REQUIRE(counter == 1u);
    REQUIRE(ta_ev.get_state()[0] == 1.);
}

TEST_CASE("copy semantics")
{
    using ev_t = taylor_adaptive<double>::t_event_t;

    auto v = make_vars("v");

    auto ex = v + 3_dbl;

    // Expression is shallow copied on construction.
    ev_t ev(ex);
    REQUIRE(std::get<func>(ex.value()).get_ptr() == std::get<func>(ev.get_expression().value()).get_ptr());

    // Copy ctor.
    auto ev2 = ev;

    REQUIRE(std::get<func>(ev.get_expression().value()).get_ptr()
            == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Self assignment.
    auto orig_id = std::get<func>(ev2.get_expression().value()).get_ptr();
    ev2 = *&ev2;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());

    // Copy assignment.
    ev2 = ev;
    REQUIRE(orig_id == std::get<func>(ev2.get_expression().value()).get_ptr());
}

TEST_CASE("taylor te")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto btup = std::make_tuple(std::true_type{}, std::false_type{});

        auto inner = [](auto bflag) {
            using Catch::Matchers::Message;

            auto v = make_vars("v");

            using ev_t = detail::t_event_impl<fp_t, decltype(bflag)::value>;

            std::ostringstream oss;
            oss << ev_t(v * v - 1e-10);
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::any"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " no"));
            oss.str("");

            oss << ev_t(v * v - 1e-10, kw::direction = event_direction::positive);
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::positive"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " no"));
            oss.str("");

            oss << ev_t(v * v - 1e-10, kw::direction = event_direction::negative);
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " no"));
            oss.str("");

            oss << ev_t(v * v - 1e-10, kw::direction = event_direction::negative);
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " no"));
            oss.str("");

            oss << ev_t(
                v * v - 1e-10, kw::direction = event_direction::negative,
                kw::callback = [](auto &, int, auto...) { return true; });
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
            oss.str("");

            oss << ev_t(
                v * v - 1e-10, kw::direction = event_direction::negative,
                kw::callback = [](auto &, int, auto...) { return true; }, kw::cooldown = fp_t(-5));
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
            REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
            oss.str("");

            oss << ev_t(
                v * v - 1e-10, kw::direction = event_direction::negative,
                kw::callback = [](auto &, int, auto...) { return true; }, kw::cooldown = fp_t(1));
            REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            REQUIRE(boost::algorithm::contains(oss.str(), " 1"));
            REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
            oss.str("");

            // Check the assignment operators.
            ev_t ev0(
                v * v - 1e-10, kw::callback = [](auto &, int, auto...) { return true; }),
                ev1(
                    v * v - 1e-10, kw::callback = [](auto &, int, auto...) { return true; },
                    kw::direction = event_direction::negative),
                ev2(
                    v * v - 1e-10, kw::callback = [](auto &, int, auto...) { return true; },
                    kw::direction = event_direction::positive);
            ev0 = ev1;
            oss << ev0;
            REQUIRE(boost::algorithm::contains(oss.str(), "event_direction::negative"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            oss.str("");

            ev0 = std::move(ev2);
            oss << ev0;
            REQUIRE(boost::algorithm::contains(oss.str(), "event_direction::positive"));
            REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
            oss.str("");

            // Failure modes.
            REQUIRE_THROWS_MATCHES(ev_t(
                                       v * v - 1e-10, kw::direction = event_direction::negative,
                                       kw::callback = [](auto &, int, auto...) { return true; },
                                       kw::cooldown = std::numeric_limits<fp_t>::quiet_NaN()),
                                   std::invalid_argument,
                                   Message("Cannot set a non-finite cooldown value for a terminal event"));
            REQUIRE_THROWS_MATCHES(ev_t(
                                       v * v - 1e-10, kw::direction = event_direction{50},
                                       kw::callback = [](auto &, int, auto...) { return true; }),
                                   std::invalid_argument,
                                   Message("Invalid value selected for the direction of a terminal event"));
        };

        tuple_for_each(btup, inner);
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("taylor te basic")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;
        using nt_ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

        // NOTE: don't make the small delta too smal in single-precision.
        const auto small_delta = std::is_same_v<fp_t, float> ? 1e-6 : 1e-10;

        // NOTE: test also sub-eps tolerance.
        for (auto cur_tol : {std::numeric_limits<fp_t>::epsilon(), std::numeric_limits<fp_t>::epsilon() / 100}) {
            auto counter_nt = 0u, counter_t = 0u;
            fp_t cur_time(0);
            bool direction = true;

            auto ta = taylor_adaptive<fp_t>{
                {prime(x) = v, prime(v) = -9.8 * sin(x)},
                {fp_t(0.), fp_t(0.25)},
                kw::tol = cur_tol,
                kw::opt_level = opt_level,
                kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode,
                kw::nt_events
                = {nt_ev_t(v * v - small_delta,
                           [&counter_nt, &cur_time, &direction, small_delta](taylor_adaptive<fp_t> &ta_, fp_t t, int) {
                               // Make sure the callbacks are called in order.
                               if (direction) {
                                   REQUIRE(t > cur_time);
                               } else {
                                   REQUIRE(t < cur_time);
                               }

                               ta_.update_d_output(t);

                               const auto vel = ta_.get_d_output()[1];
                               REQUIRE(abs(vel * vel - small_delta) < std::numeric_limits<fp_t>::epsilon());

                               ++counter_nt;

                               cur_time = t;
                           })},
                kw::t_events = {t_ev_t(
                    v, kw::callback = [&counter_t, &cur_time, &direction](taylor_adaptive<fp_t> &ta_, int) {
                        const auto t = ta_.get_time();

                        if (direction) {
                            REQUIRE(t > cur_time);
                        } else {
                            REQUIRE(t < cur_time);
                        }

                        const auto vel = ta_.get_state()[1];
                        REQUIRE(abs(vel) < std::numeric_limits<fp_t>::epsilon() * 100);

                        ++counter_t;

                        cur_time = t;

                        return true;
                    })}};

            taylor_outcome oc;
            while (true) {
                oc = std::get<0>(ta.step());
                REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            REQUIRE(static_cast<std::int64_t>(oc) == 0);
            REQUIRE(ta.get_time() < 1);
            REQUIRE(counter_nt == 1u);
            REQUIRE(counter_t == 1u);

            while (true) {
                oc = std::get<0>(ta.step());
                REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            REQUIRE(static_cast<std::int64_t>(oc) == 0);
            REQUIRE(ta.get_time() > 1);
            REQUIRE(counter_nt == 3u);
            REQUIRE(counter_t == 2u);

            // Move backwards.
            direction = false;

            while (true) {
                oc = std::get<0>(ta.step_backward());
                REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            REQUIRE(static_cast<std::int64_t>(oc) == 0);
            REQUIRE(ta.get_time() < 1);
            REQUIRE(counter_nt == 5u);
            REQUIRE(counter_t == 3u);

            while (true) {
                oc = std::get<0>(ta.step_backward());
                REQUIRE((oc == taylor_outcome::success || static_cast<std::int64_t>(oc) >= 0));
                if (oc > taylor_outcome::success) {
                    break;
                }
                REQUIRE(oc == taylor_outcome::success);
            }

            REQUIRE(static_cast<std::int64_t>(oc) == 0);
            REQUIRE(ta.get_time() < 0);
            REQUIRE(counter_nt == 7u);
            REQUIRE(counter_t == 4u);
        }
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te identical")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        t_ev_t ev(v);

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0.), fp_t(0.25)},          kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev, ev}};

        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        // One of the two events must always be detected.
        auto first_ev = -static_cast<std::int64_t>(oc) - 1;
        auto time = ta.get_time();
        REQUIRE((first_ev == 0 || first_ev == 1));

        // Taking a further step, we might either detect the second event,
        // or it may end up being ignored due to numerics.
        oc = std::get<0>(ta.step());
        if (oc > taylor_outcome::success) {
            auto second_ev = -static_cast<std::int64_t>(oc) - 1;
            REQUIRE(time == approximately(ta.get_time()));
            REQUIRE((second_ev == 0 || second_ev == 1));
            REQUIRE(second_ev != first_ev);

            // Both events should be in cooldown: taking a step
            // backwards should not run into another event.
            oc = std::get<0>(ta.step_backward());
            REQUIRE(oc == taylor_outcome::success);
        } else {
            REQUIRE(oc == taylor_outcome::success);
        }
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te close")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        t_ev_t ev1(x);
        t_ev_t ev2(
            x - std::numeric_limits<fp_t>::epsilon() * 2,
            kw::callback = [](taylor_adaptive<fp_t> &, int) { return true; });

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0.1), fp_t(0.25)},         kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev1, ev2}};

        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        // The second event must have triggered first.
        REQUIRE(static_cast<std::int64_t>(oc) == 1);

        // Next step the first event must trigger.
        oc = std::get<0>(ta.step());
        REQUIRE(static_cast<std::int64_t>(oc) == -1);

        // Next step no event must trigger: event 0 is now on cooldown
        // as it just happened, and event 1 is still close enough to be
        // on cooldown too.
        oc = std::get<0>(ta.step());
        REQUIRE(oc == taylor_outcome::success);

        // Go back.
        while (true) {
            oc = std::get<0>(ta.step_backward());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::int64_t>(oc) == -1);

        oc = std::get<0>(ta.step_backward());
        REQUIRE(static_cast<std::int64_t>(oc) == 1);

        // Taking the step forward will skip event zero as it is still
        // on cooldown.
        oc = std::get<0>(ta.step());
        REQUIRE(oc == taylor_outcome::success);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te retrigger")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        t_ev_t ev(x - (1 - std::numeric_limits<fp_t>::epsilon() * 6));

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                        {fp_t(1), fp_t(0)},
                                        kw::opt_level = opt_level,
                                        kw::high_accuracy = high_accuracy,
                                        kw::compact_mode = compact_mode,
                                        kw::t_events = {ev}};

        // First timestep triggers the event immediately.
        taylor_outcome oc;
        oc = std::get<0>(ta.step());
        REQUIRE(static_cast<std::int64_t>(oc) == -1);
        REQUIRE(ta.get_time() != 0);

        // Step until re-trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::int64_t>(oc) == -1);

        auto tm = ta.get_time();
        oc = std::get<0>(ta.step());

        REQUIRE(static_cast<std::int64_t>(oc) == -1);
        REQUIRE(ta.get_time() > tm);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te dir")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        t_ev_t ev(
            v,
            kw::callback =
                [](taylor_adaptive<fp_t> &, int d_sgn) {
                    REQUIRE(d_sgn == 1);
                    return true;
                },
            kw::direction = event_direction::positive);

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                        {fp_t(1), fp_t(0)},
                                        kw::opt_level = opt_level,
                                        kw::high_accuracy = high_accuracy,
                                        kw::compact_mode = compact_mode,
                                        kw::t_events = {ev}};

        // First timestep must not trigger the event (derivative is negative).
        taylor_outcome oc = std::get<0>(ta.step());
        REQUIRE(oc == taylor_outcome::success);

        // Step until trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::int64_t>(oc) == 0);
        REQUIRE(ta.get_state()[0] == approximately(fp_t(-1)));

        // Step until trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::int64_t>(oc) == 0);
        REQUIRE(ta.get_state()[0] == approximately(fp_t(-1)));

        // Other direction.
        auto ev1 = t_ev_t(
            v,
            kw::callback =
                [](taylor_adaptive<fp_t> &, int d_sgn) {
                    REQUIRE(d_sgn == -1);
                    return true;
                },
            kw::direction = event_direction::negative);

        ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                   {fp_t(1), fp_t(0)},
                                   kw::opt_level = opt_level,
                                   kw::high_accuracy = high_accuracy,
                                   kw::compact_mode = compact_mode,
                                   kw::t_events = {ev1}};

        // Check that zero timestep does not detect anything.
        oc = std::get<0>(ta.step(fp_t(0)));
        REQUIRE(oc == taylor_outcome::time_limit);

        // Now it must trigger immediately.
        oc = std::get<0>(ta.step());
        REQUIRE(static_cast<std::int64_t>(oc) == 0);

        // The next timestep must not trigger due to cooldown.
        oc = std::get<0>(ta.step());
        REQUIRE(oc == taylor_outcome::success);

        // Step until trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::int64_t>(oc) == 0);
        REQUIRE(ta.get_state()[0] == approximately(fp_t(1)));
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te custom cooldown")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        t_ev_t ev(
            v * v - std::numeric_limits<fp_t>::epsilon() * 4,
            kw::callback = [](taylor_adaptive<fp_t> &, int) { return true; }, kw::cooldown = fp_t(1e-1));

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0), fp_t(0.25)},           kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev}};

        // Step until trigger.
        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::int64_t>(oc) == 0);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te propagate_for")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        auto counter = 0u;

        t_ev_t ev(
            v, kw::callback = [&counter](taylor_adaptive<fp_t> &, int) {
                ++counter;
                return true;
            });

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0), fp_t(0.25)},           kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev}};

        auto oc = std::get<0>(ta.propagate_for(fp_t(100)));
        REQUIRE(oc == taylor_outcome::time_limit);
        REQUIRE(ta.get_time() == 100);

        REQUIRE(counter == 100u);

        t_ev_t ev1(v);

        ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0), fp_t(0.25)},           kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev1}};

        oc = std::get<0>(ta.propagate_for(fp_t(100)));
        REQUIRE(oc > taylor_outcome::success);
        REQUIRE(static_cast<std::int64_t>(oc) == -1);
        REQUIRE(ta.get_time() < 0.502);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("taylor te propagate_grid")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        auto counter = 0u;

        t_ev_t ev(
            v, kw::callback = [&counter](taylor_adaptive<fp_t> &, int) {
                ++counter;
                return true;
            });

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0), fp_t(0.25)},           kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev}};

        std::vector<fp_t> grid;
        for (auto i = 0; i < 101; ++i) {
            grid.emplace_back(i);
        }

        taylor_outcome oc;
        {
            auto [oc_, _1, _2, _3, _4, out] = ta.propagate_grid(grid);
            oc = oc_;
            REQUIRE(out.size() == 202u);
        }
        REQUIRE(oc == taylor_outcome::time_limit);
        REQUIRE(ta.get_time() >= 100);

        REQUIRE(counter == 100u);

        t_ev_t ev1(v);

        ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)}, {fp_t(0), fp_t(0.25)},           kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,        kw::compact_mode = compact_mode, kw::t_events = {ev1}};

        {
            auto [oc_, _1, _2, _3, _4, out] = ta.propagate_grid(grid);
            oc = oc_;
            REQUIRE(out.size() == 2u);
        }
        REQUIRE(oc > taylor_outcome::time_limit);
        REQUIRE(static_cast<std::int64_t>(oc) == -1);
        REQUIRE(ta.get_time() < 0.502);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

// Test for a bug in propagate_grid() in which the
// integration would stop at the first step, in case
// a terminal event triggers immediately.
TEST_CASE("taylor te propagate_grid first step bug")
{
    auto [x, v] = make_vars("x", "v");

    using t_ev_t = taylor_adaptive<double>::t_event_t;

    {
        t_ev_t ev(v, kw::callback = [](taylor_adaptive<double> &, int) { return true; });

        auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}, kw::t_events = {ev}};

        std::vector<double> grid;
        for (auto i = 0; i < 100; ++i) {
            grid.emplace_back(5 / 100. * i);
        }

        auto out = ta.propagate_grid(grid);

        REQUIRE(!std::get<4>(out));
        REQUIRE(std::get<5>(out).size() == 200u);
    }

    {
        t_ev_t ev(v);

        auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}, kw::t_events = {ev}};

        std::vector<double> grid;
        for (auto i = 0; i < 100; ++i) {
            grid.emplace_back(5 / 100. * i);
        }

        auto out = ta.propagate_grid(grid);

        REQUIRE(!std::get<4>(out));
        REQUIRE(std::get<5>(out).size() == 4u);
    }
}

TEST_CASE("taylor te damped pendulum")
{
    using t_ev_t = taylor_adaptive<double>::t_event_t;

    auto [x, v] = make_vars("x", "v");

    std::vector<double> zero_vel_times;

    auto callback = [&zero_vel_times](taylor_adaptive<double> &ta, int) {
        const auto tm = ta.get_time();

        if (ta.get_pars()[0] == 0) {
            ta.get_pars_data()[0] = 1;
        } else {
            ta.get_pars_data()[0] = 0;
        }

        zero_vel_times.push_back(tm);

        return true;
    };

    t_ev_t ev(v, kw::callback = callback);

    auto ta = taylor_adaptive<double>{
        {prime(x) = v, prime(v) = -9.8 * sin(x) - par[0] * v}, {0.05, 0.025}, kw::t_events = {ev}};

    ta.propagate_until(100);

    REQUIRE(zero_vel_times.size() == 99u);

    ta.step();

    REQUIRE(zero_vel_times.size() == 100u);

    // Mix use of step() and propagate like in the tutorial.
    zero_vel_times.clear();
    ta.set_time(0);
    ta.get_state_data()[0] = 0.05;
    ta.get_state_data()[1] = 0.025;

    taylor_outcome oc;
    do {
        oc = std::get<0>(ta.step());
    } while (oc == taylor_outcome::success);

    ta.propagate_until(100);

    REQUIRE(zero_vel_times.size() == 99u);

    ta.step();

    REQUIRE(zero_vel_times.size() == 100u);
}

TEST_CASE("taylor te boolean callback")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        auto counter_t = 0u;
        fp_t cur_time(0);
        bool direction = true;

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {fp_t(0.), fp_t(0.25)},
            kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,
            kw::compact_mode = compact_mode,
            kw::t_events = {t_ev_t(
                v, kw::callback = [&counter_t, &cur_time, &direction](taylor_adaptive<fp_t> &ta_, int) {
                    const auto t = ta_.get_time();

                    if (direction) {
                        REQUIRE(t > cur_time);
                    } else {
                        REQUIRE(t < cur_time);
                    }

                    const auto vel = ta_.get_state()[1];
                    REQUIRE(abs(vel) < std::numeric_limits<fp_t>::epsilon() * 100);

                    ++counter_t;

                    cur_time = t;

                    return counter_t != 5u;
                })}};

        // First we integrate up to the first
        // occurrence of the terminal event.
        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::int64_t>(oc) == 0);

        // Then we propagate for an amount of time large enough
        // to trigger the stopping terminal event.
        oc = std::get<0>(ta.propagate_until(fp_t(1000)));
        REQUIRE(static_cast<std::int64_t>(oc) == -1);

        // Reset counter_t and invert direction.
        counter_t = 0;
        direction = false;

        while (true) {
            oc = std::get<0>(ta.step_backward());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        oc = std::get<0>(ta.propagate_for(fp_t(-1000)));
        REQUIRE(static_cast<std::int64_t>(oc) == -1);

        // Some testing for propagate_grid() too.
        ta.reset_cooldowns();
        ta.set_time(fp_t{0});
        ta.get_state_data()[0] = fp_t(-0.1);
        ta.get_state_data()[1] = 0;
        cur_time = -1;
        direction = true;
        counter_t = 0;

        auto out = ta.propagate_grid({fp_t{0}});
        REQUIRE(std::get<0>(out) == taylor_outcome::time_limit);
        REQUIRE(counter_t == 0u);

        ta.reset_cooldowns();
        ta.set_time(fp_t{0});
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = fp_t(0.25);
        cur_time = 0;
        direction = true;
        counter_t = 0;

        out = ta.propagate_grid({fp_t{0}, fp_t{0.5}, fp_t{1000}});
        REQUIRE(static_cast<std::int64_t>(std::get<0>(out)) == -1);
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

// Test a terminal event exactly at the end of a timestep.
TEST_CASE("step end")
{
    auto [x, v] = make_vars("x", "v");

    using t_ev_t = typename taylor_adaptive<double>::t_event_t;

    auto counter = 0u;

    auto ta
        = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                  {0., 0.25},
                                  kw::t_events = {t_ev_t(
                                      heyoka::time - 1., kw::callback = [&counter](taylor_adaptive<double> &ta_, int) {
                                          ++counter;
                                          REQUIRE(ta_.get_time() == 1.);
                                          return true;
                                      })}};

    ta.propagate_until(10., kw::max_delta_t = 0.005);

    REQUIRE(counter == 1u);
}

struct s11n_callback {
    template <typename T>
    bool operator()(taylor_adaptive<T> &, int) const
    {
        return true;
    }

private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(s11n_callback, bool, taylor_adaptive<float> &, int)
HEYOKA_S11N_CALLABLE_EXPORT(s11n_callback, bool, taylor_adaptive<double> &, int)
HEYOKA_S11N_CALLABLE_EXPORT(s11n_callback, bool, taylor_adaptive<long double> &, int)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_CALLABLE_EXPORT(s11n_callback, bool, taylor_adaptive<mppp::real128> &, int)

#endif

TEST_CASE("t s11n")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        {
            t_event<fp_t> ev(v, kw::callback = s11n_callback{}, kw::direction = event_direction::positive,
                             kw::cooldown = fp_t(100));

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << ev;
            }

            ev = t_event<fp_t>(v + x);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> ev;
            }

            REQUIRE(ev.get_expression() == v);
            REQUIRE(ev.get_direction() == event_direction::positive);
            REQUIRE(value_type_index(ev.get_callback()) == typeid(s11n_callback));
            REQUIRE(ev.get_cooldown() == fp_t(100));
        }

        // Try also a terminal event with empty callback.
        {
            t_event<fp_t> ev(v, kw::direction = event_direction::positive, kw::cooldown = fp_t(100));

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << ev;
            }

            ev = t_event<fp_t>(v + x);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> ev;
            }

            REQUIRE(ev.get_expression() == v);
            REQUIRE(ev.get_direction() == event_direction::positive);
            REQUIRE(!ev.get_callback());
            REQUIRE(ev.get_cooldown() == fp_t(100));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("te def ctor")
{
    t_event<double> te;

    REQUIRE(te.get_expression() == 0_dbl);
    REQUIRE(!te.get_callback());
    REQUIRE(te.get_direction() == event_direction::any);
    REQUIRE(te.get_cooldown() == -1.);
}

// Test to verify an event is not detected
// when it falls exactly at the end of a timestep.
TEST_CASE("te open range")
{
    auto tester = [](auto fp_x) {
        using std::nextafter;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                        {fp_t(-0.25), fp_t(0.)},
                                        kw::t_events = {ev_t(heyoka::time - 97 / fp_t(100000))}};

        auto [oc, h] = ta.step(97 / fp_t(100000));

        REQUIRE(oc == taylor_outcome::time_limit);

        // Reset the integrator.
        ta.set_time(0);
        ta.get_state_data()[0] = fp_t(-0.25);
        ta.get_state_data()[1] = fp_t(0);
        ta.reset_cooldowns();

        // Integrate up to immediately after the event.
        std::tie(oc, h) = ta.step(nextafter(97 / fp_t(100000), fp_t(1)));

        REQUIRE(oc == taylor_outcome{-1});

        // Run also a test at the very beginning.
        ta.set_time(97 / fp_t(100000));
        ta.get_state_data()[0] = fp_t(-0.25);
        ta.get_state_data()[1] = fp_t(0);
        ta.reset_cooldowns();

        std::tie(oc, h) = ta.step();

        REQUIRE(oc == taylor_outcome{-1});
        REQUIRE(h == 0);

        // And slightly later.
        ta.set_time(nextafter(97 / fp_t(100000), fp_t(1)));
        ta.get_state_data()[0] = fp_t(-0.25);
        ta.get_state_data()[1] = fp_t(0);
        ta.reset_cooldowns();

        std::tie(oc, h) = ta.step();

        REQUIRE(oc == taylor_outcome::success);
        REQUIRE(h > 0);
    };

    tuple_for_each(fp_types, tester);
}
