// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("taylor te")
{
    using Catch::Matchers::Message;

    auto [v] = make_vars("v");

    using ev_t = taylor_adaptive<double>::t_event_t;

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
        kw::callback = [](taylor_adaptive<double> &, double, bool) {});
    REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
    REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
    oss.str("");

    oss << ev_t(
        v * v - 1e-10, kw::direction = event_direction::negative,
        kw::callback = [](taylor_adaptive<double> &, double, bool) {}, kw::cooldown = -5);
    REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    REQUIRE(boost::algorithm::contains(oss.str(), " auto"));
    REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
    oss.str("");

    oss << ev_t(
        v * v - 1e-10, kw::direction = event_direction::negative,
        kw::callback = [](taylor_adaptive<double> &, double, bool) {}, kw::cooldown = 1);
    REQUIRE(boost::algorithm::contains(oss.str(), " event_direction::negative"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    REQUIRE(boost::algorithm::contains(oss.str(), " 1"));
    REQUIRE(boost::algorithm::contains(oss.str(), " yes"));
    oss.str("");

    // Failure modes.
    REQUIRE_THROWS_MATCHES(ev_t(
                               v * v - 1e-10, kw::direction = event_direction::negative,
                               kw::callback = [](taylor_adaptive<double> &, double, bool) {},
                               kw::cooldown = std::numeric_limits<double>::quiet_NaN()),
                           std::invalid_argument,
                           Message("Cannot set a non-finite cooldown value for a terminal event"));
    REQUIRE_THROWS_MATCHES(ev_t(
                               v * v - 1e-10, kw::direction = event_direction{50},
                               kw::callback = [](taylor_adaptive<double> &, double, bool) {}),
                           std::invalid_argument,
                           Message("Invalid value selected for the direction of a terminal event"));
}

TEST_CASE("taylor te basic")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;
        using nt_ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

        auto counter_nt = 0u, counter_t = 0u;
        fp_t cur_time(0);
        bool direction = true;

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {fp_t(0.), fp_t(0.25)},
            kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,
            kw::compact_mode = compact_mode,
            kw::nt_events = {nt_ev_t(v * v - 1e-10,
                                     [&counter_nt, &cur_time, &direction](taylor_adaptive<fp_t> &ta, fp_t t) {
                                         // Make sure the callbacks are called in order.
                                         if (direction) {
                                             REQUIRE(t > cur_time);
                                         } else {
                                             REQUIRE(t < cur_time);
                                         }

                                         ta.update_d_output(t);

                                         const auto v = ta.get_d_output()[1];
                                         REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                         ++counter_nt;

                                         cur_time = t;
                                     })},
            kw::t_events = {t_ev_t(
                v, kw::callback = [&counter_t, &cur_time, &direction](taylor_adaptive<fp_t> &ta, fp_t t, bool mr) {
                    REQUIRE(!mr);

                    if (direction) {
                        REQUIRE(t > cur_time);
                    } else {
                        REQUIRE(t < cur_time);
                    }

                    ta.update_d_output(t);

                    const auto v = ta.get_d_output()[1];
                    REQUIRE(abs(v) < std::numeric_limits<fp_t>::epsilon() * 100);

                    ++counter_t;

                    cur_time = t;
                })}};

        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_time() < 1);
        REQUIRE(counter_nt == 1u);
        REQUIRE(counter_t == 1u);

        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_time() > 1);
        REQUIRE(counter_nt == 3u);
        REQUIRE(counter_t == 2u);

        // Move backwards.
        direction = false;

        while (true) {
            oc = std::get<0>(ta.step_backward());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_time() < 1);
        REQUIRE(counter_nt == 5u);
        REQUIRE(counter_t == 3u);

        while (true) {
            oc = std::get<0>(ta.step_backward());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_time() < 0);
        REQUIRE(counter_nt == 7u);
        REQUIRE(counter_t == 4u);
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
        auto first_ev = static_cast<std::uint32_t>(oc);
        auto time = ta.get_time();
        REQUIRE((first_ev == 0u || first_ev == 1u));

        // Taking a further step, we might either detect the second event,
        // or it may end up being ignored due to numerics.
        oc = std::get<0>(ta.step());
        if (oc > taylor_outcome::success) {
            auto second_ev = static_cast<std::uint32_t>(oc);
            REQUIRE(time == approximately(ta.get_time()));
            REQUIRE((second_ev == 0u || second_ev == 1u));
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
            kw::callback = [](taylor_adaptive<fp_t> &, fp_t, bool mr) { REQUIRE(!mr); });

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
        REQUIRE(static_cast<std::uint32_t>(oc) == 1u);

        // Next step the first event must trigger.
        oc = std::get<0>(ta.step());
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);

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

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);

        oc = std::get<0>(ta.step_backward());
        REQUIRE(static_cast<std::uint32_t>(oc) == 1u);

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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_time() != 0);

        // Step until re-trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);

        auto tm = ta.get_time();
        oc = std::get<0>(ta.step());

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
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
            v, kw::callback = [](taylor_adaptive<fp_t> &, fp_t, bool mr) { REQUIRE(!mr); },
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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_state()[0] == approximately(fp_t(-1)));

        // Step until trigger.
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
        REQUIRE(ta.get_state()[0] == approximately(fp_t(-1)));

        // Other direction.
        auto ev1 = t_ev_t(
            v, kw::callback = [](taylor_adaptive<fp_t> &, fp_t, bool mr) { REQUIRE(!mr); },
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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);

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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
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
            kw::callback = [](taylor_adaptive<fp_t> &, fp_t, bool mr) { REQUIRE(mr); }, kw::cooldown = fp_t(1e-1));

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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
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

TEST_CASE("taylor te propagate")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::abs;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;

        auto counter = 0u;

        t_ev_t ev(
            v, kw::callback = [&counter](taylor_adaptive<fp_t> &, fp_t, bool mr) {
                ++counter;
                REQUIRE(!mr);
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
        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);
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
