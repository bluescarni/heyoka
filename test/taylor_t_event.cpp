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

TEST_CASE("taylor te stream")
{
    auto [v] = make_vars("v");

    using ev_t = taylor_adaptive<double>::t_event_t;

    std::ostringstream oss;
    oss << ev_t(v * v - 1e-10);
    REQUIRE(boost::algorithm::contains(oss.str(), "direction::any"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    oss.str("");

    oss << ev_t(v * v - 1e-10, event_direction::positive);
    REQUIRE(boost::algorithm::contains(oss.str(), "event_direction::positive"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    oss.str("");

    oss << ev_t(v * v - 1e-10, event_direction::negative);
    REQUIRE(boost::algorithm::contains(oss.str(), "event_direction::negative"));
    REQUIRE(boost::algorithm::contains(oss.str(), " terminal"));
    oss.str("");
}

TEST_CASE("taylor te multizero")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using t_ev_t = typename taylor_adaptive<fp_t>::t_event_t;
        using nt_ev_t = typename taylor_adaptive<fp_t>::nt_event_t;

        auto counter = 0u;
        fp_t cur_time(0);

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {fp_t(0.), fp_t(0.25)},
            kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,
            kw::compact_mode = compact_mode,
            kw::nt_events = {nt_ev_t(v * v - 1e-10,
                                     [&counter, &cur_time](taylor_adaptive<fp_t> &ta, fp_t t) {
                                         using std::abs;

                                         // Make sure the callbacks are called in order.
                                         REQUIRE(t > cur_time);

                                         ta.update_d_output(t);

                                         const auto v = ta.get_d_output()[1];
                                         REQUIRE(abs(v * v - 1e-10) < std::numeric_limits<fp_t>::epsilon());

                                         ++counter;

                                         cur_time = t;
                                     })},
            kw::t_events = {t_ev_t(v)}};

        taylor_outcome oc;
        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
            REQUIRE(oc == taylor_outcome::success);
        }

        REQUIRE(static_cast<std::uint32_t>(oc) == 0u);

        while (true) {
            oc = std::get<0>(ta.step());
            if (oc > taylor_outcome::success) {
                break;
            }
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
