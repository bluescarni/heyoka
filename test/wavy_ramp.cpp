// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: a series of tests for bugs detected when implementing the wavy ramp example.

#include <cmath>
#include <initializer_list>
#include <tuple>

#include <boost/math/constants/constants.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka_test;
using namespace heyoka;

const auto pi_const = boost::math::constants::pi<double>();

auto make_wavy_ramp(bool check_event)
{
    auto cb_curve = [check_event](taylor_adaptive<double> &ta, bool, event_direction) {
        auto x = ta.get_state()[0];
        auto y = ta.get_state()[1];

        if (check_event) {
            REQUIRE(y - (1. - x + 0.05 * std::cos(11 * pi_const * x)) == approximately(0.));
        }

        auto grad_x = 1 + 0.05 * 11 * pi_const * std::sin(11 * pi_const * x);
        auto grad_y = 1.;
        auto grad_norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
        grad_x /= grad_norm;
        grad_y /= grad_norm;

        auto vx = ta.get_state()[2];
        auto vy = ta.get_state()[3];

        auto vproj = vx * grad_x + vy * grad_y;

        ta.get_state_data()[2] -= 1.8 * vproj * grad_x;
        ta.get_state_data()[3] -= 1.8 * vproj * grad_y;

        return true;
    };

    auto cb_bottom = [check_event](taylor_adaptive<double> &ta, bool, event_direction) {
        if (check_event) {
            auto y = ta.get_state()[1];

            REQUIRE(y == approximately(0.));
        }

        ta.get_state_data()[3] = -0.8 * ta.get_state_data()[3];

        return true;
    };

    auto [x, y, vx, vy] = make_vars("x", "y", "vx", "vy");

    auto eq_w_curve = y - (1. - x + 0.05 * cos(11 * pi_const * x));
    auto eq_bottom = y;

    auto ta = taylor_adaptive<double>(
        {prime(x) = vx, prime(y) = vy, prime(vx) = 0_dbl, prime(vy) = -1_dbl}, {0, 1.2, 0, 0},
        kw::t_events
        = {t_event<double>(eq_w_curve, kw::callback = cb_curve, kw::direction = event_direction::negative),
           t_event<double>(eq_bottom, kw::callback = cb_bottom, kw::direction = event_direction::negative)});

    return ta;
}

// This test case would trigger an assertion misfire
// in the event detection code.
TEST_CASE("assertion misfire")
{
    auto ta = make_wavy_ramp(false);

    REQUIRE(std::get<0>(ta.step()) == taylor_outcome::err_nf_state);
    REQUIRE(std::get<0>(ta.step(10.)) == taylor_outcome::err_nf_state);
}

// Make sure that the nonlinear event equation is correctly propagated
// at the desired precision.
TEST_CASE("accurate event propagation")
{
    auto ta = make_wavy_ramp(true);

    REQUIRE(std::get<0>(ta.propagate_until(10.)) == taylor_outcome::time_limit);
}
