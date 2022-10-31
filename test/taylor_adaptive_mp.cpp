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

#include <mp++/real.hpp>

#include <heyoka/expression.hpp>
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
            for (auto prec : {30u, 123u}) {
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
