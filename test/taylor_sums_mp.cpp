// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <mp++/real.hpp>

#include <heyoka/kw.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Wrapper to ease the transition of old test code
// after the removal of sum_sq() from the public API.
auto sum_sq(const std::vector<expression> &args)
{
    std::vector<expression> new_args;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        new_args.push_back(arg * arg);
    }

    return sum(new_args);
}

TEST_CASE("sum")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sum({par[0], 2_dbl}), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t(3, prec)}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == 2 + ta.get_pars()[0]);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sum({x, y}), prime(y) = sum({par[0], x})},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t(-4, prec)}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] + jet[1]);
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4] == approximately(fp_t(.5, prec) * (jet[2] + jet[3])));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("sum_sq")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sum_sq({par[0], 2_dbl}), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t(3, prec)}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == 2 * 2 + ta.get_pars()[0] * ta.get_pars()[0]);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sum_sq({x, y}), prime(y) = sum_sq({par[0], x})},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t(-4, prec)}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] * jet[0] + jet[1] * jet[1]);
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] * ta.get_pars()[0] + jet[0] * jet[0]));
                        REQUIRE(jet[4] == approximately(jet[0] * jet[2] + jet[1] * jet[3]));
                        REQUIRE(jet[5] == approximately(jet[2] * jet[0]));
                    }
                }
            }
        }
    }
}
