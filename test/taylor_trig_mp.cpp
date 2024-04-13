// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>

#include <mp++/real.hpp>

#include <heyoka/kw.hpp>
#include <heyoka/math/acos.hpp>
#include <heyoka/math/asin.hpp>
#include <heyoka/math/atan.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/tan.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("cos")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = cos(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(cos(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = cos(y + 2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(cos(jet[1] + fp_t(2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4] == approximately(fp_t(.5, prec) * -sin(jet[1] + fp_t(2, prec)) * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("sin")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(sin(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sin(y + 2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(sin(jet[1] + fp_t(2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4] == approximately(fp_t(.5, prec) * cos(jet[1] + fp_t(2, prec)) * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("tan")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = tan(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(tan(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = tan(y + 2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(tan(jet[1] + fp_t(2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4]
                                == approximately(fp_t(.5, prec)
                                                 / (cos(jet[1] + fp_t(2, prec)) * cos(jet[1] + fp_t(2, prec)))
                                                 * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("acos")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = acos(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{.3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(acos(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = acos(y + .2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{.3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == fp_t(.3, prec));
                        REQUIRE(jet[2] == approximately(acos(jet[1] + fp_t(.2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4]
                                == approximately(
                                    -fp_t(.5, prec)
                                    / sqrt(fp_t(1, prec) - (jet[1] + fp_t(.2, prec)) * (jet[1] + fp_t(.2, prec)))
                                    * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("asin")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = asin(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{.3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(asin(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = asin(y + .2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{.3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == fp_t(.3, prec));
                        REQUIRE(jet[2] == approximately(asin(jet[1] + fp_t(.2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4]
                                == approximately(
                                    fp_t(.5, prec)
                                    / sqrt(fp_t(1, prec) - (jet[1] + fp_t(.2, prec)) * (jet[1] + fp_t(.2, prec)))
                                    * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("atan")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = atan(par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{.3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(atan(ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                    // Test with variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = atan(y + .2_dbl), prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{.3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == fp_t(.3, prec));
                        REQUIRE(jet[2] == approximately(atan(jet[1] + fp_t(.2, prec))));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(
                            jet[4]
                            == approximately(fp_t(.5, prec)
                                             / (fp_t(1, prec) + (jet[1] + fp_t(.2, prec)) * (jet[1] + fp_t(.2, prec)))
                                             * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }
                }
            }
        }
    }
}

TEST_CASE("atan2")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(2_dbl, par[0]), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{3, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(atan2(fp_t(2, prec), ta.get_pars()[0])));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }

                    // Test with variable/number.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(y, 2_dbl), prime(y) = atan2(par[0], x)},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(atan2(jet[1], fp_t(2, prec))));
                        REQUIRE(jet[3] == approximately(atan2(ta.get_pars()[0], jet[0])));
                        REQUIRE(jet[4]
                                == approximately(fp_t(.5, prec) * (fp_t(2, prec) * jet[3])
                                                 / (fp_t(4, prec) + jet[1] * jet[1])));
                        REQUIRE(jet[5]
                                == approximately(fp_t(.5, prec) * -ta.get_pars()[0]
                                                 / (fp_t(4, prec) + ta.get_pars()[0] * ta.get_pars()[0]) * jet[2]));
                    }

                    // Test with variable/variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = atan2(x, y), prime(y) = atan2(y, x)},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == atan2(jet[0], jet[1]));
                        REQUIRE(jet[3] == atan2(jet[1], jet[0]));
                        REQUIRE(jet[4]
                                == approximately(fp_t{1, prec} / fp_t{2, prec} * (-jet[0] * jet[3] + jet[1] * jet[2])
                                                 / (jet[0] * jet[0] + jet[1] * jet[1])));
                        REQUIRE(jet[5]
                                == approximately(fp_t{1, prec} / fp_t{2, prec} * (-jet[1] * jet[2] + jet[0] * jet[3])
                                                 / (jet[0] * jet[0] + jet[1] * jet[1])));
                    }
                }
            }
        }
    }
}
