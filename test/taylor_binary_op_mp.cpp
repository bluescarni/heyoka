// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <utility>

#include <mp++/real.hpp>

#include <heyoka/detail/div.hpp>
#include <heyoka/detail/sub.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

auto add_wrapper(const expression &a, const expression &b)
{
    return expression{func{detail::sum_impl{{a, b}}}};
}

auto mul_prod_wrapper(const expression &a, const expression &b)
{
    return expression{func{detail::prod_impl{{a, b}}}};
}

auto div_wrapper(expression a, expression b)
{
    return detail::div(std::move(a), std::move(b));
}

auto sub_wrapper(expression a, expression b)
{
    return detail::sub(std::move(a), std::move(b));
}

TEST_CASE("taylor add")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = add_wrapper(2_dbl, par[0]), prime(y) = x + y},
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
                        REQUIRE(jet[2] == 5);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }

                    // Test with variable/number.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = y + 2_dbl, prime(y) = par[0] + x},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = .5,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level,
                                                        kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(jet[1] + fp_t(2, prec)));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] + jet[0]));
                        REQUIRE(jet[4] == approximately(fp_t(.5, prec) * jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(.5, prec) * jet[2]));
                    }

                    // Test with variable/variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = x + y, prime(y) = y + x},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = .5,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] + jet[1]);
                        REQUIRE(jet[3] == jet[0] + jet[1]);
                        REQUIRE(jet[4] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }
                }
            }
        }
    }
}

TEST_CASE("taylor sub")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(2_dbl, par[0]), prime(y) = x + y},
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
                        REQUIRE(jet[2] == -1);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }

                    // Test with variable/number.
                    {
                        auto ta = taylor_adaptive<fp_t>{
                            {prime(x) = sub_wrapper(y, 2_dbl), prime(y) = sub_wrapper(par[0], x)},
                            {fp_t{2, prec}, fp_t{3, prec}},
                            kw::tol = .5,
                            kw::high_accuracy = ha,
                            kw::compact_mode = cm,
                            kw::opt_level = opt_level,
                            kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(jet[1] - fp_t(2, prec)));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] - jet[0]));
                        REQUIRE(jet[4] == approximately(fp_t(.5, prec) * jet[3]));
                        REQUIRE(jet[5] == approximately(-fp_t(.5, prec) * jet[2]));
                    }

                    // Test with variable/variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = sub_wrapper(x, y), prime(y) = sub_wrapper(y, x)},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = .5,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] - jet[1]);
                        REQUIRE(jet[3] == -jet[0] + jet[1]);
                        REQUIRE(jet[4] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] - jet[3])));
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (-jet[2] + jet[3])));
                    }
                }
            }
        }
    }
}

TEST_CASE("taylor mul")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = mul_prod_wrapper(2_dbl, par[0]), prime(y) = x + y},
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
                        REQUIRE(jet[2] == 6);
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }

                    // Test with variable/number.
                    {
                        auto ta = taylor_adaptive<fp_t>{
                            {prime(x) = mul_prod_wrapper(y, 2_dbl), prime(y) = mul_prod_wrapper(par[0], x)},
                            {fp_t{2, prec}, fp_t{3, prec}},
                            kw::tol = .5,
                            kw::high_accuracy = ha,
                            kw::compact_mode = cm,
                            kw::opt_level = opt_level,
                            kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(jet[1] * fp_t(2, prec)));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] * jet[0]));
                        REQUIRE(jet[4] == approximately(jet[3]));
                        REQUIRE(jet[5] == approximately(fp_t(-2, prec) * jet[2]));
                    }

                    // Test with variable/variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{
                            {prime(x) = mul_prod_wrapper(x, y), prime(y) = mul_prod_wrapper(y, x)},
                            {fp_t{2, prec}, fp_t{3, prec}},
                            kw::tol = .5,
                            kw::high_accuracy = ha,
                            kw::compact_mode = cm,
                            kw::opt_level = opt_level};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] * jet[1]);
                        REQUIRE(jet[3] == jet[0] * jet[1]);
                        REQUIRE(jet[4]
                                == approximately(fp_t{1, prec} / fp_t{2, prec}
                                                 * (jet[2] * fp_t{3, prec} + jet[3] * fp_t{2, prec})));
                        REQUIRE(jet[5]
                                == approximately(fp_t{1, prec} / fp_t{2, prec}
                                                 * (jet[2] * fp_t{3, prec} + jet[3] * fp_t{2, prec})));
                    }
                }
            }
        }
    }
}

TEST_CASE("taylor div")
{
    using fp_t = mppp::real;

    auto x = "x"_var, y = "y"_var;

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto prec : {30, 123}) {
                    // Test with num/param.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = div_wrapper(2_dbl, par[0]), prime(y) = x + y},
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
                        REQUIRE(jet[2] == approximately(fp_t{2, prec} / ta.get_pars()[0]));
                        REQUIRE(jet[3] == approximately(fp_t{5, prec}));
                        REQUIRE(jet[4] == 0);
                        REQUIRE(jet[5] == approximately(fp_t{1, prec} / fp_t{2, prec} * (jet[2] + jet[3])));
                    }

                    // Test with variable/number.
                    {
                        auto ta = taylor_adaptive<fp_t>{
                            {prime(x) = div_wrapper(y, 2_dbl), prime(y) = div_wrapper(par[0], x)},
                            {fp_t{2, prec}, fp_t{3, prec}},
                            kw::tol = .5,
                            kw::high_accuracy = ha,
                            kw::compact_mode = cm,
                            kw::opt_level = opt_level,
                            kw::pars = {fp_t{-4, prec}}};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == approximately(jet[1] / fp_t(2, prec)));
                        REQUIRE(jet[3] == approximately(ta.get_pars()[0] / jet[0]));
                        REQUIRE(jet[4] == approximately(jet[3] / fp_t{4, prec}));
                        REQUIRE(jet[5]
                                == approximately(-ta.get_pars()[0] * jet[2] / (fp_t{2, prec} * jet[0] * jet[0])));
                    }

                    // Test with variable/variable.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = div_wrapper(x, y), prime(y) = div_wrapper(y, x)},
                                                        {fp_t{2, prec}, fp_t{3, prec}},
                                                        kw::tol = .5,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == 3);
                        REQUIRE(jet[2] == jet[0] / jet[1]);
                        REQUIRE(jet[3] == jet[1] / jet[0]);
                        REQUIRE(jet[4]
                                == approximately(fp_t{1, prec} / fp_t{2, prec}
                                                 * (jet[2] * fp_t{3, prec} - jet[3] * fp_t{2, prec})
                                                 / (fp_t{3, prec} * fp_t{3, prec})));
                        REQUIRE(jet[5]
                                == approximately(fp_t{1, prec} / fp_t{2, prec}
                                                 * (jet[3] * fp_t{2, prec} - jet[2] * fp_t{3, prec})
                                                 / (fp_t{2, prec} * fp_t{2, prec})));
                    }
                }
            }
        }
    }
}
