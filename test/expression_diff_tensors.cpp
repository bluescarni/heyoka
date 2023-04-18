// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <initializer_list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("revdiff decompose")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(detail::revdiff_decompose({x}).first == std::vector{x, "u_0"_var});
    REQUIRE(detail::revdiff_decompose({x}).second == 1u);

    REQUIRE(detail::revdiff_decompose({par[0]}).first == std::vector{par[0], "u_0"_var});
    REQUIRE(detail::revdiff_decompose({par[0]}).second == 1u);

    REQUIRE(detail::revdiff_decompose({par[0] + x}).first
            == std::vector{x, par[0], subs("u_1"_var + "u_0"_var, {{"u_1"_var, "u_0"_var}, {"u_0"_var, "u_1"_var}}),
                           "u_2"_var});
    REQUIRE(detail::revdiff_decompose({par[0] + x}).second == 2u);

    REQUIRE(detail::revdiff_decompose({(par[1] + y) * (par[0] + x)}).first
            == std::vector{x, y, par[0], par[1],
                           subs("u_2"_var + "u_0"_var, {{"u_2"_var, "u_0"_var}, {"u_0"_var, "u_2"_var}}),
                           subs("u_3"_var + "u_1"_var, {{"u_1"_var, "u_3"_var}, {"u_3"_var, "u_1"_var}}),
                           subs("u_5"_var * "u_4"_var, {{"u_5"_var, "u_4"_var}, {"u_4"_var, "u_5"_var}}), "u_6"_var});
    REQUIRE(detail::revdiff_decompose({(par[1] + y) * (par[0] + x)}).second == 4u);

    REQUIRE(detail::revdiff_decompose({subs((par[1] + y) * (par[0] + x), {{y, 1_dbl}})}).first
            == std::vector{x, par[0], par[1],
                           subs("u_1"_var + "u_0"_var, {{"u_1"_var, "u_0"_var}, {"u_0"_var, "u_1"_var}}),
                           subs("u_2"_var + y, {{y, 1_dbl}}),
                           subs("u_4"_var * "u_3"_var, {{"u_3"_var, "u_4"_var}, {"u_4"_var, "u_3"_var}}), "u_5"_var});
    REQUIRE(detail::revdiff_decompose({subs((par[1] + y) * (par[0] + x), {{y, 1_dbl}})}).second == 3u);
}

TEST_CASE("diff_tensors basic")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    // Let's begin with some trivial expressions.
    REQUIRE_THROWS_MATCHES(diff_tensors({1_dbl}, kw::diff_order = 0), std::invalid_argument,
                           Message("Cannot compute derivatives with respect to an empty set of arguments"));

    auto dt = diff_tensors({1_dbl}, kw::diff_order = 0, kw::diff_args = {x});
    REQUIRE(dt.get_tensors().size() == 1u);
    REQUIRE(dt.get_tensors()[0].size() == 1u);
    REQUIRE(dt.get_tensors()[0][0] == 1_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt.n_diffs() == 1u);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 1}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{0, 1})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 2}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{0, 2})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 1, kw::diff_args = {par[0]});
    REQUIRE(dt.get_tensors().size() == 2u);
    REQUIRE(dt.get_tensors()[0].size() == 1u);
    REQUIRE(dt.get_tensors()[0][0] == 1_dbl);
    REQUIRE(dt.get_tensors()[1].size() == 1u);
    REQUIRE(dt.get_tensors()[1][0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);
    REQUIRE(dt.n_diffs() == 2u);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 2}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{0, 2})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 2, kw::diff_args = {par[0]});
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0].size() == 1u);
    REQUIRE(dt.get_tensors()[0][0] == 1_dbl);
    REQUIRE(dt.get_tensors()[1].size() == 1u);
    REQUIRE(dt.get_tensors()[1][0] == 0_dbl);
    REQUIRE(dt.get_tensors()[2].size() == 1u);
    REQUIRE(dt.get_tensors()[2][0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);
    REQUIRE(dt[{0, 2}] == 0_dbl);
    REQUIRE(dt.n_diffs() == 3u);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 3}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{0, 3})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 3, kw::diff_args = {par[0]});
    REQUIRE(dt.get_tensors().size() == 4u);
    REQUIRE(dt.get_tensors()[0].size() == 1u);
    REQUIRE(dt.get_tensors()[0][0] == 1_dbl);
    REQUIRE(dt.get_tensors()[1].size() == 1u);
    REQUIRE(dt.get_tensors()[1][0] == 0_dbl);
    REQUIRE(dt.get_tensors()[2].size() == 1u);
    REQUIRE(dt.get_tensors()[2][0] == 0_dbl);
    REQUIRE(dt.get_tensors()[3].size() == 1u);
    REQUIRE(dt.get_tensors()[3][0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);
    REQUIRE(dt[{0, 2}] == 0_dbl);
    REQUIRE(dt[{0, 3}] == 0_dbl);
    REQUIRE(dt.n_diffs() == 4u);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 4}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the index vector {}", std::vector{0, 4})));

    // Automatically deduced diff variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2);
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0] == std::vector{x + y, x * y * y});
    REQUIRE(dt.get_tensors()[1] == std::vector{1_dbl, 1_dbl, y * y, sum({(y * x), (x * y)})});
    REQUIRE(dt.get_tensors()[2] == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 2. * y, 2. * y, 2. * x});

    // Error modes.
    REQUIRE_THROWS_MATCHES(diff_tensors({1_dbl}, kw::diff_order = 1, kw::diff_args = {x + y}), std::invalid_argument,
                           Message("Derivatives can be computed only with respect to variables and/or parameters"));
    REQUIRE_THROWS_MATCHES(
        diff_tensors({1_dbl}, kw::diff_order = 1, kw::diff_args = {x, x}), std::invalid_argument,
        Message("Duplicate entries detected in the list of variables/parameters with respect to which the "
                "derivatives are to be computed: [x, x]"));
    REQUIRE_THROWS_MATCHES(diff_tensors({1_dbl}, kw::diff_order = 1, kw::diff_args = diff_args{100}),
                           std::invalid_argument,
                           Message("An invalid diff_args enumerator was passed to diff_tensors()"));
    REQUIRE_THROWS_MATCHES(diff_tensors({}), std::invalid_argument,
                           Message("Cannot compute the derivatives of a function with zero components"));
}

// A few tests for the dtens API.
TEST_CASE("dtens basics")
{
    dtens dt;

    REQUIRE(dt.get_tensors().empty());
    REQUIRE(dt.n_diffs() == 0u);

    auto [x, y] = make_vars("x", "y");

    auto dt2 = diff_tensors({x + y, x * y}, kw::diff_order = 1);
    auto dt3(dt2);

    REQUIRE(dt3.get_tensors() == dt2.get_tensors());

    auto dt4(std::move(dt3));
    dt3 = dt4;

    REQUIRE(dt3.get_tensors() == dt2.get_tensors());

    // s11n.
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);

        oa << dt3;
    }

    dt3 = dtens();

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> dt3;
    }

    REQUIRE(dt3.get_tensors() == dt2.get_tensors());
    REQUIRE(dt3[{1, 0, 1}] == x);
}

TEST_CASE("speelpenning")
{
    fmt::print("Speelpenning's example\n");
    fmt::print("======================\n");

    std::uniform_real_distribution<double> rdist(-10., 10.);

    for (auto nvars : {3u, 10u, 20u, 50u, 100u, 200u}) {
        std::vector<double> inputs, outputs_f, outputs_r;

        std::vector<expression> vars;
        auto prod = 1_dbl;

        for (auto i = 0u; i < nvars; ++i) {
            auto cur_var = expression{fmt::format("x_{}", i)};

            vars.push_back(cur_var);
            prod *= cur_var;

            inputs.push_back(rdist(rng));
            outputs_f.push_back(0.);
            outputs_r.push_back(0.);
        }

        prod = pairwise_prod(vars);

        llvm_state s;
        // auto dc_forward = add_cfunc<double>(s, "f_forward", grad(prod, kw::diff_mode = diff_mode::forward));
        // auto dc_reverse = add_cfunc<double>(s, "f_reverse", grad(prod, kw::diff_mode = diff_mode::reverse));
        auto dc_reverse = add_cfunc<double>(s, "f_reverse", diff_tensors({prod}, kw::diff_order = 1).get_tensors()[1],
                                            kw::compact_mode = true);

        fmt::print("nvars={:<5} decomposition size={:<6}\n", nvars, dc_reverse.size() - nvars - nvars);

        s.optimise();
        s.compile();

        // auto *ff = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
        //     s.jit_lookup("f_forward"));

        // auto *fr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
        //     s.jit_lookup("f_reverse"));

        // ff(outputs_f.data(), inputs.data(), nullptr, nullptr);
        // fr(outputs_r.data(), inputs.data(), nullptr, nullptr);

        // for (auto i = 0u; i < nvars; ++i) {
        //     REQUIRE(outputs_f[i] == approximately(outputs_r[i]));
        // }
    }
}
