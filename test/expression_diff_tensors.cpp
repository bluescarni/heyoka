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

    // Diff wrt all variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2, kw::diff_args = diff_args::vars);
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0] == std::vector{x + y, x * y * y});
    REQUIRE(dt.get_tensors()[1] == std::vector{1_dbl, 1_dbl, y * y, sum({(y * x), (x * y)})});
    REQUIRE(dt.get_tensors()[2] == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 2. * y, 2. * y, 2. * x});

    // Diff wrt some variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2, kw::diff_args = {x});
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0] == std::vector{x + y, x * y * y});
    REQUIRE(dt.get_tensors()[1] == std::vector{1_dbl, y * y});
    REQUIRE(dt.get_tensors()[2] == std::vector{0_dbl, 0_dbl});

    // Diff wrt all params.
    dt = diff_tensors({par[0] + y, x * y * par[1]}, kw::diff_order = 2, kw::diff_args = diff_args::params);
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0] == std::vector{par[0] + y, x * y * par[1]});
    REQUIRE(dt.get_tensors()[1] == std::vector{1_dbl, 0_dbl, 0_dbl, x * y});
    REQUIRE(dt.get_tensors()[2] == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl});

    // Diff wrt some param.
    dt = diff_tensors({par[0] + y, x * y * par[1]}, kw::diff_order = 2, kw::diff_args = {par[1]});
    REQUIRE(dt.get_tensors().size() == 3u);
    REQUIRE(dt.get_tensors()[0] == std::vector{par[0] + y, x * y * par[1]});
    REQUIRE(dt.get_tensors()[1] == std::vector{0_dbl, x * y});
    REQUIRE(dt.get_tensors()[2] == std::vector{0_dbl, 0_dbl});

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
    REQUIRE_THROWS_MATCHES(
        diff_tensors({par[0] + y, x * y * par[1]}, kw::diff_order = 2, kw::diff_args = diff_args{-1}),
        std::invalid_argument, Message("An invalid diff_args enumerator was passed to diff_tensors()"));
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

TEST_CASE("speelpenning check")
{
    std::uniform_real_distribution<double> rdist(-10., 10.);

    const auto nvars = 5u;

    std::vector<expression> vars;
    auto prod = 1_dbl;

    for (auto i = 0u; i < nvars; ++i) {
        auto cur_var = expression{fmt::format("x_{}", i)};

        vars.push_back(cur_var);
        prod *= cur_var;
    }

    const auto dt = diff_tensors({prod}, kw::diff_order = 3);

    REQUIRE(dt.get_tensors().size() == 4u);
    REQUIRE(dt.get_indices().size() == dt.get_tensors().size());

    for (auto diff_order = 0u; diff_order <= 3u; ++diff_order) {
        const auto &dtensors = dt.get_tensors()[diff_order];
        const auto &indices = dt.get_indices()[diff_order];

        REQUIRE(indices.size() == dtensors.size());

        llvm_state s;
        add_cfunc<double>(s, "diff", dtensors, kw::vars = vars);
        s.optimise();
        s.compile();

        auto *fr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("diff"));

        std::vector<double> inputs(nvars);
        for (auto &ival : inputs) {
            ival = rdist(rng);
        }
        std::vector<double> outputs(dt.get_tensors()[3].size());

        fr(outputs.data(), inputs.data(), nullptr, nullptr);

        for (decltype(indices.size()) i = 0; i < indices.size(); ++i) {
            const auto &v_idx = indices[i];
            REQUIRE(v_idx.size() == nvars + 1u);
            REQUIRE(v_idx[0] == 0u);

            auto ex = prod;

            for (auto j = 1u; j < v_idx.size(); ++j) {
                auto order = v_idx[j];

                for (decltype(order) k = 0; k < order; ++k) {
                    ex = diff(ex, vars[j - 1u]);
                }
            }

            llvm_state s2;
            add_cfunc<double>(s2, "diff", {ex}, kw::vars = vars);
            s2.optimise();
            s2.compile();

            auto *fr2 = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                s2.jit_lookup("diff"));

            double out = 0;
            fr2(&out, inputs.data(), nullptr, nullptr);

            REQUIRE(out == approximately(outputs[i]));
        }
    }
}

TEST_CASE("speelpenning complexity")
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

// This test checks that reverse-mode differentiation produces
// expressions in which the operands to commutative functions are kept
// in a canonical order.
TEST_CASE("comm canonical")
{
    auto [x, y] = make_vars("x", "y");

    auto dt = diff_tensors({par[0] * x * y}, kw::diff_args = diff_args::all);

    REQUIRE(dt[{0, 0, 0, 1}] == x * y);
    REQUIRE(dt[{0, 0, 1, 0}] == par[0] * x);
    REQUIRE(dt[{0, 1, 0, 0}] == par[0] * y);
}
