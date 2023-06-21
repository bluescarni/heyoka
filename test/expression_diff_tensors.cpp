// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("diff decompose")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(detail::diff_decompose({x}).first == std::vector{x, "u_0"_var});
    REQUIRE(detail::diff_decompose({x}).second == 1u);

    REQUIRE(detail::diff_decompose({par[0]}).first == std::vector{par[0], "u_0"_var});
    REQUIRE(detail::diff_decompose({par[0]}).second == 1u);

    REQUIRE(detail::diff_decompose({par[0] + x}).first
            == std::vector{x, par[0], subs("u_1"_var + "u_0"_var, {{"u_1"_var, "u_0"_var}, {"u_0"_var, "u_1"_var}}),
                           "u_2"_var});
    REQUIRE(detail::diff_decompose({par[0] + x}).second == 2u);

// TODO restore.
#if 0
    REQUIRE(detail::diff_decompose({(par[1] + y) * (par[0] + x)}).first
            == std::vector{x, y, par[0], par[1],
                           subs("u_2"_var + "u_0"_var, {{"u_2"_var, "u_0"_var}, {"u_0"_var, "u_2"_var}}),
                           subs("u_3"_var + "u_1"_var, {{"u_1"_var, "u_3"_var}, {"u_3"_var, "u_1"_var}}),
                           subs("u_5"_var * "u_4"_var, {{"u_5"_var, "u_4"_var}, {"u_4"_var, "u_5"_var}}), "u_6"_var});
    REQUIRE(detail::diff_decompose({(par[1] + y) * (par[0] + x)}).second == 4u);

    REQUIRE(detail::diff_decompose({subs((par[1] + y) * (par[0] + x), {{y, 1_dbl}})}).first
            == std::vector{x, par[0], par[1],
                           subs("u_1"_var + "u_0"_var, {{"u_1"_var, "u_0"_var}, {"u_0"_var, "u_1"_var}}),
                           subs("u_2"_var + y, {{y, 1_dbl}}),
                           subs("u_4"_var * "u_3"_var, {{"u_3"_var, "u_4"_var}, {"u_4"_var, "u_3"_var}}), "u_5"_var});
    REQUIRE(detail::diff_decompose({subs((par[1] + y) * (par[0] + x), {{y, 1_dbl}})}).second == 3u);
#endif
}

TEST_CASE("diff_tensors basic")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    // Let's begin with some trivial expressions.
    REQUIRE_THROWS_MATCHES(diff_tensors({1_dbl}, kw::diff_order = 0), std::invalid_argument,
                           Message("Cannot compute derivatives with respect to an empty set of arguments"));

    auto dt = diff_tensors({1_dbl}, kw::diff_order = 0, kw::diff_args = {x});

    std::ostringstream oss;
    oss << dt;
    REQUIRE(oss.str() == "Highest diff order: 0\nNumber of outputs : 1\nDiff arguments    : [x]\n");

    REQUIRE(dt.size() == 1u);
    REQUIRE(dt.get_order() == 0u);
    REQUIRE(dt.get_nvars() == 1u);
    REQUIRE(dt.get_nouts() == 1u);
    REQUIRE(dt[{0, 0}] == 1_dbl);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 1}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{0, 1})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 2}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{0, 2})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 1, kw::diff_args = {par[0]});

    std::vector<expression> diff_vec;

    auto assign_sr = [&diff_vec](const auto &sr) {
        diff_vec.clear();
        for (const auto &p : sr) {
            diff_vec.push_back(p.second);
        }
    };

    REQUIRE(dt.size() == 2u);

    assign_sr(dt.get_derivatives(0, 0));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 1_dbl);
    assign_sr(dt.get_derivatives(0, 1));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 2}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{0, 2})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 2, kw::diff_args = {par[0]});
    REQUIRE(dt.size() == 3u);
    assign_sr(dt.get_derivatives(0, 0));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 1_dbl);
    assign_sr(dt.get_derivatives(0, 1));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    assign_sr(dt.get_derivatives(0, 2));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);
    REQUIRE(dt[{0, 2}] == 0_dbl);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 3}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{0, 3})));

    dt = diff_tensors({1_dbl}, kw::diff_order = 3, kw::diff_args = {par[0]});
    REQUIRE(dt.size() == 4u);
    assign_sr(dt.get_derivatives(0, 0));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 1_dbl);
    assign_sr(dt.get_derivatives(0, 1));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    assign_sr(dt.get_derivatives(0, 2));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    assign_sr(dt.get_derivatives(0, 3));
    REQUIRE(diff_vec.size() == 1u);
    REQUIRE(diff_vec[0] == 0_dbl);
    REQUIRE(dt[{0, 0}] == 1_dbl);
    REQUIRE(dt[{0, 1}] == 0_dbl);
    REQUIRE(dt[{0, 2}] == 0_dbl);
    REQUIRE(dt[{0, 3}] == 0_dbl);

    REQUIRE_THROWS_MATCHES(
        (dt[{1, 0}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{1, 0})));
    REQUIRE_THROWS_MATCHES(
        (dt[{0, 4}]), std::out_of_range,
        Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}", std::vector{0, 4})));

    // Automatically deduced diff variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2);
    REQUIRE(dt.size() == 12u);
    assign_sr(dt.get_derivatives(0));
    REQUIRE(diff_vec == std::vector{x + y, x * y * y});
    assign_sr(dt.get_derivatives(1));
    // TODO restore.
    // REQUIRE(diff_vec == std::vector{1_dbl, 1_dbl, y * y, sum({(y * x), (x * y)})});
    assign_sr(dt.get_derivatives(2));
    REQUIRE(diff_vec == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 2. * y, 2. * x});

    // Diff wrt all variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2, kw::diff_args = diff_args::vars);
    REQUIRE(dt.size() == 12u);
    assign_sr(dt.get_derivatives(0));
    REQUIRE(diff_vec == std::vector{x + y, x * y * y});
    assign_sr(dt.get_derivatives(1));
    // TODO restore.
    // REQUIRE(diff_vec == std::vector{1_dbl, 1_dbl, y * y, sum({(y * x), (x * y)})});
    assign_sr(dt.get_derivatives(2));
    REQUIRE(diff_vec == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 2. * y, 2. * x});

    // Diff wrt some variables.
    dt = diff_tensors({x + y, x * y * y}, kw::diff_order = 2, kw::diff_args = {x});
    REQUIRE(dt.size() == 6u);
    assign_sr(dt.get_derivatives(0));
    REQUIRE(diff_vec == std::vector{x + y, x * y * y});
    assign_sr(dt.get_derivatives(1));
    REQUIRE(diff_vec == std::vector{1_dbl, y * y});
    assign_sr(dt.get_derivatives(2));
    REQUIRE(diff_vec == std::vector{0_dbl, 0_dbl});

    // Diff wrt all params.
    dt = diff_tensors({par[0] + y, x * y * par[1]}, kw::diff_order = 2, kw::diff_args = diff_args::params);
    REQUIRE(dt.size() == 12u);
    assign_sr(dt.get_derivatives(0));
    REQUIRE(diff_vec == std::vector{par[0] + y, x * y * par[1]});
    assign_sr(dt.get_derivatives(1));
    REQUIRE(diff_vec == std::vector{1_dbl, 0_dbl, 0_dbl, x * y});
    assign_sr(dt.get_derivatives(2));
    REQUIRE(diff_vec == std::vector{0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl, 0_dbl});

    // Diff wrt some param.
    dt = diff_tensors({par[0] + y, x * y * par[1]}, kw::diff_order = 2, kw::diff_args = {par[1]});
    REQUIRE(dt.size() == 6u);
    assign_sr(dt.get_derivatives(0));
    REQUIRE(diff_vec == std::vector{par[0] + y, x * y * par[1]});
    assign_sr(dt.get_derivatives(1));
    REQUIRE(diff_vec == std::vector{0_dbl, x * y});
    assign_sr(dt.get_derivatives(2));
    REQUIRE(diff_vec == std::vector{0_dbl, 0_dbl});

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
    using Catch::Matchers::Message;

    dtens dt;

    std::ostringstream oss;
    oss << dt;
    REQUIRE(oss.str() == "Highest diff order: 0\nNumber of outputs : 0\nDiff arguments    : []\n");
    REQUIRE(oss.str() == fmt::format("{}", dt));

    REQUIRE(dt.get_order() == 0u);
    REQUIRE(dt.get_nvars() == 0u);
    REQUIRE(dt.get_nouts() == 0u);
    REQUIRE(dt.size() == 0u);
    REQUIRE(dt.index_of(dt.end()) == 0u);
    REQUIRE(dt.index_of(dtens::v_idx_t{}) == 0u);
    REQUIRE(dt.index_of(dtens::v_idx_t{1, 2, 3}) == 0u);

    REQUIRE(dt.begin() == dt.end());
    REQUIRE(dt.find({}) == dt.end());
    REQUIRE(dt.find({0, 1, 2}) == dt.end());

    REQUIRE_THROWS_MATCHES((dt[{}]), std::out_of_range,
                           Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}",
                                               std::vector<int>{})));
    REQUIRE_THROWS_MATCHES((dt[{0, 1, 2}]), std::out_of_range,
                           Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}",
                                               std::vector<int>{0, 1, 2})));

    auto sr = dt.get_derivatives(5);
    REQUIRE(sr.begin() == dt.end());
    REQUIRE(sr.end() == dt.end());

    auto sr2(sr);
    REQUIRE(sr2.begin() == sr.begin());
    REQUIRE(sr2.end() == sr.end());

    auto sr3(std::move(sr2));
    REQUIRE(sr3.begin() == sr.begin());
    REQUIRE(sr3.end() == sr.end());

    sr2 = sr3;
    REQUIRE(sr2.begin() == sr.begin());
    REQUIRE(sr2.end() == sr.end());

    sr = dt.get_derivatives(3, 5);
    REQUIRE(sr.begin() == dt.end());
    REQUIRE(sr.end() == dt.end());

    auto [x, y] = make_vars("x", "y");

    auto dt2 = diff_tensors({x + y, x * y}, kw::diff_order = 1);

    REQUIRE(dt2.get_order() == 1u);
    REQUIRE(dt2.get_nvars() == 2u);
    REQUIRE(dt2.get_nouts() == 2u);
    REQUIRE(dt2.get_args() == std::vector{x, y});
    REQUIRE(dt2.find({0, 1, 0}) != dt2.end());
    REQUIRE(dt2.find({0, 1}) == dt2.end());
    REQUIRE(dt2.find({0, 3, 0}) == dt2.end());
    REQUIRE(dt2.index_of(dt2.begin()) == 0u);
    REQUIRE(dt2.index_of(dt2.begin() + 1) == 1u);
    REQUIRE(dt2.index_of(dt2.end()) == dt2.size());
    REQUIRE(dt2.index_of(dtens::v_idx_t{}) == dt2.size());
    REQUIRE(dt2.index_of(dtens::v_idx_t{0, 0, 0}) == 0u);
    REQUIRE(dt2.index_of(dtens::v_idx_t{4, 0, 0}) == dt2.size());

    REQUIRE(dt2[{0, 1, 0}] == 1_dbl);
    REQUIRE_THROWS_MATCHES((dt2[{0, 1}]), std::out_of_range,
                           Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}",
                                               std::vector<int>{0, 1})));
    REQUIRE_THROWS_MATCHES((dt2[{0, 3, 0}]), std::out_of_range,
                           Message(fmt::format("Cannot locate the derivative corresponding to the indices vector {}",
                                               std::vector<int>{0, 3, 0})));

    sr = dt2.get_derivatives(1);
    REQUIRE(sr.begin() != dt2.end());
    REQUIRE(sr.end() == dt2.end());

    sr = dt2.get_derivatives(3);
    REQUIRE(sr.begin() == dt2.end());
    REQUIRE(sr.end() == dt2.end());

    sr = dt2.get_derivatives(0, 1);
    REQUIRE(sr.begin() != dt2.end());
    REQUIRE(sr.end() != dt2.end());

    sr = dt2.get_derivatives(1, 1);
    REQUIRE(sr.begin() != dt2.end());
    REQUIRE(sr.end() == dt2.end());

    sr = dt2.get_derivatives(2, 1);
    REQUIRE(sr.begin() == dt2.end());
    REQUIRE(sr.end() == dt2.end());

    // Overflow throwing.
    REQUIRE_THROWS(dt2[{0, std::numeric_limits<std::uint32_t>::max(), 1u}]);

    // Copy/move semantics.
    auto dt3(dt2);

    REQUIRE(dt3.size() == dt2.size());
    REQUIRE(std::equal(dt3.begin(), dt3.end(), dt2.begin()));
    REQUIRE(dt3.get_args() == dt2.get_args());

    auto dt4(std::move(dt3));
    dt3 = dt4;

    REQUIRE(dt3.size() == dt2.size());
    REQUIRE(std::equal(dt3.begin(), dt3.end(), dt2.begin()));
    REQUIRE(dt3.get_args() == dt2.get_args());

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

    REQUIRE(dt3.size() == dt2.size());
    REQUIRE(std::equal(dt3.begin(), dt3.end(), dt2.begin()));
    REQUIRE(dt3.get_args() == dt2.get_args());
}

TEST_CASE("fixed centres check")
{
    std::uniform_real_distribution<double> rdist(-10., 10.);

    const auto fc_energy = model::fixed_centres_energy(
        kw::masses = {1., 1., 1.},
        kw::positions = {par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8]});

    const auto fc_pot = model::fixed_centres_potential(
        kw::masses = {1., 1., 1.},
        kw::positions = {par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8]});

    const auto dt = diff_tensors({fc_energy, fc_pot}, kw::diff_order = 3, kw::diff_args = diff_args::all);

    const auto vars = std::vector{"vx"_var, "vy"_var, "vz"_var, "x"_var, "y"_var, "z"_var};

    const auto diff_vars = std::vector{"vx"_var, "vy"_var, "vz"_var, "x"_var, "y"_var, "z"_var, par[0], par[1],
                                       par[2],   par[3],   par[4],   par[5],  par[6],  par[7],  par[8]};

    const auto nvars = 6u;
    const auto npars = 9u;

    std::vector<expression> diff_vec;

    auto assign_sr = [&diff_vec](const auto &sr) {
        diff_vec.clear();
        for (const auto &p : sr) {
            diff_vec.push_back(p.second);
        }
    };

    for (auto diff_order = 0u; diff_order <= 3u; ++diff_order) {
        const auto sr = dt.get_derivatives(diff_order);
        REQUIRE(sr.begin() != dt.end());
        REQUIRE(sr.begin() != sr.end());
        assign_sr(sr);

        llvm_state s;
        add_cfunc<double>(s, "diff", diff_vec, kw::vars = vars);
        s.optimise();
        s.compile();

        auto *fr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("diff"));

        // Randomly-generate inputs.
        std::vector<double> inputs(nvars);
        for (auto &ival : inputs) {
            ival = rdist(rng);
        }

        // Randomly generate pars.
        std::vector<double> pars(npars);
        for (auto &pval : pars) {
            pval = rdist(rng);
        }

        // Prepare the outputs vector.
        std::vector<double> outputs(diff_vec.size());

        // Evaluate.
        fr(outputs.data(), inputs.data(), pars.data(), nullptr);

        auto sr_it = sr.begin();
        for (decltype(diff_vec.size()) i = 0; i < diff_vec.size(); ++i, ++sr_it) {
            REQUIRE(sr_it != sr.end());

            const auto &v_idx = sr_it->first;

            REQUIRE(v_idx.size() == diff_vars.size() + 1u);

            // Build the current derivative via repeated
            // invocations of diff().
            auto ex = v_idx[0] == 0u ? fc_energy : fc_pot;

            for (auto j = 1u; j < v_idx.size(); ++j) {
                auto order = v_idx[j];

                for (decltype(order) k = 0; k < order; ++k) {
                    ex = diff(ex, diff_vars[j - 1u]);
                }
            }

            // Compile and fetch the expression of the derivative.
            llvm_state s2;
            add_cfunc<double>(s2, "diff", {ex}, kw::vars = vars);
            s2.optimise();
            s2.compile();

            auto *fr2 = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                s2.jit_lookup("diff"));

            // Evaluate the derivative.
            double out = 0;
            fr2(&out, inputs.data(), pars.data(), nullptr);

            REQUIRE(out == approximately(outputs[i]));
        }

        REQUIRE(sr_it == sr.end());
    }
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

    std::vector<expression> diff_vec;

    auto assign_sr = [&diff_vec](const auto &sr) {
        diff_vec.clear();
        for (const auto &p : sr) {
            diff_vec.push_back(p.second);
        }
    };

    for (auto diff_order = 0u; diff_order <= 3u; ++diff_order) {
        const auto sr = dt.get_derivatives(diff_order);
        REQUIRE(sr.begin() != dt.end());
        REQUIRE(sr.begin() != sr.end());
        assign_sr(sr);

        llvm_state s;
        add_cfunc<double>(s, "diff", diff_vec, kw::vars = vars);
        s.optimise();
        s.compile();

        auto *fr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("diff"));

        // Randomly-generate inputs.
        std::vector<double> inputs(nvars);
        for (auto &ival : inputs) {
            ival = rdist(rng);
        }

        // Prepare the outputs vector.
        std::vector<double> outputs(diff_vec.size());

        // Evaluate.
        fr(outputs.data(), inputs.data(), nullptr, nullptr);

        auto sr_it = sr.begin();
        for (decltype(diff_vec.size()) i = 0; i < diff_vec.size(); ++i, ++sr_it) {
            REQUIRE(sr_it != sr.end());

            const auto &v_idx = sr_it->first;

            REQUIRE(v_idx.size() == nvars + 1u);
            REQUIRE(v_idx[0] == 0u);

            // Build the current derivative via repeated
            // invocations of diff().
            auto ex = prod;

            for (auto j = 1u; j < v_idx.size(); ++j) {
                auto order = v_idx[j];

                for (decltype(order) k = 0; k < order; ++k) {
                    ex = diff(ex, vars[j - 1u]);
                }
            }

            // Compile and fetch the expression of the derivative.
            llvm_state s2;
            add_cfunc<double>(s2, "diff", {ex}, kw::vars = vars);
            s2.optimise();
            s2.compile();

            auto *fr2 = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                s2.jit_lookup("diff"));

            // Evaluate the derivative.
            double out = 0;
            fr2(&out, inputs.data(), nullptr, nullptr);

            REQUIRE(out == approximately(outputs[i]));
        }

        REQUIRE(sr_it == sr.end());
    }
}

TEST_CASE("speelpenning complexity")
{
    fmt::print("Speelpenning's example\n");
    fmt::print("======================\n");

    std::uniform_real_distribution<double> rdist(-10., 10.);

    std::vector<expression> diff_vec;

    auto assign_sr = [&diff_vec](const auto &sr) {
        diff_vec.clear();
        for (const auto &p : sr) {
            diff_vec.push_back(p.second);
        }
    };

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
        auto dt = diff_tensors({prod}, kw::diff_order = 1);
        auto sr = dt.get_derivatives(1);
        assign_sr(sr);

        auto dc_reverse = add_cfunc<double>(s, "f_reverse", diff_vec, kw::compact_mode = true);

        fmt::print("nvars={:<5} decomposition size={:<6}\n", nvars, dc_reverse.size() - nvars - nvars);

        s.optimise();
        s.compile();
    }
}

// TODO restore.
#if 0

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

#endif
