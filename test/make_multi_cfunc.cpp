// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sgp4.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("basic")
{
    auto [x, y] = make_vars("x", "y");

    // Scalar test.
    for (auto opt_level : {0u, 1u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        auto [ms, dc] = detail::make_multi_cfunc<double>(tplt, "test", {x + y + heyoka::time, x - y - par[0]}, {x, y},
                                                         1, false, false, 0);

        ms.compile();

        {
            // Scalar unstrided.
            auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                ms.jit_lookup("test.unstrided.batch_size_1"));

            std::vector<double> ins{1, 2}, outs(2u), pars = {-0.25}, time{0.5};

            cf_s_u(outs.data(), ins.data(), pars.data(), time.data());

            REQUIRE(outs[0] == ins[0] + ins[1] + time[0]);
            REQUIRE(outs[1] == ins[0] - ins[1] - pars[0]);
        }

        {
            // Scalar strided.
            auto *cf_s_s
                = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                    ms.jit_lookup("test.strided.batch_size_1"));

            // Stride value of 3.
            std::vector<double> ins{1, 0, 0, 2, 0, 0}, outs(6u), pars = {-0.25, 0, 0}, time{0.5, 0, 0};

            cf_s_s(outs.data(), ins.data(), pars.data(), time.data(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
        }
    }

    // Batch test.
    for (auto opt_level : {0u, 1u, 3u}) {
        llvm_state tplt{kw::opt_level = opt_level};

        auto [ms, dc] = detail::make_multi_cfunc<double>(tplt, "test", {x + y + heyoka::time, x - y - par[0]}, {x, y},
                                                         2, false, false, 0);

        ms.compile();

        {
            // Scalar unstrided.
            auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
                ms.jit_lookup("test.unstrided.batch_size_1"));

            std::vector<double> ins{1, 2}, outs(2u), pars = {-0.25}, time{0.5};

            cf_s_u(outs.data(), ins.data(), pars.data(), time.data());

            REQUIRE(outs[0] == ins[0] + ins[1] + time[0]);
            REQUIRE(outs[1] == ins[0] - ins[1] - pars[0]);
        }

        {
            // Scalar strided.
            auto *cf_s_s
                = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                    ms.jit_lookup("test.strided.batch_size_1"));

            // Stride value of 3.
            std::vector<double> ins{1, 0, 0, 2, 0, 0}, outs(6u), pars = {-0.25, 0, 0}, time{0.5, 0, 0};

            cf_s_s(outs.data(), ins.data(), pars.data(), time.data(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
        }

        {
            // Batch strided.
            auto *cf_b_s
                = reinterpret_cast<void (*)(double *, const double *, const double *, const double *, std::size_t)>(
                    ms.jit_lookup("test.strided.batch_size_2"));

            // Stride value of 3.
            std::vector<double> ins{1, 1.1, 0, 2, 2.1, 0}, outs(6u), pars = {-0.25, -0.26, 0}, time{0.5, 0.51, 0};

            cf_b_s(outs.data(), ins.data(), pars.data(), time.data(), 3);

            REQUIRE(outs[0] == ins[0] + ins[3] + time[0]);
            REQUIRE(outs[1] == ins[1] + ins[4] + time[1]);
            REQUIRE(outs[3] == ins[0] - ins[3] - pars[0]);
            REQUIRE(outs[4] == ins[1] - ins[4] - pars[1]);
        }

        REQUIRE_THROWS_AS(ms.jit_lookup("test.unstrided.batch_size_2"), std::invalid_argument);
    }
}

TEST_CASE("sgp4")
{
    detail::edb_disabler ed;

    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");

    auto cf = cfunc<double>(model::sgp4(), inputs);

    llvm_state tplt;

    auto [ms, dc] = detail::make_multi_cfunc<double>(tplt, "test", model::sgp4(),
                                                     std::vector(inputs.begin(), inputs.end()), 1, false, false, 0);

    ms.compile();

    auto *cf_s_u = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
        ms.jit_lookup("test.unstrided.batch_size_1"));

    const auto revday2radmin = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 1440.; };
    const auto deg2rad = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 360.; };

    std::vector<double> ins = {revday2radmin(15.50103472202482),
                               0.0007417,
                               deg2rad(51.6439),
                               deg2rad(211.2001),
                               deg2rad(17.6667),
                               deg2rad(85.6398),
                               .38792e-4,
                               0.},
                        outs1(7u), outs2(7u);

    cf_s_u(outs1.data(), ins.data(), nullptr, nullptr);
    cf(outs2, ins);

    for (auto i = 0u; i < 7u; ++i) {
        REQUIRE(outs1[i] == approximately(outs2[i]));
    }
}
