// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <initializer_list>
#include <random>
#include <tuple>
#include <vector>

#include <mp++/real.hpp>

#include <heyoka/detail/real_helpers.hpp>
#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

static std::mt19937 rng;

TEST_CASE("propagate until")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                            {fp_t(0., prec), fp_t(1., prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            const auto n_iter = 128u;

            std::vector<std::vector<fp_t>> ics;
            ics.resize(n_iter);

            std::uniform_real_distribution<float> rdist(-static_cast<float>(detail::eps_from_prec(prec) * 100),
                                                        static_cast<float>(detail::eps_from_prec(prec) * 100));
            for (auto &ic : ics) {
                ic.emplace_back(rdist(rng), prec);
                ic.push_back(fp_t(1, prec) + fp_t(rdist(rng), prec));
            }

            REQUIRE(ensemble_propagate_until<fp_t>(ta, 20, 0, [&ics](auto tint, std::size_t i) {
                        tint.get_state_data()[0] = ics[i][0];
                        tint.get_state_data()[1] = ics[i][1];

                        return tint;
                    }).empty());

            auto res = ensemble_propagate_until<fp_t>(ta, fp_t(20, prec), n_iter, [&ics](auto tint, std::size_t i) {
                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];

                return tint;
            });

            REQUIRE(res.size() == n_iter);

            // Compare.
            for (auto i = 0u; i < n_iter; ++i) {
                // Use ta for the comparison.
                ta.set_time(fp_t(0, prec));
                ta.get_state_data()[0] = ics[i][0];
                ta.get_state_data()[1] = ics[i][1];

                auto loc_res = ta.propagate_until(fp_t(20, prec));

                REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20, prec), fp_t(10, prec)));
                REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
                REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
                REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
                REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
                REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
                REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
            }

            // Do it with continuous output too.
            ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                       {fp_t(0., prec), fp_t(1., prec)},
                                       kw::opt_level = opt_level,
                                       kw::compact_mode = true};

            res = ensemble_propagate_until<fp_t>(
                ta, fp_t(20, prec), n_iter,
                [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];

                    return tint;
                },
                kw::c_output = true);

            for (auto i = 0u; i < n_iter; ++i) {
                // Use ta for the comparison.
                ta.set_time(fp_t(0, prec));
                ta.get_state_data()[0] = ics[i][0];
                ta.get_state_data()[1] = ics[i][1];

                auto loc_res = ta.propagate_until(fp_t(20, prec), kw::c_output = true);

                REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20, prec), fp_t(10, prec)));
                REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
                REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
                REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
                REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
                REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
                REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
                REQUIRE((*std::get<5>(res[i]))(fp_t(1.5, prec)) == (*std::get<4>(loc_res))(fp_t(1.5, prec)));
            }
        }
    }
}

TEST_CASE("ensemble propagate grid")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30, 123}) {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x},
                                            {fp_t(0., prec), fp_t(1., prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true};

            const auto n_iter = 128u;

            std::vector<std::vector<fp_t>> ics;
            ics.resize(n_iter);

            std::uniform_real_distribution<float> rdist(-static_cast<float>(detail::eps_from_prec(prec) * 100),
                                                        static_cast<float>(detail::eps_from_prec(prec) * 100));
            for (auto &ic : ics) {
                ic.emplace_back(rdist(rng), prec);
                ic.push_back(fp_t(1, prec) + fp_t(rdist(rng), prec));
            }

            // Create a regular grid.
            std::vector<fp_t> grid;
            for (auto i = 0; i <= 20; ++i) {
                grid.emplace_back(i, prec);
            }

            REQUIRE(ensemble_propagate_grid<fp_t>(ta, grid, 0, [&ics](auto tint, std::size_t i) {
                        tint.get_state_data()[0] = ics[i][0];
                        tint.get_state_data()[1] = ics[i][1];

                        return tint;
                    }).empty());

            auto res = ensemble_propagate_grid<fp_t>(ta, grid, n_iter, [&ics](auto tint, std::size_t i) {
                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];

                return tint;
            });

            REQUIRE(res.size() == n_iter);

            // Compare.
            for (auto i = 0u; i < n_iter; ++i) {
                // Use ta for the comparison.
                ta.set_time(fp_t(0, prec));
                ta.get_state_data()[0] = ics[i][0];
                ta.get_state_data()[1] = ics[i][1];

                auto loc_res = ta.propagate_grid(grid);

                REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20, prec), fp_t(10, prec)));
                REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
                REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
                REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
                REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
                REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
                REQUIRE(std::get<5>(res[i]) == std::get<4>(loc_res));
            }
        }
    }
}
