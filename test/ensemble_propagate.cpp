// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/ensemble_propagate.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(__FreeBSD__)
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
#endif
                                 >{};

static std::mt19937 rng;

TEST_CASE("scalar propagate until")
{

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        REQUIRE(ensemble_propagate_until<fp_t>(ta, 20, 0, [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];

                    return tint;
                }).empty());

        auto res = ensemble_propagate_until<fp_t>(ta, 20, n_iter, [&ics](auto tint, std::size_t i) {
            tint.get_state_data()[0] = ics[i][0];
            tint.get_state_data()[1] = ics[i][1];

            return tint;
        });

        REQUIRE(res.size() == n_iter);

        // Compare.
        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(0);
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];

            auto loc_res = ta.propagate_until(20);

            REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20), fp_t(10)));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
            REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
            REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
            REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
            REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
        }

        // Do it with continuous output too.
        ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

        res = ensemble_propagate_until<fp_t>(
            ta, 20, n_iter,
            [&ics](auto tint, std::size_t i) {
                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];

                return tint;
            },
            kw::c_output = true);

        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(0);
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];

            auto loc_res = ta.propagate_until(20, kw::c_output = true);

            REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20), fp_t(10)));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
            REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
            REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
            REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
            REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
            REQUIRE((*std::get<5>(res[i]))(1.5) == (*std::get<4>(loc_res))(1.5));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("scalar propagate for")
{

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        REQUIRE(ensemble_propagate_for<fp_t>(ta, 20, 0, [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];

                    return tint;
                }).empty());

        auto res = ensemble_propagate_for<fp_t>(ta, 20, n_iter, [&ics](auto tint, std::size_t i) {
            tint.set_time(10);
            tint.get_state_data()[0] = ics[i][0];
            tint.get_state_data()[1] = ics[i][1];

            return tint;
        });

        REQUIRE(res.size() == n_iter);

        // Compare.
        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(10);
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];

            auto loc_res = ta.propagate_for(20);

            REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(30), fp_t(10)));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
            REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
            REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
            REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
            REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
        }

        // Do it with continuous output too.
        ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

        res = ensemble_propagate_for<fp_t>(
            ta, 20, n_iter,
            [&ics](auto tint, std::size_t i) {
                tint.set_time(10);
                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];

                return tint;
            },
            kw::c_output = true);

        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(10);
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];

            auto loc_res = ta.propagate_for(20, kw::c_output = true);

            REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(30), fp_t(10)));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
            REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
            REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
            REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
            REQUIRE(std::get<5>(res[i]).has_value() == std::get<4>(loc_res).has_value());
            REQUIRE((*std::get<5>(res[i]))(1.5) == (*std::get<4>(loc_res))(1.5));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("scalar propagate grid")
{

    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        // Create a regular grid.
        std::vector<fp_t> grid;
        for (auto i = 0; i <= 20; ++i) {
            grid.emplace_back(i);
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
            ta.set_time(0);
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];

            auto loc_res = ta.propagate_grid(grid);

            REQUIRE(std::get<0>(res[i]).get_time() == approximately(fp_t(20), fp_t(10)));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<1>(res[i]) == std::get<0>(loc_res));
            REQUIRE(std::get<2>(res[i]) == std::get<1>(loc_res));
            REQUIRE(std::get<3>(res[i]) == std::get<2>(loc_res));
            REQUIRE(std::get<4>(res[i]) == std::get<3>(loc_res));
            REQUIRE(std::get<5>(res[i]) == std::get<4>(loc_res));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("batch propagate until")
{
    auto tester = [](auto fp_x) {
        const auto batch_size = 2u;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, batch_size};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        REQUIRE(ensemble_propagate_until_batch<fp_t>(ta, 20, 0, [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];
                    tint.get_state_data()[2] = ics[i][2];
                    tint.get_state_data()[3] = ics[i][3];

                    return tint;
                }).empty());

        auto res = ensemble_propagate_until_batch<fp_t>(ta, 20, n_iter, [&ics](auto tint, std::size_t i) {
            tint.get_state_data()[0] = ics[i][0];
            tint.get_state_data()[1] = ics[i][1];
            tint.get_state_data()[2] = ics[i][2];
            tint.get_state_data()[3] = ics[i][3];

            return tint;
        });

        REQUIRE(res.size() == n_iter);

        // Compare.
        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(std::vector<fp_t>(batch_size, fp_t(0)));
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];
            ta.get_state_data()[2] = ics[i][2];
            ta.get_state_data()[3] = ics[i][3];

            auto loc_res = ta.propagate_until(20);

            REQUIRE(std::all_of(std::get<0>(res[i]).get_time().begin(), std::get<0>(res[i]).get_time().end(),
                                [](fp_t t) { return t == approximately(fp_t(20), fp_t(10)); }));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<0>(res[i]).get_propagate_res() == ta.get_propagate_res());
            REQUIRE(std::get<1>(res[i]).has_value() == loc_res.has_value());
        }

        // Do it with continuous output too.
        ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, batch_size};

        res = ensemble_propagate_until_batch<fp_t>(
            ta, 20, n_iter,
            [&ics](auto tint, std::size_t i) {
                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];
                tint.get_state_data()[2] = ics[i][2];
                tint.get_state_data()[3] = ics[i][3];

                return tint;
            },
            kw::c_output = true);

        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(std::vector<fp_t>(batch_size, fp_t(0)));
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];
            ta.get_state_data()[2] = ics[i][2];
            ta.get_state_data()[3] = ics[i][3];

            auto loc_res = ta.propagate_until(std::vector<fp_t>(batch_size, fp_t(20)), kw::c_output = true);

            REQUIRE(std::all_of(std::get<0>(res[i]).get_time().begin(), std::get<0>(res[i]).get_time().end(),
                                [](fp_t t) { return t == approximately(fp_t(20), fp_t(10)); }));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<0>(res[i]).get_propagate_res() == ta.get_propagate_res());
            REQUIRE((*std::get<1>(res[i]))(1.5) == (*(loc_res))(1.5));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("batch propagate for")
{
    auto tester = [](auto fp_x) {
        const auto batch_size = 2u;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, batch_size};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        REQUIRE(ensemble_propagate_for_batch<fp_t>(ta, 20, 0, [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];
                    tint.get_state_data()[2] = ics[i][2];
                    tint.get_state_data()[3] = ics[i][3];

                    return tint;
                }).empty());

        auto res = ensemble_propagate_for_batch<fp_t>(ta, 20, n_iter, [&ics](auto tint, std::size_t i) {
            tint.set_time(fp_t(10));

            tint.get_state_data()[0] = ics[i][0];
            tint.get_state_data()[1] = ics[i][1];
            tint.get_state_data()[2] = ics[i][2];
            tint.get_state_data()[3] = ics[i][3];

            return tint;
        });

        REQUIRE(res.size() == n_iter);

        // Compare.
        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(fp_t(10));
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];
            ta.get_state_data()[2] = ics[i][2];
            ta.get_state_data()[3] = ics[i][3];

            auto loc_res = ta.propagate_for(20);

            REQUIRE(std::all_of(std::get<0>(res[i]).get_time().begin(), std::get<0>(res[i]).get_time().end(),
                                [](fp_t t) { return t == approximately(fp_t(30), fp_t(10)); }));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<0>(res[i]).get_propagate_res() == ta.get_propagate_res());
            REQUIRE(std::get<1>(res[i]).has_value() == loc_res.has_value());
        }

        // Do it with continuous output too.
        ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, batch_size};

        res = ensemble_propagate_for_batch<fp_t>(
            ta, 20, n_iter,
            [&ics](auto tint, std::size_t i) {
                tint.set_time(fp_t(10));

                tint.get_state_data()[0] = ics[i][0];
                tint.get_state_data()[1] = ics[i][1];
                tint.get_state_data()[2] = ics[i][2];
                tint.get_state_data()[3] = ics[i][3];

                return tint;
            },
            kw::c_output = true);

        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(fp_t(10));
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];
            ta.get_state_data()[2] = ics[i][2];
            ta.get_state_data()[3] = ics[i][3];

            auto loc_res = ta.propagate_for(std::vector<fp_t>(batch_size, fp_t(20)), kw::c_output = true);

            REQUIRE(std::all_of(std::get<0>(res[i]).get_time().begin(), std::get<0>(res[i]).get_time().end(),
                                [](fp_t t) { return t == approximately(fp_t(30), fp_t(10)); }));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<0>(res[i]).get_propagate_res() == ta.get_propagate_res());
            REQUIRE((*std::get<1>(res[i]))(1.5) == (*(loc_res))(1.5));
        }
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("batch propagate grid")
{

    auto tester = [](auto fp_x) {
        const auto batch_size = 2u;

        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 0., 1., 1.}, batch_size};

        const auto n_iter = 128u;

        std::vector<std::vector<fp_t>> ics;
        ics.resize(n_iter);

        std::uniform_real_distribution<float> rdist(-static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100),
                                                    static_cast<float>(std::numeric_limits<fp_t>::epsilon() * 100));
        for (auto &ic : ics) {
            ic.push_back(rdist(rng));
            ic.push_back(rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
            ic.push_back(fp_t(1) + rdist(rng));
        }

        // Create regular grids.
        std::vector<fp_t> grid, grid_splat;
        for (auto i = 0; i <= 20; ++i) {
            grid.emplace_back(i);

            for (auto j = 0u; j < batch_size; ++j) {
                grid_splat.emplace_back(i);
            }
        }

        REQUIRE(ensemble_propagate_grid_batch<fp_t>(ta, grid, 0, [&ics](auto tint, std::size_t i) {
                    tint.get_state_data()[0] = ics[i][0];
                    tint.get_state_data()[1] = ics[i][1];
                    tint.get_state_data()[2] = ics[i][2];
                    tint.get_state_data()[3] = ics[i][3];

                    return tint;
                }).empty());

        auto res = ensemble_propagate_grid_batch<fp_t>(ta, grid, n_iter, [&ics](auto tint, std::size_t i) {
            tint.get_state_data()[0] = ics[i][0];
            tint.get_state_data()[1] = ics[i][1];
            tint.get_state_data()[2] = ics[i][2];
            tint.get_state_data()[3] = ics[i][3];

            return tint;
        });

        REQUIRE(res.size() == n_iter);

        // Compare.
        for (auto i = 0u; i < n_iter; ++i) {
            // Use ta for the comparison.
            ta.set_time(std::vector<fp_t>(batch_size, fp_t(0)));
            ta.get_state_data()[0] = ics[i][0];
            ta.get_state_data()[1] = ics[i][1];
            ta.get_state_data()[2] = ics[i][2];
            ta.get_state_data()[3] = ics[i][3];

            auto loc_res = ta.propagate_grid(grid_splat);

            REQUIRE(std::all_of(std::get<0>(res[i]).get_time().begin(), std::get<0>(res[i]).get_time().end(),
                                [](fp_t t) { return t == approximately(fp_t(20), fp_t(10)); }));
            REQUIRE(std::get<0>(res[i]).get_state() == ta.get_state());
            REQUIRE(std::get<0>(res[i]).get_propagate_res() == ta.get_propagate_res());
            REQUIRE(std::get<1>(res[i]) == loc_res);
        }
    };

    tuple_for_each(fp_types, tester);
}
