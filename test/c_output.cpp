// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <initializer_list>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("scalar")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        auto ta = taylor_adaptive<fp_t>{{prime(x) = v, prime(v) = -x}, {0., 1.}, kw::opt_level = opt_level};

        auto d_out_opt = std::get<4>(ta.propagate_until(10., kw::c_output = true));

        REQUIRE(d_out_opt.has_value());

        auto &d_out = *d_out_opt;

        // Reset time/state.
        ta.get_state_data()[0] = 0;
        ta.get_state_data()[1] = 1;
        ta.set_time(0);

        // Run a grid propagation.
        const auto t_grid = std::vector<fp_t>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
        auto grid_out = std::get<4>(ta.propagate_grid(t_grid));

        for (auto i = 0u; i < 11u; ++i) {
            d_out(t_grid[i]);
            REQUIRE(d_out.get_output()[0] == grid_out[2u * i]);
            REQUIRE(d_out.get_output()[1] == grid_out[2u * i + 1u]);
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tuple_for_each(fp_types, [&tester, opt_level](auto x) { tester(x, opt_level); });
    }
}
