// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const expression &x)
{
    return x * x;
}

TEST_CASE("decompose sys")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    using sys_t = std::vector<std::pair<expression, expression>>;

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{}, {}), std::invalid_argument,
                           Message("Cannot decompose a system of zero equations"));

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{prime(x) = y, prime(x) = y}, {}), std::invalid_argument,
                           Message("Error in the Taylor decomposition of a system of equations: the variable 'x' "
                                   "appears in the left-hand side twice"));

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{{par[0], x}}, {}), std::invalid_argument,
                           Message("Error in the Taylor decomposition of a system of equations: the "
                                   "left-hand side contains the expression 'p0', which is not a variable"));

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{prime(x) = y}, {}), std::invalid_argument,
                           Message("Error in the Taylor decomposition of a system of equations: the variable 'y' "
                                   "appears in the right-hand side but not in the left-hand side"));

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{prime(x) = x}, {y}), std::invalid_argument,
                           Message("The extra functions in a Taylor decomposition contain the variable 'y', "
                                   "which is not a state variable"));

    REQUIRE_THROWS_MATCHES(taylor_decompose(sys_t{prime(x) = x}, {1_dbl}), std::invalid_argument,
                           Message("The extra functions in a Taylor decomposition cannot be constants or parameters"));

    auto [dc, sv_funcs_dc] = taylor_decompose(sys_t{prime(x) = x}, std::vector<expression>{x + 1.});

    REQUIRE(dc.size() == 3u);
    REQUIRE(sv_funcs_dc.size() == 1u);
    REQUIRE(sv_funcs_dc[0] == 1u);

    std::tie(dc, sv_funcs_dc)
        = taylor_decompose(sys_t{prime(y) = x + y, prime(x) = x - y}, std::vector<expression>{x, x * y});

    REQUIRE(dc.size() == 7u);
    REQUIRE(sv_funcs_dc.size() == 2u);
    REQUIRE(sv_funcs_dc[0] == 1u);
    REQUIRE(sv_funcs_dc[1] == 4u);

    std::tie(dc, sv_funcs_dc)
        = taylor_decompose(sys_t{prime(y) = x + y, prime(x) = x - y}, std::vector<expression>{x, x, y, y});

    REQUIRE(dc.size() == 6u);
    REQUIRE(sv_funcs_dc.size() == 4u);
    REQUIRE(sv_funcs_dc[0] == 1u);
    REQUIRE(sv_funcs_dc[1] == 1u);
    REQUIRE(sv_funcs_dc[2] == 0u);
    REQUIRE(sv_funcs_dc[3] == 0u);

    std::tie(dc, sv_funcs_dc)
        = taylor_decompose(sys_t{prime(y) = x + y, prime(x) = x - y}, std::vector<expression>{x + y, x, y, x - y});

    REQUIRE(dc.size() == 6u);
    REQUIRE(sv_funcs_dc.size() == 4u);
    REQUIRE(sv_funcs_dc[0] == 2u);
    REQUIRE(sv_funcs_dc[1] == 1u);
    REQUIRE(sv_funcs_dc[2] == 0u);
    REQUIRE(sv_funcs_dc[3] == 3u);

    // A more complex example with a 6-body system and collision-like events.
    auto sys = model::nbody(6);

    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;
    std::vector<expression> sv_funcs;

    for (auto i = 0; i < 6; ++i) {
        const auto i_str = std::to_string(i);

        x_vars.emplace_back("x_" + i_str);
        y_vars.emplace_back("y_" + i_str);
        z_vars.emplace_back("z_" + i_str);

        vx_vars.emplace_back("vx_" + i_str);
        vy_vars.emplace_back("vy_" + i_str);
        vz_vars.emplace_back("vz_" + i_str);
    }

    for (unsigned i = 0; i < 6u; ++i) {
        for (unsigned j = i + 1u; j < 6u; ++j) {
            auto diff_x = x_vars[j] - x_vars[i];
            auto diff_y = y_vars[j] - y_vars[i];
            auto diff_z = z_vars[j] - z_vars[i];

            sv_funcs.push_back(sqrt(square_wrapper(diff_x) + square_wrapper(diff_y) + square_wrapper(diff_z)));
        }
    }

    // Let the internal assertions of taylor_decompose do the job.
    std::tie(dc, sv_funcs_dc) = taylor_decompose(sys, sv_funcs);
}
