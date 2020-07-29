// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("two body")
{
    auto tester = [](auto fp_x, unsigned opt_level) {
        using fp_t = decltype(fp_x);

        const auto batch_size = llvm_state{""}.vector_size<fp_t>();

        if (batch_size == 0u) {
            return;
        }

        auto [vx0, vx1, vy0, vy1, vz0, vz1, x0, x1, y0, y1, z0, z1]
            = make_vars("vx0", "vx1", "vy0", "vy1", "vz0", "vz1", "x0", "x1", "y0", "y1", "z0", "z1");

        auto x01 = x1 - x0;
        auto y01 = y1 - y0;
        auto z01 = z1 - z0;
        auto r01_m3
            = pow(x01 * x01 + y01 * y01 + z01 * z01, expression{number{fp_t{-3}}} / expression{number{fp_t{2}}});

        // The system of equations.
        auto sys = {x01 * r01_m3, -x01 * r01_m3, y01 * r01_m3, -y01 * r01_m3, z01 * r01_m3, -z01 * r01_m3,
                    vx0,          vx1,           vy0,          vy1,           vz0,          vz1};

        std::vector<fp_t> init_state{fp_t{0.593},     fp_t{-0.593},   fp_t{0},  fp_t{0}, fp_t{0}, fp_t{0},
                                     fp_t{-1.000001}, fp_t{1.000001}, fp_t{-1}, fp_t{1}, fp_t{0}, fp_t{0}};

        // Initialise time and states for the batch integrator.
        std::vector<fp_t> init_states, times;
        for (const auto &x : init_state) {
            for (auto i = 0u; i < batch_size; ++i) {
                init_states.push_back(x);
            }
        }
        for (auto i = 0u; i < batch_size; ++i) {
            times.push_back(fp_t{0});
        }

        taylor_adaptive_batch<fp_t> tab{sys,
                                        std::move(init_states),
                                        std::move(times),
                                        std::numeric_limits<fp_t>::epsilon(),
                                        std::numeric_limits<fp_t>::epsilon(),
                                        batch_size,
                                        opt_level};

        const auto &bst = tab.get_states();

        std::vector<std::tuple<taylor_outcome, fp_t, std::uint32_t>> res, s_res;
        s_res.resize(batch_size);

        // Create corresponding scalar integrators.
        std::vector<taylor_adaptive<fp_t>> v_ta;
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            v_ta.emplace_back(sys, init_state, fp_t{0}, std::numeric_limits<fp_t>::epsilon(),
                              std::numeric_limits<fp_t>::epsilon(), opt_level);
        }

        for (auto i = 0; i < 200; ++i) {
            tab.step(res);

            for (std::uint32_t i = 0; i < batch_size; ++i) {
                s_res[i] = v_ta[i].step();

                // NOTE: these 1E5 tolerances can be reduced once
                // we have a way of disabling the fast math flags
                // in llvm_state.
                REQUIRE(std::get<0>(s_res[i]) == std::get<0>(res[i]));
                REQUIRE(std::get<1>(s_res[i]) == approximately(std::get<1>(res[i]), fp_t{1E5}));
                REQUIRE(std::get<2>(s_res[i]) == std::get<2>(res[i]));

                const auto &st = v_ta[i].get_state();

                for (std::uint32_t j = 0; j < 12u; ++j) {
                    REQUIRE(st[j] == approximately(bst[j * batch_size + i], fp_t{1E5}));
                }
            }
        }
    };

    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 0); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 1); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 2); });
    tuple_for_each(fp_types, [&tester](auto x) { tester(x, 3); });
}
