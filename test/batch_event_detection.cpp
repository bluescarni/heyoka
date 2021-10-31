// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
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

TEST_CASE("single step")
{
    auto tester = [](unsigned opt_level) {
        using fp_t = double;

        const auto batch_size = 4u;

        auto [x, v] = make_vars("x", "v");

        // Batches of initial conditions and parameters.
        const std::vector<fp_t> ic_vals = {0.00, 0.01, 0.02, 0.03, 1.85, 1.86, 1.87, 1.88};
        const std::vector<fp_t> par_vals = {0.10, 0.11, 0.12, 0.13};

        // Dynamics.
        auto dynamics = {prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)};

        // Events.
        const auto ev_eq = x + .1;
        std::vector<fp_t> trig_times;
        nt_event<fp_t> ev{ev_eq, [&trig_times](auto &, fp_t tm, int) { trig_times.push_back(tm); },
                          kw::direction = event_direction::negative};
        std::vector<std::vector<fp_t>> trig_times_batch(batch_size);
        nt_batch_event<fp_t> ev_batch{ev_eq,
                                      [&trig_times_batch](auto &, fp_t tm, int, std::uint32_t batch_idx) {
                                          trig_times_batch[batch_idx].push_back(tm);
                                      },
                                      kw::direction = event_direction::negative};

        auto ta = taylor_adaptive<fp_t>{dynamics, {0., 0.}, kw::opt_level = opt_level, kw::nt_events = {ev}};
        auto ta_batch = taylor_adaptive_batch<fp_t>{
            dynamics, ic_vals, batch_size, kw::opt_level = opt_level, kw::pars = par_vals, kw::nt_events = {ev_batch}};

        // Integrate the batch integrator up to t = 20.
        while (std::any_of(ta_batch.get_time().begin(), ta_batch.get_time().end(), [](auto tm) { return tm < 20; })) {
            ta_batch.step();

            for (std::uint32_t i = 0; i < batch_size; ++i) {
                REQUIRE(std::get<0>(ta_batch.get_step_res()[i]) == taylor_outcome::success);
            }
        }

        // For each batch index, integrate the scalar integrator up to
        // t = 20 and compare the results.
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            // Reset the scalar integrator.
            trig_times.clear();
            ta.set_time(0.);
            ta.get_state_data()[0] = ic_vals[i];
            ta.get_state_data()[1] = ic_vals[batch_size + i];
            ta.get_pars_data()[0] = par_vals[i];

            while (ta.get_time() < 20) {
                ta.step();
            }

            REQUIRE(trig_times.size() == trig_times_batch[i].size());
            for (decltype(trig_times.size()) j = 0; j < trig_times.size(); ++j) {
                REQUIRE(trig_times[j] == approximately(trig_times_batch[i][j]));
            }
        }
    };

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        tester(opt_level);
    }
}
