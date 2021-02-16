// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("streaming op")
{
    auto sys = make_nbody_sys(2, kw::masses = {1., 0.});

    auto tad = taylor_adaptive<double>{std::move(sys), {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}};

    std::ostringstream oss;
    oss << tad;

    REQUIRE(!oss.str().empty());
}

TEST_CASE("param scalar")
{
    using std::sqrt;
    const auto pi = boost::math::constants::pi<double>();

    const auto init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0};

    auto tad = taylor_adaptive<double>{make_nbody_par_sys(2, kw::n_massive = 1), init_state, kw::tol = 1e-18};

    REQUIRE(tad.get_pars().size() == 1u);

    // Set the central mass to 1.
    tad.get_pars_data()[0] = 1.;

    tad.propagate_for(2 * pi);

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[1] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[3] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[5] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(1.));
    REQUIRE(tad.get_state()[7] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[9] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(1.));
    REQUIRE(tad.get_state()[11] == approximately(0.));

    // Set the central mass to 8
    tad.get_pars_data()[0] = 8.;
    // Set the initial speed so that the orbit stays circular.
    std::copy(init_state.begin(), init_state.end(), tad.get_state_data());
    tad.get_state_data()[10] = sqrt(8.);

    tad.propagate_for(2 * pi * sqrt(1 / 8.));

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[1] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[3] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[5] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(1.));
    REQUIRE(tad.get_state()[7] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[9] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(sqrt(8.)));
    REQUIRE(tad.get_state()[11] == approximately(0.));
}

TEST_CASE("param batch")
{
    using std::cbrt;
    using std::sqrt;
    const auto pi = boost::math::constants::pi<double>();

    auto a0 = 1.;
    auto v0 = 1.;
    auto a1 = cbrt(2);
    auto v1 = sqrt(2 / a1);

    auto init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a0, a1, 0, 0, 0, 0, 0, 0, v0, v1, 0, 0};

    auto tad = taylor_adaptive_batch<double>{make_nbody_par_sys(2, kw::n_massive = 1), init_state, 2, kw::tol = 1e-18};

    REQUIRE(tad.get_pars().size() == 2u);

    // Set the central masses to 1 and 2.
    tad.get_pars_data()[0] = 1.;
    tad.get_pars_data()[1] = 2.;

    tad.propagate_for({2 * pi, 2 * pi});

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(0.));
    REQUIRE(tad.get_state()[12] == approximately(a0));
    REQUIRE(tad.get_state()[14] == approximately(0.));
    REQUIRE(tad.get_state()[16] == approximately(0.));
    REQUIRE(tad.get_state()[18] == approximately(0.));
    REQUIRE(tad.get_state()[20] == approximately(v0));
    REQUIRE(tad.get_state()[22] == approximately(0.));

    REQUIRE(tad.get_state()[0 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[2 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[4 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[6 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[8 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[10 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[12 + 1] == approximately(a1));
    REQUIRE(tad.get_state()[14 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[16 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[18 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[20 + 1] == approximately(v1));
    REQUIRE(tad.get_state()[22 + 1] == approximately(0.));

    // Flip around.
    init_state = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a1, a0, 0, 0, 0, 0, 0, 0, v1, v0, 0, 0};
    std::copy(init_state.begin(), init_state.end(), tad.get_state_data());
    tad.get_pars_data()[0] = 2.;
    tad.get_pars_data()[1] = 1.;

    tad.propagate_for({2 * pi, 2 * pi});

    REQUIRE(tad.get_state()[0] == approximately(0.));
    REQUIRE(tad.get_state()[2] == approximately(0.));
    REQUIRE(tad.get_state()[4] == approximately(0.));
    REQUIRE(tad.get_state()[6] == approximately(0.));
    REQUIRE(tad.get_state()[8] == approximately(0.));
    REQUIRE(tad.get_state()[10] == approximately(0.));
    REQUIRE(tad.get_state()[12] == approximately(a1));
    REQUIRE(tad.get_state()[14] == approximately(0.));
    REQUIRE(tad.get_state()[16] == approximately(0.));
    REQUIRE(tad.get_state()[18] == approximately(0.));
    REQUIRE(tad.get_state()[20] == approximately(v1));
    REQUIRE(tad.get_state()[22] == approximately(0.));

    REQUIRE(tad.get_state()[0 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[2 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[4 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[6 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[8 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[10 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[12 + 1] == approximately(a0));
    REQUIRE(tad.get_state()[14 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[16 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[18 + 1] == approximately(0.));
    REQUIRE(tad.get_state()[20 + 1] == approximately(v0));
    REQUIRE(tad.get_state()[22 + 1] == approximately(0.));
}

// Make sure the last timestep is properly recorded when using the
// step/propagate functions.
TEST_CASE("last h")
{
    auto [x, v] = make_vars("x", "v");

    // Scalar test.
    {
        auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, {0.05, 0.025}};

        REQUIRE(ta.get_last_h() == 0.);
        auto lh = ta.get_last_h();

        ta.step();

        REQUIRE(ta.get_last_h() != 0.);
        lh = ta.get_last_h();

        ta.step(1e-4);

        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.step_backward();

        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.propagate_for(1.23);
        REQUIRE(ta.get_last_h() != lh);
        lh = ta.get_last_h();

        ta.propagate_until(0.);
        REQUIRE(ta.get_last_h() != lh);
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        std::vector<double> init_state;
        for (auto i = 0u; i < batch_size; ++i) {
            init_state.push_back(0.05 + i / 100.);
        }
        for (auto i = 0u; i < batch_size; ++i) {
            init_state.push_back(0.025 + i / 1000.);
        }

        auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)}, init_state, batch_size};

        REQUIRE(std::all_of(ta.get_last_h().begin(), ta.get_last_h().end(), [](auto x) { return x == 0.; }));
        auto lh = ta.get_last_h();

        ta.step();
        REQUIRE(std::all_of(ta.get_last_h().begin(), ta.get_last_h().end(), [](auto x) { return x != 0.; }));
        lh = ta.get_last_h();

        ta.step(std::vector<double>(batch_size, 1e-4));
        for (auto i = 0u; i < batch_size; ++i) {
            REQUIRE(lh[i] != ta.get_last_h()[i]);
        }
    }
}

TEST_CASE("dense output")
{
    auto [x, v] = make_vars("x", "v");

    // Scalar test.
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                auto ta = taylor_adaptive<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                  {0.05, 0.025},
                                                  kw::high_accuracy = ha,
                                                  kw::compact_mode = cm,
                                                  kw::opt_level = opt_level};

                auto d_out = xt::adapt(ta.get_d_output());

                // Take a first step.
                ta.step(true);

                // The d_out at t = 0 must be the same
                // as the IC.
                ta.update_d_output(0);
                REQUIRE(d_out[0] == approximately(0.05, 100.));
                REQUIRE(d_out[1] == approximately(0.025, 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Store the state at the end of the first step.
                auto old_state1 = ta.get_state();

                // Take a second step.
                ta.step(true);

                // The d_out at the beginning of the timestep
                // must be equal to the state at the end of the
                // previous timestep.
                ta.update_d_output(ta.get_time() - ta.get_last_h());
                REQUIRE(d_out[0] == approximately(old_state1[0], 100.));
                REQUIRE(d_out[1] == approximately(old_state1[1], 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Store the state at the end of the second timestep.
                auto old_state2 = ta.get_state();

                // Take a third timestep.
                ta.step(true);

                // The d_out at the beginning of the timestep
                // must be equal to the state at the end of the
                // previous timestep.
                ta.update_d_output(ta.get_time() - ta.get_last_h());
                REQUIRE(d_out[0] == approximately(old_state2[0], 100.));
                REQUIRE(d_out[1] == approximately(old_state2[1], 100.));

                // The d_out at the end of the timestep must be
                // equal to the current state.
                ta.update_d_output(ta.get_time());
                REQUIRE(d_out[0] == approximately(ta.get_state()[0], 100.));
                REQUIRE(d_out[1] == approximately(ta.get_state()[1], 100.));

                // Do it a few more times.
                for (auto i = 0; i < 100; ++i) {
                    old_state2 = ta.get_state();

                    auto [oc, _] = ta.step(true);

                    REQUIRE(oc == taylor_outcome::success);

                    ta.update_d_output(ta.get_time() - ta.get_last_h());
                    REQUIRE(d_out[0] == approximately(old_state2[0], 1000.));
                    REQUIRE(d_out[1] == approximately(old_state2[1], 1000.));

                    ta.update_d_output(ta.get_time());
                    REQUIRE(d_out[0] == approximately(ta.get_state()[0], 1000.));
                    REQUIRE(d_out[1] == approximately(ta.get_state()[1], 1000.));
                }
            }
        }
    }

    // Batch test.
    for (auto batch_size : {1u, 4u, 23u}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            for (auto cm : {false, true}) {
                for (auto ha : {false, true}) {
                    std::vector<double> init_state;
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.05 + i / 100.);
                    }
                    for (auto i = 0u; i < batch_size; ++i) {
                        init_state.push_back(0.025 + i / 1000.);
                    }

                    auto ta = taylor_adaptive_batch<double>{{prime(x) = v, prime(v) = -9.8 * sin(x)},
                                                            init_state,
                                                            batch_size,
                                                            kw::high_accuracy = ha,
                                                            kw::compact_mode = cm,
                                                            kw::opt_level = opt_level};

                    auto sa = xt::adapt(ta.get_state_data(), {2u, batch_size});
                    auto isa = xt::adapt(init_state.data(), {2u, batch_size});
                    auto coa = xt::adapt(ta.get_d_output(), {2u, batch_size});

                    ta.step(true);

                    ta.update_d_output(std::vector<double>(batch_size, 0.));
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(isa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(isa(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    auto old_state1 = ta.get_state();
                    auto ost1a = xt::adapt(old_state1, {2u, batch_size});
                    auto old_time1 = ta.get_time();

                    ta.step(true);

                    ta.update_d_output(old_time1);
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(ost1a(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(ost1a(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    auto old_state2 = ta.get_state();
                    auto ost2a = xt::adapt(old_state2, {2u, batch_size});
                    auto old_time2 = ta.get_time();

                    ta.step(true);

                    ta.update_d_output(old_time2);
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(ost2a(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(ost2a(1u, i), 100.));
                    }

                    ta.update_d_output(ta.get_time());
                    for (auto i = 0u; i < batch_size; ++i) {
                        REQUIRE(coa(0u, i) == approximately(sa(0u, i), 100.));
                        REQUIRE(coa(1u, i) == approximately(sa(1u, i), 100.));
                    }

                    for (auto _ = 0; _ < 100; ++_) {
                        auto old_state = ta.get_state();
                        auto osta = xt::adapt(old_state, {2u, batch_size});
                        auto old_time = ta.get_time();

                        ta.step(true);

                        ta.update_d_output(old_time);
                        for (auto i = 0u; i < batch_size; ++i) {
                            REQUIRE(coa(0u, i) == approximately(osta(0u, i), 10000.));
                            REQUIRE(coa(1u, i) == approximately(osta(1u, i), 10000.));
                        }

                        ta.update_d_output(ta.get_time());
                        for (auto i = 0u; i < batch_size; ++i) {
                            REQUIRE(coa(0u, i) == approximately(sa(0u, i), 10000.));
                            REQUIRE(coa(1u, i) == approximately(sa(1u, i), 10000.));
                        }
                    }

                    using Catch::Matchers::Message;

                    REQUIRE_THROWS_MATCHES(
                        ta.update_d_output({}), std::invalid_argument,
                        Message("Invalid number of time coordinates specified for the dense output in a Taylor "
                                "integrator in batch "
                                "mode: the batch size is "
                                + std::to_string(batch_size) + ", but the number of time coordinates is 0"));
                    REQUIRE_THROWS_MATCHES(
                        ta.update_d_output(std::vector<double>(123u)), std::invalid_argument,
                        Message("Invalid number of time coordinates specified for the dense output in a Taylor "
                                "integrator in batch "
                                "mode: the batch size is "
                                + std::to_string(batch_size) + ", but the number of time coordinates is 123"));
                }
            }
        }
    }
}
