// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include <heyoka/llvm_state.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Test to verify the machinery of custom timestep.
// We will be checking that a custom timestep strategy
// matching Jorba's produces the same result as
// the taylor_adaptive integrators.
TEST_CASE("outer solar system custom step")
{
    using fp_t = double;

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    auto ic = std::vector{// Sun.
                          -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6, +6.69048890636161e-6,
                          -6.33922479583593e-6, -3.13202145590767e-9,
                          // Jupiter.
                          +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2, -5.59797969310664e-3,
                          +5.51815399480116e-3, -2.66711392865591e-6,
                          // Saturn.
                          +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1, -4.17354020307064e-3,
                          +3.99723751748116e-3, +1.67206320571441e-5,
                          // Uranus.
                          +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1, -3.25884806151064e-3,
                          +2.06438412905916e-3, -2.17699042180559e-5,
                          // Neptune.
                          -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1, -2.17471785045538e-4,
                          -3.11361111025884e-3, +3.58344705491441e-5,
                          // Pluto.
                          -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0, -1.76936577252484e-3,
                          -2.06720938381724e-3, +6.58091931493844e-4};

    for (auto cm : {false, true}) {
        for (auto ha : {false, true}) {
            // Init the adaptive integrator.
            taylor_adaptive<fp_t> ta{sys, ic, kw::high_accuracy = ha, kw::compact_mode = cm};

            // Add the custom timestep machinery.
            llvm_state s;
            const std::uint32_t order = ta.get_order();
            REQUIRE(order == 20u);
            taylor_add_custom_step<fp_t>(s, "cstep", sys, order, 1, ha, cm);
            s.compile();

            // Fetch the functions.
            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *)>(s.jit_lookup("cstep_jet"));
            auto uptr = reinterpret_cast<void (*)(fp_t *, const fp_t *)>(s.jit_lookup("cstep_updater"));

            // Prepare the buffer for the jet of derivatives and integration timesteps.
            std::vector<fp_t> jet_buffer(36u * (order + 1u));
            std::vector<fp_t> h_buffer(1);

            // Copy over the initial conditions.
            std::copy(ic.begin(), ic.end(), jet_buffer.begin());

            // Run a few timesteps with both integrators and compare the results.
            for (auto _ = 0; _ < 10; ++_) {
                // Do one timestep with ta.
                const auto [ta_res, ta_h] = ta.step();
                REQUIRE(ta_res == taylor_outcome::success);

                // Compute the jet of derivatives.
                jptr(jet_buffer.data(), nullptr);

                // Apply Jorba's timestep heuristic.
                // Step 1: norm infinity of the state vector.
                using std::abs;
                const auto max_abs_state = abs(*std::max_element(jet_buffer.data(), jet_buffer.data() + 36,
                                                                 [](auto a, auto b) { return abs(a) < abs(b); }));
                // NOTE: we know this from the specific initial conditions we are using.
                REQUIRE(max_abs_state > 1);

                // Step 2: norm infinity of the derivatives at order 'order - 1'.
                const auto max_abs_diff_om1
                    = abs(*std::max_element(jet_buffer.data() + 36u * (order - 1u), jet_buffer.data() + 36u * order,
                                            [](auto a, auto b) { return abs(a) < abs(b); }));

                // Step 3: norm infinity of the derivatives at order 'order'.
                const auto max_abs_diff_o
                    = abs(*std::max_element(jet_buffer.data() + 36u * order, jet_buffer.data() + 36u * (order + 1u),
                                            [](auto a, auto b) { return abs(a) < abs(b); }));

                // Step 4: compute the rho_m quantity.
                using std::pow;
                const auto rho_m = std::min(pow(max_abs_state / max_abs_diff_om1, 1. / (order - 1u)),
                                            pow(max_abs_state / max_abs_diff_o, 1. / order));

                // Step 5: determine the timestep from rho_m + safety factor.
                using std::exp;
                h_buffer[0] = rho_m / (exp(1.) * exp(1.)) * exp((-7 / 10.) / (order - 1u));

                // Compare the two timesteps.
                REQUIRE(ta_h == approximately(h_buffer[0]));

                // Now propagate the new state.
                uptr(jet_buffer.data(), h_buffer.data());

                // Compare the states.
                for (auto i = 0u; i < 36u; ++i) {
                    REQUIRE(ta.get_state()[i] == approximately(jet_buffer[i], 1e4));
                }
            }
        }
    }
}
