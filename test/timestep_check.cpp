// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Tests to double-check the timestep deduction
// logic in the LLVM code.
TEST_CASE("scalar")
{
    using std::abs;
    using std::exp;
    using std::pow;

    using fp_t = double;

    auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895;

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);

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

            const auto order = ta.get_order();

            for (auto _ = 0; _ < 10; ++_) {
                // Take a step forward writing the Taylor coefficients.
                ta.step(true);

                // Determine the norm infinity of the state vector and the
                // normalised derivatives at the last 2 orders.
                fp_t x_inf = 0, do_inf = 0, dom1_inf = 0;

                for (auto i = 0u; i < 36u; ++i) {
                    x_inf = std::max(x_inf, abs(ta.get_tc()[i * (order + 1u)]));
                    do_inf = std::max(do_inf, abs(ta.get_tc()[i * (order + 1u) + order]));
                    dom1_inf = std::max(dom1_inf, abs(ta.get_tc()[i * (order + 1u) + order - 1u]));
                }

                auto rho_o = pow(x_inf / do_inf, fp_t{1} / order);
                auto rho_om1 = pow(x_inf / dom1_inf, fp_t{1} / (order - 1u));

                auto rho_m = std::min(rho_o, rho_om1);

                auto h = rho_m / (exp(fp_t(1)) * exp(fp_t(1))) * exp((-fp_t(7) / 10) / (order - 1u));

                REQUIRE(h == approximately(ta.get_last_h()));
            }
        }
    }
}
