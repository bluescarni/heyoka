// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <mp++/real.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

template <typename V>
bool check_close(const V &v1, const V &v2, mpfr_prec_t prec)
{
    using vt = typename V::value_type;

    for (decltype(v1.size()) i = 0; i < v1.size(); ++i) {
        if (!(v1[i] == approximately(v2[i], vt(10, prec)))) {
            return false;
        }
    }

    return true;
}

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const expression &x)
{
    return x * x;
}

// Check that the results of integration in parallel
// mode are consistent with serial mode.
TEST_CASE("parallel consistency")
{
    using fp_t = mppp::real;

    for (auto prec : {30, 123}) {
        auto masses = std::vector<fp_t>{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};
        for (auto &m : masses) {
            m.prec_round(prec);
        }

        const auto G = fp_t(0.01720209895 * 0.01720209895 * 365 * 365, prec);

        auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);

        auto ic = std::vector<fp_t>{// Sun.
                                    -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6,
                                    +6.69048890636161e-6 * 365, -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                                    // Jupiter.
                                    +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2,
                                    -5.59797969310664e-3 * 365, +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                                    // Saturn.
                                    +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1,
                                    -4.17354020307064e-3 * 365, +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                                    // Uranus.
                                    +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1,
                                    -3.25884806151064e-3 * 365, +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                                    // Neptune.
                                    -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1,
                                    -2.17471785045538e-4 * 365, -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                                    // Pluto.
                                    -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0,
                                    -1.76936577252484e-3 * 365, -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};
        for (auto &val : ic) {
            val.prec_round(prec);
        }

        // Create the events. We will assign to each body
        // the radius of jupiter to keep thing simple.
        const auto jradius = fp_t(0.000477895, prec);
        std::vector<nt_event<fp_t>> evs;
        auto cb = [](auto &, const auto &, int) { throw; };
        for (auto i = 0; i < 6; ++i) {
            auto xi = expression(fmt::format("x_{}", i));
            auto yi = expression(fmt::format("y_{}", i));
            auto zi = expression(fmt::format("z_{}", i));

            for (auto j = i + 1; j < 6; ++j) {
                auto xj = expression(fmt::format("x_{}", j));
                auto yj = expression(fmt::format("y_{}", j));
                auto zj = expression(fmt::format("z_{}", j));

                auto diff_x = xj - xi;
                auto diff_y = yj - yi;
                auto diff_z = zj - zi;

                auto ev_eq
                    = (square_wrapper(diff_x) + square_wrapper(diff_y) + square_wrapper(diff_z) - 4 * jradius * jradius)
                      * (1 / 100.);

                evs.emplace_back(std::move(ev_eq), cb);
            }
        }

        std::vector<fp_t> t_grid;
        for (auto i = 0; i < 20; ++i) {
            t_grid.emplace_back(i * .5, prec);
        }

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            taylor_adaptive<fp_t> ta_serial{sys, ic, kw::opt_level = opt_level, kw::compact_mode = true,
                                            kw::nt_events = evs};

            taylor_adaptive<fp_t> ta_parallel{sys,
                                              ic,
                                              kw::opt_level = opt_level,
                                              kw::compact_mode = true,
                                              kw::parallel_mode = true,
                                              kw::nt_events = evs};

            auto out_serial = std::get<4>(ta_serial.propagate_grid(t_grid));
            auto out_parallel = std::get<4>(ta_parallel.propagate_grid(t_grid));

            REQUIRE(check_close(out_serial, out_parallel, prec));
        }
    }
}

// Check the mechanism that sets the pars and time
// pointers in the global variable in parallel mode.
TEST_CASE("par time ptr")
{
    using fp_t = mppp::real;

    auto [x, v] = make_vars("x", "v");

    for (auto prec : {30, 123}) {
        std::vector<fp_t> t_grid;
        for (auto i = 0; i < 20; ++i) {
            t_grid.emplace_back(i * .5, prec);
        }

        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            taylor_adaptive<fp_t> ta_serial{{prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)},
                                            {fp_t(0., prec), fp_t(1.85, prec)},
                                            kw::opt_level = opt_level,
                                            kw::compact_mode = true,
                                            kw::pars = {fp_t(.1, prec)}};

            taylor_adaptive<fp_t> ta_parallel{{prime(x) = v, prime(v) = cos(heyoka::time) - par[0] * v - sin(x)},
                                              {fp_t(0., prec), fp_t(1.85, prec)},
                                              kw::opt_level = opt_level,
                                              kw::compact_mode = true,
                                              kw::pars = {fp_t(.1, prec)},
                                              kw::parallel_mode = true};

            auto out_serial = std::get<4>(ta_serial.propagate_grid(t_grid));
            auto out_parallel = std::get<4>(ta_parallel.propagate_grid(t_grid));

            REQUIRE(check_close(out_serial, out_parallel, prec));
        }
    }
}
