// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/config.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/nbody.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

// Wrapper to ease the transition of old test code
// after the removal of sum_sq() from the public API.
auto sum_sq(const std::vector<expression> &args)
{
    std::vector<expression> new_args;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        new_args.push_back(arg * arg);
    }

    return sum(new_args);
}

TEST_CASE("add jet sv_funcs")
{
    auto [x, y] = make_vars("x", "y");

    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x + y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 3);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == 6);

                    REQUIRE(jet[3] == 12);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 6);

                    REQUIRE(jet[6] == -3);
                    REQUIRE(jet[7] == 6);
                    REQUIRE(jet[8] == 3);

                    REQUIRE(jet[9] == 2);
                    REQUIRE(jet[10] == -1);
                    REQUIRE(jet[11] == 1);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 4);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -6);
                    REQUIRE(jet[3] == 12);

                    REQUIRE(jet[4] == 12);
                    REQUIRE(jet[5] == -6);
                    REQUIRE(jet[6] == 12);
                    REQUIRE(jet[7] == -6);

                    REQUIRE(jet[8] == -3);
                    REQUIRE(jet[9] == 6);
                    REQUIRE(jet[10] == -3);
                    REQUIRE(jet[11] == 6);

                    REQUIRE(jet[12] == 2);
                    REQUIRE(jet[13] == -1);
                    REQUIRE(jet[14] == 2);
                    REQUIRE(jet[15] == -1);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, cm, ha, {x, y, x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -6);
                    REQUIRE(jet[3] == 12);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 12);

                    REQUIRE(jet[6] == 12);
                    REQUIRE(jet[7] == -6);
                    REQUIRE(jet[8] == 12);
                    REQUIRE(jet[9] == -6);
                    REQUIRE(jet[10] == 12);
                    REQUIRE(jet[11] == -6);

                    REQUIRE(jet[12] == -3);
                    REQUIRE(jet[13] == 6);
                    REQUIRE(jet[14] == -3);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -3);
                    REQUIRE(jet[17] == 6);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -1);
                    REQUIRE(jet[20] == 2);
                    REQUIRE(jet[21] == -1);
                    REQUIRE(jet[22] == 2);
                    REQUIRE(jet[23] == -1);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x, y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == -6);
                    REQUIRE(jet[5] == 12);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == 6);
                    REQUIRE(jet[11] == -18);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -6);
                    REQUIRE(jet[17] == 12);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == 2);
                    REQUIRE(jet[23] == -6);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x + 2. * y, y + 2. * x});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == 18);
                    REQUIRE(jet[5] == 0);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == -30);
                    REQUIRE(jet[11] == -6);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == 18);
                    REQUIRE(jet[17] == 0);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == -10);
                    REQUIRE(jet[23] == -2);
                }

                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = x + y, prime(y) = x - y}, 3, 1, cm, ha,
                                           {x - y, x + y, x - y, x + y});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6, 12};
                    jet.resize((3 + 1) * 6);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == 12);
                    REQUIRE(jet[2] == -18);
                    REQUIRE(jet[3] == 6);
                    REQUIRE(jet[4] == -18);
                    REQUIRE(jet[5] == 6);

                    REQUIRE(jet[6] == 6);
                    REQUIRE(jet[7] == -18);
                    REQUIRE(jet[8] == 24);
                    REQUIRE(jet[9] == -12);
                    REQUIRE(jet[10] == 24);
                    REQUIRE(jet[11] == -12);

                    REQUIRE(jet[12] == -6);
                    REQUIRE(jet[13] == 12);
                    REQUIRE(jet[14] == -18);
                    REQUIRE(jet[15] == 6);
                    REQUIRE(jet[16] == -18);
                    REQUIRE(jet[17] == 6);

                    REQUIRE(jet[18] == 2);
                    REQUIRE(jet[19] == -6);
                    REQUIRE(jet[20] == 8);
                    REQUIRE(jet[21] == -4);
                    REQUIRE(jet[22] == 8);
                    REQUIRE(jet[23] == -4);
                }

                // Run an example in which the last sv func is not at the end of the decomposition.
                {
                    llvm_state s{kw::opt_level = opt_level};

                    taylor_add_jet<double>(s, "jet", {prime(x) = cos(x) + x}, 3, 1, cm, ha, {cos(x)});

                    s.compile();

                    auto jptr
                        = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));

                    std::vector<double> jet{-6};
                    jet.resize((3 + 1) * 2);

                    jptr(jet.data(), nullptr, nullptr);

                    REQUIRE(jet[0] == -6);
                    REQUIRE(jet[1] == approximately(std::cos(-6)));

                    REQUIRE(jet[2] == approximately(std::cos(-6) - 6));
                    REQUIRE(jet[3] == approximately(-std::sin(-6) * jet[2]));

                    REQUIRE(jet[4] == approximately(.5 * (jet[2] - jet[2] * std::sin(jet[0]))));
                    REQUIRE(
                        jet[5]
                        == approximately(.5 * (-(jet[2] * jet[2]) * std::cos(jet[0]) - std::sin(jet[0]) * jet[4] * 2)));

                    REQUIRE(jet[6]
                            == approximately(
                                1 / 6.
                                * (2 * jet[4] - 2 * jet[4] * std::sin(jet[0]) - jet[2] * jet[2] * std::cos(jet[0]))));
                    REQUIRE(
                        jet[7]
                        == approximately(1 / 6.
                                         * (-2 * jet[2] * 2 * jet[4] * std::cos(jet[0])
                                            + jet[2] * jet[2] * jet[2] * std::sin(jet[0])
                                            - jet[2] * std::cos(jet[0]) * 2 * jet[4] - std::sin(jet[0]) * 6 * jet[6])));
                }
            }
        }
    }
}

TEST_CASE("nbody")
{
    const auto init_state
        = std::vector{// Sun.
                      -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6, +6.69048890636161e-6 * 365,
                      -6.33922479583593e-6 * 365, -3.13202145590767e-9 * 365,
                      // Jupiter.
                      +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2, -5.59797969310664e-3 * 365,
                      +5.51815399480116e-3 * 365, -2.66711392865591e-6 * 365,
                      // Saturn.
                      +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1, -4.17354020307064e-3 * 365,
                      +3.99723751748116e-3 * 365, +1.67206320571441e-5 * 365,
                      // Uranus.
                      +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1, -3.25884806151064e-3 * 365,
                      +2.06438412905916e-3 * 365, -2.17699042180559e-5 * 365,
                      // Neptune.
                      -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1, -2.17471785045538e-4 * 365,
                      -3.11361111025884e-3 * 365, +3.58344705491441e-5 * 365,
                      // Pluto.
                      -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0, -1.76936577252484e-3 * 365,
                      -2.06720938381724e-3 * 365, +6.58091931493844e-4 * 365};

    const auto masses = std::vector{1.00000597682, 1 / 1047.355, 1 / 3501.6, 1 / 22869., 1 / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895 * 365 * 365;

    auto sys = model::nbody(6, kw::masses = masses, kw::Gconst = G);

    // Create the state variables.
    std::vector<expression> x_vars, y_vars, z_vars, vx_vars, vy_vars, vz_vars;

    for (auto i = 0; i < 6; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
        y_vars.emplace_back(fmt::format("y_{}", i));
        z_vars.emplace_back(fmt::format("z_{}", i));

        vx_vars.emplace_back(fmt::format("vx_{}", i));
        vy_vars.emplace_back(fmt::format("vy_{}", i));
        vz_vars.emplace_back(fmt::format("vz_{}", i));
    }

    for (auto cm : {false, true}) {
        llvm_state s{kw::opt_level = 0u};

        taylor_add_jet<double>(s, "jet", sys, 3, 1, cm, false);

        std::vector<expression> sv_funcs;
        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = i + 1u; j < 6u; ++j) {
                auto diff_x = x_vars[j] - x_vars[i];
                auto diff_y = y_vars[j] - y_vars[i];
                auto diff_z = z_vars[j] - z_vars[i];

                sv_funcs.push_back(sum_sq({diff_x, diff_y, diff_z}));
            }
        }

        taylor_add_jet<double>(s, "jet_sv", sys, 3, 1, cm, false, sv_funcs);

        s.compile();

        auto jptr = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet"));
        auto jptr_sv = reinterpret_cast<void (*)(double *, const double *, const double *)>(s.jit_lookup("jet_sv"));

        auto state(init_state), state_sv(init_state);

        state.resize(36 * 4);
        state_sv.resize((36 + 15) * 4);

        jptr(state.data(), nullptr, nullptr);
        jptr_sv(state_sv.data(), nullptr, nullptr);

        auto sarr = xt::adapt(state, {4, 36});
        auto sarr_sv = xt::adapt(state_sv, {4, 36 + 15});

        // Verify all the derivatives of the state variables are identical
        // with and without the sv_funcs.
        for (auto i = 0; i <= 3; ++i) {
            REQUIRE(xt::all(xt::equal(xt::view(sarr, i, xt::all()), xt::view(sarr_sv, i, xt::range(0, 36)))));
        }

        // Verify that the order 0 of the sv funcs is correctly evaluated.
        auto sarr2 = xt::adapt(state, {4, 6, 6});
        int sv_idx = 0;
        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = i + 1u; j < 6u; ++j) {
                auto diff_x = sarr2(0, i, 0) - sarr2(0, j, 0);
                auto diff_y = sarr2(0, i, 1) - sarr2(0, j, 1);
                auto diff_z = sarr2(0, i, 2) - sarr2(0, j, 2);

                auto dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                REQUIRE(dist2 == approximately(sarr_sv(0, 36 + sv_idx)));
                ++sv_idx;
            }
        }

        // Verify that the order 1 of the sv funcs is correctly evaluated.
        sv_idx = 0;
        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = i + 1u; j < 6u; ++j) {
                auto diff_x = sarr2(0, i, 0) - sarr2(0, j, 0);
                auto diff_y = sarr2(0, i, 1) - sarr2(0, j, 1);
                auto diff_z = sarr2(0, i, 2) - sarr2(0, j, 2);

                auto diff_vx = sarr2(0, i, 3) - sarr2(0, j, 3);
                auto diff_vy = sarr2(0, i, 4) - sarr2(0, j, 4);
                auto diff_vz = sarr2(0, i, 5) - sarr2(0, j, 5);

                auto dist2_prime = 2 * diff_x * diff_vx + 2 * diff_y * diff_vy + 2 * diff_z * diff_vz;

                REQUIRE(dist2_prime == approximately(sarr_sv(1, 36 + sv_idx)));
                ++sv_idx;
            }
        }

        // Verify that the order 2 of the sv funcs is correctly evaluated.
        sv_idx = 0;
        for (auto i = 0u; i < 6u; ++i) {
            for (auto j = i + 1u; j < 6u; ++j) {
                auto diff_x = sarr2(0, i, 0) - sarr2(0, j, 0);
                auto diff_y = sarr2(0, i, 1) - sarr2(0, j, 1);
                auto diff_z = sarr2(0, i, 2) - sarr2(0, j, 2);

                auto diff_vx = sarr2(0, i, 3) - sarr2(0, j, 3);
                auto diff_vy = sarr2(0, i, 4) - sarr2(0, j, 4);
                auto diff_vz = sarr2(0, i, 5) - sarr2(0, j, 5);

                auto diff_ax = sarr2(1, i, 3) - sarr2(1, j, 3);
                auto diff_ay = sarr2(1, i, 4) - sarr2(1, j, 4);
                auto diff_az = sarr2(1, i, 5) - sarr2(1, j, 5);

                auto dist2_second = 2 * (diff_vx * diff_vx + diff_x * diff_ax)
                                    + 2 * (diff_vy * diff_vy + diff_y * diff_ay)
                                    + 2 * (diff_vz * diff_vz + diff_z * diff_az);

                REQUIRE(.5 * dist2_second == approximately(sarr_sv(2, 36 + sv_idx)));
                ++sv_idx;
            }
        }
    }
}

#if defined(HEYOKA_ARCH_PPC)

TEST_CASE("ppc long double")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    llvm_state s;

    REQUIRE_THROWS_MATCHES(
        (taylor_add_jet<long double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, false, false, {x + y})),
        not_implemented_error, Message("'long double' computations are not supported on PowerPC"));

    REQUIRE_THROWS_MATCHES(
        (taylor_add_jet<long double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 2, false, false, {x + y})),
        not_implemented_error, Message("'long double' computations are not supported on PowerPC"));
}

#endif

TEST_CASE("parallel non compact")
{
    using Catch::Matchers::Message;

    llvm_state s;

    auto [x, y] = make_vars("x", "y");

    REQUIRE_THROWS_MATCHES(taylor_add_jet<double>(s, "jet", {prime(x) = y, prime(y) = x}, 3, 1, false, false, {}, true),
                           std::invalid_argument,
                           Message("Parallel mode can only be enabled in conjunction with compact mode"));

    REQUIRE_THROWS_MATCHES(taylor_add_jet<double>(s, "jet", {y, x}, 3, 1, false, false, {}, true),
                           std::invalid_argument,
                           Message("Parallel mode can only be enabled in conjunction with compact mode"));
}
