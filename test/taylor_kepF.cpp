// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// Issue in the decomposition when h/k = 0.
TEST_CASE("taylor kepF decompose bug 00")
{
    llvm_state s;

    auto lam = make_vars("lam");

    taylor_add_jet<double>(s, "jet", {kepF(0_dbl, 0_dbl, lam)}, 1, 1, false, false);
}

TEST_CASE("taylor kepF")
{
    using std::cos;
    using std::sin;

    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y, z] = make_vars("x", "y", "z");

        // cfunc for testing purposes.
        llvm_state s_cfunc;

        add_cfunc<fp_t>(s_cfunc, "cfunc", {kepF(x, y, z)});

        s_cfunc.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s_cfunc.jit_lookup("cfunc"));

        auto kepF_num = [cf_ptr](fp_t h, fp_t k, fp_t lam) {
            const fp_t cf_in[3] = {h, k, lam};
            fp_t cf_out(0);

            cf_ptr(&cf_out, cf_in, nullptr, nullptr);

            return cf_out;
        };

        // Number-number-number test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1), par[0], .3_dbl), x + y}, 2, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(fp_t(.1), par[0], .3_dbl), x + y}, 2, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.num_par_num"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}}, pars{fp_t(.1), fp_t(.2)};
            jet.resize(12);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(fp_t(.1), pars[0], fp_t(.3))));
            REQUIRE(jet[5] == approximately(kepF_num(fp_t(.1), pars[1], fp_t(.3))));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            REQUIRE(jet[8] == 0);
            REQUIRE(jet[9] == 0);

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);
        }

        // Number-number-var test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1), par[0], x), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(fp_t(.1), par[0], x), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.num_par_var"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{5}}, pars{fp_t(.1), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -1);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(fp_t(.1), pars[0], jet[0])));
            REQUIRE(jet[5] == approximately(kepF_num(fp_t(.1), pars[1], jet[1])));

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == 4);

            auto den0 = 1 - fp_t(.1) * sin(jet[4]) - pars[0] * cos(jet[4]);
            auto den1 = 1 - fp_t(.1) * sin(jet[5]) - pars[1] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[4] / den0) / 2));
            REQUIRE(jet[9] == approximately((jet[5] / den1) / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto tmp0 = -fp_t(.1) * cos(jet[4]) * jet[8] * 2 + pars[0] * sin(jet[4]) * jet[8] * 2;
            auto tmp1 = -fp_t(.1) * cos(jet[5]) * jet[9] * 2 + pars[1] * sin(jet[5]) * jet[9] * 2;

            REQUIRE(jet[12] == approximately(((jet[8] * 2 * den0 - jet[4] * tmp0) / (den0 * den0)) / 6));
            REQUIRE(jet[13] == approximately(((jet[9] * 2 * den1 - jet[5] * tmp1) / (den1 * den1)) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Number-var-number test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1), x, par[0]), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(fp_t(.1), x, par[0]), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.num_var_par"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{3}, fp_t{5}}, pars{fp_t(.2), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(fp_t(.1), jet[0], pars[0])));
            REQUIRE(jet[5] == approximately(kepF_num(fp_t(.1), jet[1], pars[1])));

            REQUIRE(jet[6] == 3.5);
            REQUIRE(jet[7] == 4.875);

            auto den0 = 1 - fp_t(.1) * sin(jet[4]) - jet[0] * cos(jet[4]);
            auto den1 = 1 - fp_t(.1) * sin(jet[5]) - jet[1] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[4] * sin(jet[4]) / den0) / 2));
            REQUIRE(jet[9] == approximately((jet[5] * sin(jet[5]) / den1) / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto Fp0 = jet[8] * 2;
            auto Fp1 = jet[9] * 2;

            auto tmp0 = -fp_t(.1) * cos(jet[4]) * Fp0 - jet[4] * cos(jet[4]) + jet[0] * sin(jet[4]) * Fp0;
            auto tmp1 = -fp_t(.1) * cos(jet[5]) * Fp1 - jet[5] * cos(jet[5]) + jet[1] * sin(jet[5]) * Fp1;

            REQUIRE(jet[12]
                    == approximately(
                        ((jet[8] * 2 * sin(jet[4]) + jet[4] * Fp0 * cos(jet[4])) * den0 - jet[4] * sin(jet[4]) * tmp0)
                        / (den0 * den0) / 6));
            REQUIRE(jet[13]
                    == approximately(
                        ((jet[9] * 2 * sin(jet[5]) + jet[5] * Fp1 * cos(jet[5])) * den1 - jet[5] * sin(jet[5]) * tmp1)
                        / (den1 * den1) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Var-number-number test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(x, fp_t(.1), par[0]), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(x, fp_t(.1), par[0]), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.var_num_par"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{3}, fp_t{5}}, pars{fp_t(.2), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(jet[0], fp_t(.1), pars[0])));
            REQUIRE(jet[5] == approximately(kepF_num(jet[1], fp_t(.1), pars[1])));

            REQUIRE(jet[6] == 3.5);
            REQUIRE(jet[7] == 4.875);

            auto den0 = 1 - jet[0] * sin(jet[4]) - fp_t(.1) * cos(jet[4]);
            auto den1 = 1 - jet[1] * sin(jet[5]) - fp_t(.1) * cos(jet[5]);

            REQUIRE(jet[8] == approximately(-(jet[4] * cos(jet[4]) / den0) / 2));
            REQUIRE(jet[9] == approximately(-(jet[5] * cos(jet[5]) / den1) / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto Fp0 = jet[8] * 2;
            auto Fp1 = jet[9] * 2;

            auto tmp0 = -jet[4] * sin(jet[4]) - jet[0] * cos(jet[4]) * Fp0 + fp_t(.1) * sin(jet[4]) * Fp0;
            auto tmp1 = -jet[5] * sin(jet[5]) - jet[1] * cos(jet[5]) * Fp1 + fp_t(.1) * sin(jet[5]) * Fp1;

            REQUIRE(jet[12]
                    == approximately(
                        ((jet[4] * Fp0 * sin(jet[4]) - 2 * jet[8] * cos(jet[4])) * den0 + jet[4] * cos(jet[4]) * tmp0)
                        / (den0 * den0) / 6));
            REQUIRE(jet[13]
                    == approximately(
                        ((jet[5] * Fp1 * sin(jet[5]) - 2 * jet[9] * cos(jet[5])) * den1 + jet[5] * cos(jet[5]) * tmp1)
                        / (den1 * den1) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Number-var-var test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(par[0], x, y), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(par[0], x, y), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.par_var_var"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{3}, fp_t{5}}, pars{fp_t(.2), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(pars[0], jet[0], jet[2])));
            REQUIRE(jet[5] == approximately(kepF_num(pars[1], jet[1], jet[3])));

            REQUIRE(jet[6] == 3.5);
            REQUIRE(jet[7] == 4.875);

            auto den0 = 1 - pars[0] * sin(jet[4]) - jet[0] * cos(jet[4]);
            auto den1 = 1 - pars[1] * sin(jet[5]) - jet[1] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[4] * sin(jet[4]) + jet[6]) / den0 / 2));
            REQUIRE(jet[9] == approximately((jet[5] * sin(jet[5]) + jet[7]) / den1 / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto Fp0 = jet[8] * 2;
            auto Fp1 = jet[9] * 2;

            auto tmp0 = -pars[0] * cos(jet[4]) * Fp0 - jet[4] * cos(jet[4]) + jet[0] * sin(jet[4]) * Fp0;
            auto tmp1 = -pars[1] * cos(jet[5]) * Fp1 - jet[5] * cos(jet[5]) + jet[1] * sin(jet[5]) * Fp1;

            REQUIRE(jet[12]
                    == approximately(((Fp0 * sin(jet[4]) + jet[4] * cos(jet[4]) * Fp0 + 2 * jet[10]) * den0
                                      - (jet[4] * sin(jet[4]) + jet[6]) * tmp0)
                                     / (den0 * den0) / 6));
            REQUIRE(jet[13]
                    == approximately(((Fp1 * sin(jet[5]) + jet[5] * cos(jet[5]) * Fp1 + 2 * jet[11]) * den1
                                      - (jet[5] * sin(jet[5]) + jet[7]) * tmp1)
                                     / (den1 * den1) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Var-number-var test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(x, par[0], y), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(x, par[0], y), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.var_par_var"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{3}, fp_t{5}}, pars{fp_t(.2), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == 5);

            REQUIRE(jet[4] == approximately(kepF_num(jet[0], pars[0], jet[2])));
            REQUIRE(jet[5] == approximately(kepF_num(jet[1], pars[1], jet[3])));

            REQUIRE(jet[6] == 3.5);
            REQUIRE(jet[7] == 4.875);

            auto den0 = 1 - jet[0] * sin(jet[4]) - pars[0] * cos(jet[4]);
            auto den1 = 1 - jet[1] * sin(jet[5]) - pars[1] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[6] - jet[4] * cos(jet[4])) / den0 / 2));
            REQUIRE(jet[9] == approximately((jet[7] - jet[5] * cos(jet[5])) / den1 / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto Fp0 = jet[8] * 2;
            auto Fp1 = jet[9] * 2;

            auto tmp0 = -jet[4] * sin(jet[4]) - jet[0] * cos(jet[4]) * Fp0 + pars[0] * sin(jet[4]) * Fp0;
            auto tmp1 = -jet[5] * sin(jet[5]) - jet[1] * cos(jet[5]) * Fp1 + pars[1] * sin(jet[5]) * Fp1;

            REQUIRE(jet[12]
                    == approximately(((2 * jet[10] - Fp0 * cos(jet[4]) + jet[4] * sin(jet[4]) * Fp0) * den0
                                      - (jet[6] - jet[4] * cos(jet[4])) * tmp0)
                                     / (den0 * den0) / 6));
            REQUIRE(jet[13]
                    == approximately(((2 * jet[11] - Fp1 * cos(jet[5]) + jet[5] * sin(jet[5]) * Fp1) * den1
                                      - (jet[7] - jet[5] * cos(jet[5])) * tmp1)
                                     / (den1 * den1) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Var-var-number test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(x, y, par[0]), x + y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(x, y, par[0]), x + y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.var_var_par"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{0.1875}, fp_t{-0.3125}}, pars{fp_t(.2), fp_t(.2)};
            jet.resize(16);

            jptr(jet.data(), pars.data(), nullptr);

            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 0.1875);
            REQUIRE(jet[3] == -0.3125);

            REQUIRE(jet[4] == approximately(kepF_num(jet[0], jet[2], pars[0])));
            REQUIRE(jet[5] == approximately(kepF_num(jet[1], jet[3], pars[1])));

            REQUIRE(jet[6] == jet[0] + jet[2]);
            REQUIRE(jet[7] == jet[1] + jet[3]);

            auto den0 = 1 - jet[0] * sin(jet[4]) - jet[2] * cos(jet[4]);
            auto den1 = 1 - jet[1] * sin(jet[5]) - jet[3] * cos(jet[5]);

            REQUIRE(jet[8] == approximately((jet[6] * sin(jet[4]) - jet[4] * cos(jet[4])) / den0 / 2));
            REQUIRE(jet[9] == approximately((jet[7] * sin(jet[5]) - jet[5] * cos(jet[5])) / den1 / 2));

            REQUIRE(jet[10] == (jet[4] + jet[6]) / 2);
            REQUIRE(jet[11] == (jet[5] + jet[7]) / 2);

            auto Fp0 = jet[8] * 2;
            auto Fp1 = jet[9] * 2;

            auto tmp0 = -jet[4] * sin(jet[4]) - jet[0] * cos(jet[4]) * Fp0 - jet[6] * cos(jet[4])
                        + jet[2] * sin(jet[4]) * Fp0;
            auto tmp1 = -jet[5] * sin(jet[5]) - jet[1] * cos(jet[5]) * Fp1 - jet[7] * cos(jet[5])
                        + jet[3] * sin(jet[5]) * Fp1;

            REQUIRE(jet[12]
                    == approximately(((2 * jet[10] * sin(jet[4]) + jet[6] * cos(jet[4]) * Fp0 - Fp0 * cos(jet[4])
                                       + jet[4] * sin(jet[4]) * Fp0)
                                          * den0
                                      - (jet[6] * sin(jet[4]) - jet[4] * cos(jet[4])) * tmp0)
                                     / (den0 * den0) / 6));
            REQUIRE(jet[13]
                    == approximately(((2 * jet[11] * sin(jet[5]) + jet[7] * cos(jet[5]) * Fp1 - Fp1 * cos(jet[5])
                                       + jet[5] * sin(jet[5]) * Fp1)
                                          * den1
                                      - (jet[7] * sin(jet[5]) - jet[5] * cos(jet[5])) * tmp1)
                                     / (den1 * den1) / 6));

            REQUIRE(jet[14] == approximately((jet[8] * 2 + jet[10] * 2) / 6));
            REQUIRE(jet[15] == approximately((jet[9] * 2 + jet[11] * 2) / 6));
        }

        // Var-var-var test.
        {
            llvm_state s{kw::opt_level = opt_level};

            taylor_add_jet<fp_t>(s, "jet", {kepF(x, y, z), x + y, y}, 3, 2, high_accuracy, compact_mode);
            taylor_add_jet<fp_t>(s, "jet2", {kepF(x, y, z), x + y, y}, 3, 2, high_accuracy, compact_mode);

            s.compile();

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "@heyoka.taylor_c_diff.kepF.var_var_var"));
            }

            auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

            std::vector<fp_t> jet{fp_t{.5}, fp_t{-.125}, fp_t{0.1875}, fp_t{-0.3125}, fp_t{1}, fp_t{2}};
            jet.resize(24);

            jptr(jet.data(), nullptr, nullptr);

            // Order 0.
            REQUIRE(jet[0] == .5);
            REQUIRE(jet[1] == -.125);

            REQUIRE(jet[2] == 0.1875);
            REQUIRE(jet[3] == -0.3125);

            REQUIRE(jet[4] == 1);
            REQUIRE(jet[5] == 2);

            // Order 1.
            REQUIRE(jet[6] == approximately(kepF_num(jet[0], jet[2], jet[4])));
            REQUIRE(jet[7] == approximately(kepF_num(jet[1], jet[3], jet[5])));

            REQUIRE(jet[8] == jet[0] + jet[2]);
            REQUIRE(jet[9] == jet[1] + jet[3]);

            REQUIRE(jet[10] == jet[2]);
            REQUIRE(jet[11] == jet[3]);

            auto den0 = 1 - jet[0] * sin(jet[6]) - jet[2] * cos(jet[6]);
            auto den1 = 1 - jet[1] * sin(jet[7]) - jet[3] * cos(jet[7]);

            // Order 2.
            REQUIRE(jet[12] == approximately((jet[8] * sin(jet[6]) - jet[6] * cos(jet[6]) + jet[10]) / den0 / 2));
            REQUIRE(jet[13] == approximately((jet[9] * sin(jet[7]) - jet[7] * cos(jet[7]) + jet[11]) / den1 / 2));

            REQUIRE(jet[14] == (jet[6] + jet[8]) / 2);
            REQUIRE(jet[15] == (jet[7] + jet[9]) / 2);

            REQUIRE(jet[16] == jet[8] / 2);
            REQUIRE(jet[17] == jet[9] / 2);

            auto Fp0 = jet[12] * 2;
            auto Fp1 = jet[13] * 2;

            auto tmp0 = -jet[6] * sin(jet[6]) - jet[0] * cos(jet[6]) * Fp0 - jet[8] * cos(jet[6])
                        + jet[2] * sin(jet[6]) * Fp0;
            auto tmp1 = -jet[7] * sin(jet[7]) - jet[1] * cos(jet[7]) * Fp1 - jet[9] * cos(jet[7])
                        + jet[3] * sin(jet[7]) * Fp1;

            // Order 3.
            REQUIRE(jet[18]
                    == approximately(((2 * jet[14] * sin(jet[6]) + jet[8] * cos(jet[6]) * Fp0 - Fp0 * cos(jet[6])
                                       + jet[6] * sin(jet[6]) * Fp0 + 2 * jet[16])
                                          * den0
                                      - (jet[8] * sin(jet[6]) - jet[6] * cos(jet[6]) + jet[10]) * tmp0)
                                     / (den0 * den0) / 6));
            REQUIRE(jet[19]
                    == approximately(((2 * jet[15] * sin(jet[7]) + jet[9] * cos(jet[7]) * Fp1 - Fp1 * cos(jet[7])
                                       + jet[7] * sin(jet[7]) * Fp1 + 2 * jet[17])
                                          * den1
                                      - (jet[9] * sin(jet[7]) - jet[7] * cos(jet[7]) + jet[11]) * tmp1)
                                     / (den1 * den1) / 6));

            REQUIRE(jet[20] == approximately((jet[12] * 2 + jet[14] * 2) / 6));
            REQUIRE(jet[21] == approximately((jet[13] * 2 + jet[15] * 2) / 6));

            REQUIRE(jet[22] == approximately((jet[14] * 2) / 6));
            REQUIRE(jet[23] == approximately((jet[15] * 2) / 6));
        }
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}
