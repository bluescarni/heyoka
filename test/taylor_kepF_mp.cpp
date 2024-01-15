// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <mp++/real.hpp>

#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/kepF.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("kepF")
{
    using fp_t = mppp::real;

    auto [x, y, z] = make_vars("x", "y", "z");

    for (auto prec : {30, 123}) {
        // cfunc for testing purposes.
        llvm_state s_cfunc;

        add_cfunc<fp_t>(s_cfunc, "cfunc", {kepF(x, y, z)}, {x, y, z}, kw::prec = prec);

        s_cfunc.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s_cfunc.jit_lookup("cfunc"));

        auto kepF_num = [cf_ptr, prec](fp_t h, fp_t k, fp_t lam) {
            const fp_t cf_in[3] = {h, k, lam};
            fp_t cf_out(0, prec);

            cf_ptr(&cf_out, cf_in, nullptr, nullptr);

            return cf_out;
        };

        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto opt_level : {0u, 3u}) {
                    // Test with num/param/var.
                    {
                        llvm_state s{kw::opt_level = opt_level};

                        taylor_add_jet<fp_t>(s, "jet", {kepF(fp_t(.1, prec), par[0], x), x + y}, 2, 1, ha, cm, {},
                                             false, prec);

                        s.compile();

                        if (opt_level == 0u && cm) {
                            REQUIRE(boost::contains(s.get_ir(), "heyoka.taylor_c_diff.kepF.num_par_var"));
                        }

                        auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                        std::vector<fp_t> jet{fp_t{2, prec}, fp_t{-1, prec}}, pars{fp_t(.1, prec)};
                        jet.resize(6, fp_t(0, prec));

                        jptr(jet.data(), pars.data(), nullptr);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == -1);
                        REQUIRE(jet[2] == approximately(kepF_num(fp_t(.1, prec), pars[0], jet[0])));
                        REQUIRE(jet[3] == 1);
                        auto den0 = fp_t(1, prec) - fp_t(.1, prec) * sin(jet[2]) - pars[0] * cos(jet[2]);
                        REQUIRE(jet[4] == approximately((jet[2] / den0) / fp_t(2, prec)));
                        REQUIRE(jet[5] == (jet[2] + jet[3]) / fp_t(2, prec));
                    }
                }
            }
        }
    }
}
