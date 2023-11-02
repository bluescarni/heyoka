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

#include <heyoka/llvm_state.hpp>
#include <heyoka/math/relu.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

template <typename T>
T cpp_relu(T x)
{
    return x > 0 ? x : T(0, x.get_prec());
}

template <typename T>
T cpp_relup(T x)
{
    return x > 0 ? T(1, x.get_prec()) : T(0, x.get_prec());
}

TEST_CASE("relu")
{
    using fp_t = mppp::real;

    auto [x, y] = make_vars("x", "y");

    for (auto prec : {30, 123}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto opt_level : {0u, 3u}) {
                    // Test with num/param/var.
                    {
                        llvm_state s{kw::opt_level = opt_level};

                        taylor_add_jet<fp_t>(s, "jet", {relu(x) + relup(y), x + y}, 2, 1, ha, cm, {}, false, prec);

                        s.compile();

                        if (opt_level == 0u && cm) {
                            REQUIRE(boost::contains(s.get_ir(), "heyoka.taylor_c_diff.relu.var"));
                            REQUIRE(boost::contains(s.get_ir(), "heyoka.taylor_c_diff.relup.var"));
                        }

                        auto jptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("jet"));

                        std::vector<fp_t> jet{fp_t{2, prec}, fp_t{-1, prec}};
                        jet.resize(6, fp_t(0, prec));

                        jptr(jet.data(), nullptr, nullptr);

                        REQUIRE(jet[0] == 2);
                        REQUIRE(jet[1] == -1);
                        REQUIRE(jet[2] == approximately(cpp_relu(jet[0]) + cpp_relup(jet[1])));
                        REQUIRE(jet[3] == 1);
                        REQUIRE(jet[4] == approximately(cpp_relup(jet[0]) * jet[2] / fp_t(2, prec)));
                        REQUIRE(jet[5] == approximately((jet[2] + jet[3]) / fp_t(2, prec)));
                    }
                }
            }
        }
    }
}
