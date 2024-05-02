// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>

#include <boost/algorithm/string/predicate.hpp>

#include <mp++/real.hpp>

#include <heyoka/kw.hpp>
#include <heyoka/math/relu.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

template <typename T>
T cpp_relu(T x, T slope = T(0))
{
    return x > 0 ? x : x * T(slope, x.get_prec());
}

template <typename T>
T cpp_relup(T x, T slope = T(0))
{
    return x > 0 ? T(1, x.get_prec()) : T(slope, x.get_prec());
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
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = relu(x) + relup(y), prime(y) = x + y},
                                                        {fp_t{2, prec}, fp_t{-1, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        if (opt_level == 0u && cm) {
                            REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.relu.var"));
                            REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.relup.var"));
                        }

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

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

TEST_CASE("relu leaky")
{
    using fp_t = mppp::real;

    auto [x, y] = make_vars("x", "y");

    for (auto prec : {30, 123}) {
        for (auto cm : {false, true}) {
            for (auto ha : {false, true}) {
                for (auto opt_level : {0u, 3u}) {
                    // Test with num/param/var.
                    {
                        auto ta = taylor_adaptive<fp_t>{{prime(x) = relu(x, 0.01) + relup(y, 0.02), prime(y) = x + y},
                                                        {fp_t{-2, prec}, fp_t{-1, prec}},
                                                        kw::tol = 1,
                                                        kw::high_accuracy = ha,
                                                        kw::compact_mode = cm,
                                                        kw::opt_level = opt_level};

                        if (opt_level == 0u && cm) {
                            REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.relu_0x"));
                            REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.relup_0x"));
                            REQUIRE(boost::contains(ta.get_llvm_state().get_ir(), ".var"));
                        }

                        ta.step(true);

                        const auto jet = tc_to_jet(ta);

                        REQUIRE(jet[0] == -2);
                        REQUIRE(jet[1] == -1);
                        REQUIRE(
                            jet[2]
                            == approximately(cpp_relu(jet[0], mppp::real(0.01)) + cpp_relup(jet[1], mppp::real(0.02))));
                        REQUIRE(jet[3] == jet[0] + jet[1]);
                        REQUIRE(jet[4] == approximately(cpp_relup(jet[0], mppp::real(0.01)) * jet[2] / fp_t(2, prec)));
                        REQUIRE(jet[5] == approximately((jet[2] + jet[3]) / fp_t(2, prec)));
                    }
                }
            }
        }
    }
}
