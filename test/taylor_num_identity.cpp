// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <sstream>
#include <vector>

#include <heyoka/detail/num_identity.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;
namespace hy = heyoka;

TEST_CASE("taylor num_identity")
{
    using fp_t = double;

    auto x = "x"_var, y = "y"_var;

    for (auto cm : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            {
                auto ta = taylor_adaptive<fp_t>{{prime(x) = hy::detail::num_identity(42_dbl), prime(y) = x + y},
                                                {fp_t(2), fp_t(3)},
                                                kw::tol = 1,
                                                kw::compact_mode = cm,
                                                kw::opt_level = opt_level};

                ta.step(true);

                const auto jet = tc_to_jet(ta);

                REQUIRE(jet[0] == 2);
                REQUIRE(jet[1] == 3);
                REQUIRE(jet[2] == 42);
                REQUIRE(jet[3] == 5);
            }

            {
                auto ta = taylor_adaptive<fp_t>{{prime(x) = hy::detail::num_identity(42_dbl), prime(y) = x + y},
                                                {fp_t(2), fp_t(3)},
                                                kw::tol = .1,
                                                kw::compact_mode = cm,
                                                kw::opt_level = opt_level};

                ta.step(true);

                const auto jet = tc_to_jet(ta);

                REQUIRE(jet[0] == 2);
                REQUIRE(jet[1] == 3);
                REQUIRE(jet[2] == 42);
                REQUIRE(jet[3] == 5);
                REQUIRE(jet[4] == 0);
                REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + jet[2])));
                REQUIRE(jet[6] == 0);
                REQUIRE(jet[7] == approximately(fp_t{1} / 6 * (2 * jet[5] + 2 * jet[4])));
            }
        }
    }

    // Def ctor.
    detail::num_identity_impl nu;
    REQUIRE(nu.args() == std::vector{0_dbl});

    // s11n.
    std::stringstream ss;

    auto ex = hy::detail::num_identity(42_dbl) + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == hy::detail::num_identity(42_dbl) + x);
}
