// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <functional>
#include <sstream>
#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/math/tpoly.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("tpoly basics")
{
    using Catch::Matchers::Message;

    {
        detail::tpoly_impl tp;

        REQUIRE(tp.m_b_idx == 0u);
        REQUIRE(tp.m_e_idx == 1u);

        REQUIRE(tp.args()[0] == par[0]);
        REQUIRE(tp.args()[1] == par[1]);
    }

    {
        detail::tpoly_impl tp(par[10], par[12]);

        REQUIRE(tp.m_b_idx == 10u);
        REQUIRE(tp.m_e_idx == 12u);

        REQUIRE(tp.args()[0] == par[10]);
        REQUIRE(tp.args()[1] == par[12]);
    }

    // Verify equality/hashing.
    REQUIRE(tpoly(par[0], par[10]) == tpoly(par[0], par[10]));
    REQUIRE(tpoly(par[0], par[10]) != tpoly(par[10], par[20]));
    REQUIRE(std::hash<expression>{}(tpoly(par[0], par[10])) == std::hash<expression>{}(tpoly(par[0], par[10])));
    REQUIRE(std::hash<expression>{}(tpoly(par[0], par[10])) != std::hash<expression>{}(tpoly(par[10], par[20])));

    REQUIRE_THROWS_MATCHES(tpoly(par[10], par[9]), std::invalid_argument,
                           Message("Cannot construct a time polynomial from param indices 10 and 9: the first index is "
                                   "not less than the second"));
    REQUIRE_THROWS_MATCHES(
        tpoly(par[11], par[11]), std::invalid_argument,
        Message("Cannot construct a time polynomial from param indices 11 and 11: the first index is "
                "not less than the second"));

    REQUIRE_THROWS_MATCHES(tpoly("x"_var, par[11]), std::invalid_argument,
                           Message("Cannot construct a time polynomial from a non-param argument"));
    REQUIRE_THROWS_MATCHES(tpoly(par[11], "x"_var), std::invalid_argument,
                           Message("Cannot construct a time polynomial from a non-param argument"));
}

TEST_CASE("tpoly s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = tpoly(par[10], par[20]) + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == tpoly(par[10], par[20]) + x);
}
