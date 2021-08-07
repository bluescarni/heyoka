// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <variant>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/atan2.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("atan2 def ctor")
{
    detail::atan2_impl a;

    REQUIRE(a.args().size() == 2u);
    REQUIRE(a.args()[0] == 0_dbl);
    REQUIRE(a.args()[1] == 1_dbl);
}

TEST_CASE("atan2 diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(atan2(y, x), "x") == (-y) / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "y") == x / (x * x + y * y));
    REQUIRE(diff(atan2(y, x), "z") == 0_dbl);
    REQUIRE(diff(atan2(x * y, y / x), "x")
            == (y / x * y - (x * y) * (-y / (x * x))) / ((y / x) * (y / x) + (x * y) * (x * y)));
}

TEST_CASE("atan2 decompose")
{
    auto [u0, u1] = make_vars("u_0", "u_1");

    {
        taylor_dc_t dec;
        dec.emplace_back("y"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("x"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(atan2(u0, u1), dec);

        REQUIRE(dec.size() == 6u);

        REQUIRE(dec[2].first == u1 * u1);
        REQUIRE(dec[2].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[3].first == u0 * u0);
        REQUIRE(dec[3].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[4].first == "u_2"_var + "u_3"_var);
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[5].first == atan2(u0, u1));
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{4});
    }

    {
        taylor_dc_t dec;
        dec.emplace_back("y"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("x"_var, std::vector<std::uint32_t>{});
        taylor_decompose_in_place(atan2(u0 + u1, u1 - u0), dec);

        REQUIRE(dec.size() == 8u);

        REQUIRE(dec[2].first == u0 + u1);
        REQUIRE(dec[2].second.empty());

        REQUIRE(dec[3].first == u1 - u0);
        REQUIRE(dec[3].second.empty());

        REQUIRE(dec[4].first == "u_3"_var * "u_3"_var);
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[5].first == "u_2"_var * "u_2"_var);
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[6].first == "u_4"_var + "u_5"_var);
        REQUIRE(dec[6].second == std::vector<std::uint32_t>{});

        REQUIRE(dec[7].first == atan2("u_2"_var, "u_3"_var));
        REQUIRE(dec[7].second == std::vector<std::uint32_t>{6});
    }
}

TEST_CASE("atan2 overloads")
{
    auto k = atan2("x"_var, 1.1);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1});

    k = atan2("x"_var, 1.1l);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = atan2("x"_var, mppp::real128{"1.1"});
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{mppp::real128{"1.1"}});
#endif

    k = atan2(1.1, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1});

    k = atan2(1.1l, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = atan2(mppp::real128{"1.1"}, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{mppp::real128{"1.1"}});
#endif
}

TEST_CASE("atan2 cse")
{
    auto x = "x"_var, y = "y"_var;

    llvm_state s;

    auto dc = taylor_add_jet<double>(s, "jet", {atan2(y, x) + (x * x + y * y), x}, 1, 1, false, false);

    REQUIRE(dc.size() == 9u);
}

TEST_CASE("atan2 s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = atan2(x, y);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == atan2(x, y));
}
