// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <sstream>
#include <variant>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("fix s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = fix(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == fix(x));
}

TEST_CASE("fix stream")
{
    auto [x, y] = make_vars("x", "y");

    std::stringstream ss;

    ss << fix(sum({x, y}));

    REQUIRE(ss.str() == "{(x + y)}");
}

TEST_CASE("fix diff var")
{
    auto [x, y] = make_vars("x", "y");

    auto df = diff(fix(sin(sum({pow(x, 2_dbl), y}))), x);

    REQUIRE(detail::is_fixed(df));
    REQUIRE(std::get<func>(df.value()).args().size() == 1u);
    REQUIRE(std::get<func>(df.value()).args()[0] == diff(sin(sum({pow(x, 2_dbl), y})), x));
}

TEST_CASE("fix diff par")
{
    auto [x, y] = make_vars("x", "y");

    auto df = diff(fix(sin(sum({pow(par[0], 2_dbl), y}))), x);

    REQUIRE(detail::is_fixed(df));
    REQUIRE(std::get<func>(df.value()).args().size() == 1u);
    REQUIRE(std::get<func>(df.value()).args()[0] == diff(sin(sum({pow(par[0], 2_dbl), y})), x));
}

TEST_CASE("fix")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(!detail::is_fixed(x));
    REQUIRE(!detail::is_fixed(sum({x})));
    REQUIRE(detail::is_fixed(fix(x)));
    REQUIRE(detail::is_fixed(fix(sum({x}))));

    REQUIRE(std::get<func>(sum({1_dbl, sum({x, y})}).value()).args().size() == 3u);
    REQUIRE(std::get<func>(sum({1_dbl, fix(sum({x, y}))}).value()).args().size() == 2u);

    REQUIRE(std::get<func>(prod({pow(x, 2_dbl), x}).value()).extract<detail::pow_impl>() != nullptr);
    REQUIRE(std::get<func>(prod({fix(pow(x, 2_dbl)), x}).value()).args().size() == 2u);
    REQUIRE(std::get<func>(prod({fix(pow(x, 2_dbl)), x}).value()).extract<detail::prod_impl>() != nullptr);

    REQUIRE(cos(fix(1_dbl)) != cos(1_dbl));
}

TEST_CASE("unfix")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(unfix(fix(x)) == x);
    REQUIRE(unfix(x + y) == x + y);
    REQUIRE(unfix(cos(x + y)) == cos(x + y));

    auto tmp = x + y;

    auto ret = unfix({cos(tmp), sin(tmp)});

    REQUIRE(ret[0] == cos(x + y));
    REQUIRE(ret[1] == sin(x + y));

    tmp = fix(x + y);

    ret = unfix({cos(tmp), sin(tmp)});

    REQUIRE(ret[0] == cos(x + y));
    REQUIRE(ret[1] == sin(x + y));

    tmp = fix(fix(fix(x + y)));

    ret = unfix({cos(tmp), sin(tmp)});

    REQUIRE(ret[0] == cos(x + y));
    REQUIRE(ret[1] == sin(x + y));
}
