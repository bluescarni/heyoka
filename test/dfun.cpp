// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/dfun.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

using didx_t = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    auto [y, z] = make_vars("y", "z");

    auto df_def = expression{func{detail::dfun_impl{}}};
    REQUIRE(std::holds_alternative<func>(df_def.value()));
    REQUIRE(std::get<func>(df_def.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df_def.value()).extract<detail::dfun_impl>()->get_didx().empty());
    REQUIRE(std::get<func>(df_def.value()).get_name() == "dfun__x");
    REQUIRE(std::get<func>(df_def.value()).args().empty());
    REQUIRE(fmt::format("{}", df_def) == "(d^0 x)");

    auto df = dfun("x", {});
    REQUIRE(df == df_def);
    REQUIRE(!std::less<expression>{}(df, df_def));
    REQUIRE(!std::less<expression>{}(df_def, df));
    REQUIRE(std::holds_alternative<func>(df.value()));
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_didx().empty());
    REQUIRE(std::get<func>(df.value()).get_name() == "dfun__x");
    REQUIRE(std::get<func>(df.value()).args().empty());
    REQUIRE(fmt::format("{}", df) == "(d^0 x)");

    df = dfun("x", {y, z});
    REQUIRE(df != df_def);
    REQUIRE((std::less<expression>{}(df, df_def) || std::less<expression>{}(df_def, df)));
    REQUIRE(std::holds_alternative<func>(df.value()));
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_didx().empty());
    REQUIRE(std::get<func>(df.value()).get_name() == "dfun__x");
    REQUIRE(std::get<func>(df.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df) == "(d^0 x)");

    auto df2 = dfun("x", {y, z}, {{0, 1}});
    REQUIRE(df2 != df);
    REQUIRE((std::less<expression>{}(df, df2) || std::less<expression>{}(df2, df)));
    REQUIRE(std::holds_alternative<func>(df2.value()));
    REQUIRE(std::get<func>(df2.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df2.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}});
    REQUIRE(std::get<func>(df2.value()).get_name() == "dfun_0,1 _x");
    REQUIRE(std::get<func>(df2.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df2) == "(dx)/(da0)");

    auto df3 = dfun("x", {y, z}, {{0, 1}, {1, 2}});
    REQUIRE(df3 != df2);
    REQUIRE((std::less<expression>{}(df2, df3) || std::less<expression>{}(df3, df2)));
    REQUIRE(std::holds_alternative<func>(df3.value()));
    REQUIRE(std::get<func>(df3.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df3.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {1, 2}});
    REQUIRE(std::get<func>(df3.value()).get_name() == "dfun_0,1 1,2 _x");
    REQUIRE(std::get<func>(df3.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df3) == "(d^3 x)/(da0 da1^2)");

    auto df4 = dfun("x", {y, z}, {{1, 3}});
    REQUIRE(df4 != df3);
    REQUIRE((std::less<expression>{}(df3, df4) || std::less<expression>{}(df4, df3)));
    REQUIRE(std::holds_alternative<func>(df4.value()));
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{1, 3}});
    REQUIRE(std::get<func>(df4.value()).get_name() == "dfun_1,3 _x");
    REQUIRE(std::get<func>(df4.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df4) == "(d^3 x)/(da1^3)");

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        (dfun("x", {y, z}, {{2, 3}})), std::invalid_argument,
        Message("Invalid index 2 detected in the indices vector passed to the constructor of a dfun: "
                "the index must be less than the number of arguments (2)"));
    REQUIRE_THROWS_MATCHES((dfun("x", {y, z}, {{0, 1}, {1, 0}})), std::invalid_argument,
                           Message("Invalid zero derivative order detected in the indices vector passed to "
                                   "the constructor of a dfun: all derivative orders must be positive"));
    REQUIRE_THROWS_MATCHES((dfun("x", {y, z}, {{1, 1}, {0, 1}})), std::invalid_argument,
                           Message("The indices in the indices vector passed to "
                                   "the constructor of a dfun must be sorted in strictly ascending order"));
}

TEST_CASE("sin s11n")
{
    std::stringstream ss;

    auto [y, z] = make_vars("y", "z");

    auto ex = dfun("x", {y, z}, {{1, 3}});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == dfun("x", {y, z}, {{1, 3}}));
}
