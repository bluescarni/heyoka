// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/dfun.hpp>
#include <heyoka/math/sin.hpp>
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
    REQUIRE(fmt::format("{}", df_def) == "(∂^0 x)");

    auto df = dfun("x", std::vector<expression>{});
    REQUIRE(df == df_def);
    REQUIRE(!std::less<expression>{}(df, df_def));
    REQUIRE(!std::less<expression>{}(df_def, df));
    REQUIRE(std::holds_alternative<func>(df.value()));
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_didx().empty());
    REQUIRE(std::get<func>(df.value()).get_name() == "dfun__x");
    REQUIRE(std::get<func>(df.value()).args().empty());
    REQUIRE(fmt::format("{}", df) == "(∂^0 x)");

    df = dfun("x", {y, z});
    REQUIRE(df != df_def);
    REQUIRE((std::less<expression>{}(df, df_def) || std::less<expression>{}(df_def, df)));
    REQUIRE(std::holds_alternative<func>(df.value()));
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df.value()).extract<detail::dfun_impl>()->get_didx().empty());
    REQUIRE(std::get<func>(df.value()).get_name() == "dfun__x");
    REQUIRE(std::get<func>(df.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df) == "(∂^0 x)");

    auto df2 = dfun("x", {y, z}, {{0, 1}});
    REQUIRE(df2 != df);
    REQUIRE((std::less<expression>{}(df, df2) || std::less<expression>{}(df2, df)));
    REQUIRE(std::holds_alternative<func>(df2.value()));
    REQUIRE(std::get<func>(df2.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df2.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}});
    REQUIRE(std::get<func>(df2.value()).get_name() == "dfun_0,1 _x");
    REQUIRE(std::get<func>(df2.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df2) == "(∂x)/(∂a0)");

    auto df3 = dfun("x", {y, z}, {{0, 1}, {1, 2}});
    REQUIRE(df3 != df2);
    REQUIRE((std::less<expression>{}(df2, df3) || std::less<expression>{}(df3, df2)));
    REQUIRE(std::holds_alternative<func>(df3.value()));
    REQUIRE(std::get<func>(df3.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df3.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {1, 2}});
    REQUIRE(std::get<func>(df3.value()).get_name() == "dfun_0,1 1,2 _x");
    REQUIRE(std::get<func>(df3.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df3) == "(∂^3 x)/(∂a0 ∂a1^2)");

    auto df4 = dfun("x", {y, z}, {{1, 3}});
    REQUIRE(df4 != df3);
    REQUIRE((std::less<expression>{}(df3, df4) || std::less<expression>{}(df4, df3)));
    REQUIRE(std::holds_alternative<func>(df4.value()));
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{1, 3}});
    REQUIRE(std::get<func>(df4.value()).get_name() == "dfun_1,3 _x");
    REQUIRE(std::get<func>(df4.value()).args() == std::vector{y, z});
    REQUIRE(fmt::format("{}", df4) == "(∂^3 x)/(∂a1^3)");

    // Test with the overload from shared_ptr.
    auto sptr = std::make_shared<const std::vector<expression>>(std::vector{y, z});
    df4 = dfun("x", sptr, {{1, 3}});
    REQUIRE(df4 != df3);
    REQUIRE((std::less<expression>{}(df3, df4) || std::less<expression>{}(df4, df3)));
    REQUIRE(std::holds_alternative<func>(df4.value()));
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(df4.value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{1, 3}});
    REQUIRE(std::get<func>(df4.value()).get_name() == "dfun_1,3 _x");
    REQUIRE(std::get<func>(df4.value()).args() == std::vector{y, z});
    REQUIRE(&std::get<func>(df4.value()).args() == sptr.get());
    REQUIRE(fmt::format("{}", df4) == "(∂^3 x)/(∂a1^3)");

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
    REQUIRE_THROWS_MATCHES((dfun("x", func_args::shared_args_t{}, {{0, 1}})), std::invalid_argument,
                           Message("Cannot construct a dfun from a null shared pointer to its arguments"));
    REQUIRE_THROWS_MATCHES((dfun("x", func_args{}, {{0, 1}})), std::invalid_argument,
                           Message("Shared function arguments are required when constructing a dfun() instance"));
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

TEST_CASE("gradient")
{
    auto [y, z, s, t] = make_vars("y", "z", "s", "t");

    auto df = expression{func{detail::dfun_impl{}}};
    auto grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.empty());

    df = dfun("x", {y, z});
    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 2u);

    REQUIRE(grad[0] == dfun("x", {y, z}, {{0, 1}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,1 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂x)/(∂a0)");

    REQUIRE(grad[1] == dfun("x", {y, z}, {{1, 1}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{1, 1}});
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_1,1 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂x)/(∂a1)");

    df = dfun("x", {y, z}, {{0, 2}, {1, 3}});
    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 2u);

    REQUIRE(grad[0] == dfun("x", {y, z}, {{0, 3}, {1, 3}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 3}, {1, 3}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,3 1,3 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^6 x)/(∂a0^3 ∂a1^3)");

    REQUIRE(grad[1] == dfun("x", {y, z}, {{0, 2}, {1, 4}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}, {1, 4}});
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_0,2 1,4 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^6 x)/(∂a0^2 ∂a1^4)");

    df = dfun("x", {y, z, s}, {{0, 2}, {2, 3}});

    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 3u);

    REQUIRE(grad[0] == dfun("x", {y, z, s}, {{0, 3}, {2, 3}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 3}, {2, 3}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,3 2,3 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z, s});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^6 x)/(∂a0^3 ∂a2^3)");

    REQUIRE(grad[1] == dfun("x", {y, z, s}, {{0, 2}, {1, 1}, {2, 3}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}, {1, 1}, {2, 3}});
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_0,2 1,1 2,3 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z, s});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^6 x)/(∂a0^2 ∂a1 ∂a2^3)");

    REQUIRE(grad[2] == dfun("x", {y, z, s}, {{0, 2}, {2, 4}}));
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}, {2, 4}});
    REQUIRE(std::get<func>(grad[2].value()).get_name() == "dfun_0,2 2,4 _x");
    REQUIRE(std::get<func>(grad[2].value()).args() == std::vector{y, z, s});
    REQUIRE(&std::get<func>(grad[2].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[2]) == "(∂^6 x)/(∂a0^2 ∂a2^4)");

    df = dfun("x", {y, z, s, t}, {{0, 2}, {2, 3}, {3, 1}});

    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 4u);

    REQUIRE(grad[0] == dfun("x", {y, z, s, t}, {{0, 3}, {2, 3}, {3, 1}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 3}, {2, 3}, {3, 1}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,3 2,3 3,1 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^7 x)/(∂a0^3 ∂a2^3 ∂a3)");

    REQUIRE(grad[1] == dfun("x", {y, z, s, t}, {{0, 2}, {1, 1}, {2, 3}, {3, 1}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx()
            == didx_t{{0, 2}, {1, 1}, {2, 3}, {3, 1}});
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_0,2 1,1 2,3 3,1 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^7 x)/(∂a0^2 ∂a1 ∂a2^3 ∂a3)");

    REQUIRE(grad[2] == dfun("x", {y, z, s, t}, {{0, 2}, {2, 4}, {3, 1}}));
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}, {2, 4}, {3, 1}});
    REQUIRE(std::get<func>(grad[2].value()).get_name() == "dfun_0,2 2,4 3,1 _x");
    REQUIRE(std::get<func>(grad[2].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[2].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[2]) == "(∂^7 x)/(∂a0^2 ∂a2^4 ∂a3)");

    REQUIRE(grad[3] == dfun("x", {y, z, s, t}, {{0, 2}, {2, 3}, {3, 2}}));
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}, {2, 3}, {3, 2}});
    REQUIRE(std::get<func>(grad[3].value()).get_name() == "dfun_0,2 2,3 3,2 _x");
    REQUIRE(std::get<func>(grad[3].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[3].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[3]) == "(∂^7 x)/(∂a0^2 ∂a2^3 ∂a3^2)");

    df = dfun("x", {y, z, s, t}, {{0, 1}});

    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 4u);

    REQUIRE(grad[0] == dfun("x", {y, z, s, t}, {{0, 2}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 2}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,2 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^2 x)/(∂a0^2)");

    REQUIRE(grad[1] == dfun("x", {y, z, s, t}, {{0, 1}, {1, 1}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx()
            == didx_t{
                {0, 1},
                {1, 1},
            });
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_0,1 1,1 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^2 x)/(∂a0 ∂a1)");

    REQUIRE(grad[2] == dfun("x", {y, z, s, t}, {{0, 1}, {2, 1}}));
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {2, 1}});
    REQUIRE(std::get<func>(grad[2].value()).get_name() == "dfun_0,1 2,1 _x");
    REQUIRE(std::get<func>(grad[2].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[2].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[2]) == "(∂^2 x)/(∂a0 ∂a2)");

    REQUIRE(grad[3] == dfun("x", {y, z, s, t}, {{0, 1}, {3, 1}}));
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {3, 1}});
    REQUIRE(std::get<func>(grad[3].value()).get_name() == "dfun_0,1 3,1 _x");
    REQUIRE(std::get<func>(grad[3].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[3].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[3]) == "(∂^2 x)/(∂a0 ∂a3)");

    df = dfun("x", {y, z, s, t}, {{2, 1}});

    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 4u);

    REQUIRE(grad[0] == dfun("x", {y, z, s, t}, {{0, 1}, {2, 1}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {2, 1}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,1 2,1 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^2 x)/(∂a0 ∂a2)");

    REQUIRE(grad[1] == dfun("x", {y, z, s, t}, {{1, 1}, {2, 1}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx()
            == didx_t{
                {1, 1},
                {2, 1},
            });
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_1,1 2,1 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^2 x)/(∂a1 ∂a2)");

    REQUIRE(grad[2] == dfun("x", {y, z, s, t}, {{2, 2}}));
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{2, 2}});
    REQUIRE(std::get<func>(grad[2].value()).get_name() == "dfun_2,2 _x");
    REQUIRE(std::get<func>(grad[2].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[2].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[2]) == "(∂^2 x)/(∂a2^2)");

    REQUIRE(grad[3] == dfun("x", {y, z, s, t}, {{2, 1}, {3, 1}}));
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{2, 1}, {3, 1}});
    REQUIRE(std::get<func>(grad[3].value()).get_name() == "dfun_2,1 3,1 _x");
    REQUIRE(std::get<func>(grad[3].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[3].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[3]) == "(∂^2 x)/(∂a2 ∂a3)");

    df = dfun("x", {y, z, s, t}, {{3, 1}});

    grad = std::get<func>(df.value()).extract<detail::dfun_impl>()->gradient();

    REQUIRE(grad.size() == 4u);

    REQUIRE(grad[0] == dfun("x", {y, z, s, t}, {{0, 1}, {3, 1}}));
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[0].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{0, 1}, {3, 1}});
    REQUIRE(std::get<func>(grad[0].value()).get_name() == "dfun_0,1 3,1 _x");
    REQUIRE(std::get<func>(grad[0].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[0].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[0]) == "(∂^2 x)/(∂a0 ∂a3)");

    REQUIRE(grad[1] == dfun("x", {y, z, s, t}, {{1, 1}, {3, 1}}));
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[1].value()).extract<detail::dfun_impl>()->get_didx()
            == didx_t{
                {1, 1},
                {3, 1},
            });
    REQUIRE(std::get<func>(grad[1].value()).get_name() == "dfun_1,1 3,1 _x");
    REQUIRE(std::get<func>(grad[1].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[1].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[1]) == "(∂^2 x)/(∂a1 ∂a3)");

    REQUIRE(grad[2] == dfun("x", {y, z, s, t}, {{2, 1}, {3, 1}}));
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[2].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{2, 1}, {3, 1}});
    REQUIRE(std::get<func>(grad[2].value()).get_name() == "dfun_2,1 3,1 _x");
    REQUIRE(std::get<func>(grad[2].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[2].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[2]) == "(∂^2 x)/(∂a2 ∂a3)");

    REQUIRE(grad[3] == dfun("x", {y, z, s, t}, {{3, 2}}));
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_id_name() == "x");
    REQUIRE(std::get<func>(grad[3].value()).extract<detail::dfun_impl>()->get_didx() == didx_t{{3, 2}});
    REQUIRE(std::get<func>(grad[3].value()).get_name() == "dfun_3,2 _x");
    REQUIRE(std::get<func>(grad[3].value()).args() == std::vector{y, z, s, t});
    REQUIRE(&std::get<func>(grad[3].value()).args() == &std::get<func>(df.value()).args());
    REQUIRE(fmt::format("{}", grad[3]) == "(∂^2 x)/(∂a3^2)");
}

TEST_CASE("contains_dfun")
{
    auto [y, z] = make_vars("y", "z");

    REQUIRE(!detail::contains_dfun({y}));
    REQUIRE(!detail::contains_dfun({y, z}));
    REQUIRE(!detail::contains_dfun({y + z, 2_dbl * z - 1_dbl}));

    auto ex = y + z;

    REQUIRE(!detail::contains_dfun({cos(ex + 1_dbl), sin(ex)}));

    ex = dfun("x", {y, z}, {{0, 1}});

    REQUIRE(detail::contains_dfun({cos(ex + 1_dbl), sin(ex)}));
    REQUIRE(detail::contains_dfun({cos("x"_var + 1_dbl), sin(ex)}));
}
