// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR == 13 || LLVM_VERSION_MAJOR == 14
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum");
    }

    {
        detail::sum_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == "()");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y)");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y + z)");
    }

    {
        std::ostringstream oss;

        oss << sum({x, y, z}, 2u);

        REQUIRE(oss.str() == "((x + y) + z)");
    }

    {
        std::ostringstream oss;

        oss << sum({x, y, z, x - y}, 2u);

        REQUIRE(oss.str() == "((x + y) + (z + (x - y)))");
    }

    {
        std::ostringstream oss;

        oss << sum({x, par[42], z, 4_dbl}, 2u);

        REQUIRE(boost::starts_with(oss.str(), "((x + p42) + (z + 4"));
    }
}

TEST_CASE("diff test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;
        REQUIRE(ss.gradient().empty());
    }

    {
        detail::sum_impl ss({x, y, z});
        REQUIRE(ss.gradient() == std::vector{1_dbl, 1_dbl, 1_dbl});
    }

    {
        REQUIRE(diff(sum({x, y, z}), "x") == 1_dbl);
        REQUIRE(diff(sum({x, x * x, z}), "x") == sum({1_dbl, 2_dbl * x}));
        REQUIRE(diff(sum({x, x * x, -z}), "z") == -1_dbl);
    }
}

TEST_CASE("sum function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum({}) == 0_dbl);
    REQUIRE(sum({x}) == x);

    REQUIRE_THROWS_MATCHES(sum({x}, 0), std::invalid_argument,
                           Message("The 'split' value for a sum must be at least 2, but it is 0 instead"));
    REQUIRE_THROWS_MATCHES(sum({x}, 1), std::invalid_argument,
                           Message("The 'split' value for a sum must be at least 2, but it is 1 instead"));

    REQUIRE(sum({x, y, z, t}, 2) == sum({sum({x, y}), sum({z, t})}));
    REQUIRE(sum({x, y, z, t}, 3) == sum({sum({x, y, z}), sum({t})}));
    REQUIRE(sum({x, y, z, t}, 4) == sum({x, y, z, t}));
    REQUIRE(sum({x, y, z, t, 2_dbl * x}, 3) == sum({sum({x, y, z}), sum({t, 2_dbl * x})}));
    REQUIRE(sum({0_dbl, y, 0_dbl, t, 2_dbl * x}, 3) == sum({y, t, 2_dbl * x}));
}

TEST_CASE("sum s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = sum({x, y, z});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sum({x, y, z}));
}

TEST_CASE("sum number compress")
{
    REQUIRE(sum({1_dbl, 2_dbl, 3_dbl}) == 6_dbl);
    REQUIRE(sum({1_dbl, -2_dbl, 1_dbl}) == 0_dbl);

    REQUIRE(sum({1_dbl, 2_dbl, "x"_var}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({2_dbl, "x"_var}) == sum({"x"_var, 2_dbl}));
    REQUIRE(sum({1_dbl, "x"_var, 2_dbl}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({1_dbl, 2_dbl, "x"_var}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({"x"_var, 2_dbl}) == sum({"x"_var, 2_dbl}));
    REQUIRE(sum({-1_dbl, "x"_var, 2_dbl, -1_dbl}) == "x"_var);

    REQUIRE(sum({"y"_var, 1_dbl, -21_dbl, "x"_var}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({1_dbl, "y"_var, -21_dbl, "x"_var}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({"y"_var, 1_dbl, "x"_var, -21_dbl}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({"x"_var, 1_dbl, "y"_var, -21_dbl}) == sum({"x"_var, "y"_var, -20_dbl}));
    REQUIRE(sum({"x"_var, 1_dbl, "y"_var, par[0]}) == sum({"x"_var, "y"_var, par[0], 1_dbl}));
    REQUIRE(sum({"x"_var, "y"_var, par[0]}) == sum({"x"_var, "y"_var, par[0]}));
    REQUIRE(sum({1_dbl, "y"_var, -21_dbl, "x"_var, 20_dbl}) == sum({"y"_var, "x"_var}));

    REQUIRE(std::get<func>(sum({"y"_var, 1_dbl, "x"_var, -21_dbl}).value()).args()
            == std::vector{"y"_var, "x"_var, -20_dbl});
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-10., 10.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 3u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {sum({x, y}), sum({x, expression{fp_t(.5)}}), sum({par[0], y})}, batch_size,
                            high_accuracy, compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sum."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(ins[i] + ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(ins[i] + static_cast<fp_t>(.5), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pars[i] + ins[i + batch_size], fp_t(100)));
            }
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
