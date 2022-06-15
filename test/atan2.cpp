// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
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
#include <heyoka/math/atan2.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

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

    REQUIRE(diff(atan2(y, par[0]), par[0]) == (-y) / (par[0] * par[0] + y * y));
    REQUIRE(diff(atan2(par[1], x), par[1]) == x / (x * x + par[1] * par[1]));
    REQUIRE(diff(atan2(y, x), par[2]) == 0_dbl);
    REQUIRE(diff(atan2(par[0] * par[1], par[1] / par[0]), par[0])
            == (par[1] / par[0] * par[1] - (par[0] * par[1]) * (-par[1] / (par[0] * par[0])))
                   / ((par[1] / par[0]) * (par[1] / par[0]) + (par[0] * par[1]) * (par[0] * par[1])));
}

TEST_CASE("atan2 decompose")
{
    auto [u0, u1] = make_vars("u_0", "u_1");

    {
        taylor_dc_t dec;
        dec.emplace_back("y"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("x"_var, std::vector<std::uint32_t>{});
        taylor_decompose(atan2(u0, u1), dec);

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
        taylor_decompose(atan2(u0 + u1, u1 - u0), dec);

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

TEST_CASE("atan2 integration")
{
    auto x = "x"_var;

    // NOTE: the solution of this ODE is exp(t).
    auto ta = taylor_adaptive<double>{{prime(x) = atan2(sin(x), cos(x))}, {1.}};

    ta.propagate_until(1.5);

    // Check the value of e**1.5.
    REQUIRE(ta.get_state()[0] == approximately(4.4816890703380645));
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

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::atan2;

        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-10., 10.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 5u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc",
                            {atan2(x, y), atan2(x, par[0]), atan2(x, 3_dbl), atan2(par[0], y), atan2(1_dbl, y)},
                            batch_size, high_accuracy, compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.atan2"));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(atan2(ins[i], ins[i + batch_size]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(atan2(ins[i], pars[i]), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(atan2(ins[i], fp_t(3)), fp_t(100)));
                REQUIRE(outs[i + 3u * batch_size] == approximately(atan2(pars[i], ins[i + batch_size]), fp_t(100)));
                REQUIRE(outs[i + 4u * batch_size] == approximately(atan2(fp_t(1), ins[i + batch_size]), fp_t(100)));
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
