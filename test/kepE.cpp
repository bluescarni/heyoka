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
#include <utility>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/cstdint.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/roots.hpp>

#include <llvm/Config/llvm-config.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

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

// Boost-based Kepler solver.
auto bmt_inv_kep_E = [](auto ecc, auto M) {
    using std::sin;
    using std::cos;

    using fp_t = decltype(ecc);

    // Initial guess.
    auto ig = ecc < 0.8 ? M : static_cast<fp_t>(boost::math::constants::pi<double>());

    auto func = [ecc, M](auto E) { return std::make_pair(E - ecc * sin(E) - M, 1 - ecc * cos(E)); };

    boost::uintmax_t max_iter = 50;

    return boost::math::tools::newton_raphson_iterate(func, ig, fp_t(0), fp_t(2 * boost::math::constants::pi<double>()),
                                                      std::numeric_limits<fp_t>::digits - 2, max_iter);
};

TEST_CASE("kepE def ctor")
{
    detail::kepE_impl k;

    REQUIRE(k.args().size() == 2u);
    REQUIRE(k.args()[0] == 0_dbl);
    REQUIRE(k.args()[1] == 0_dbl);
}

TEST_CASE("kepE diff")
{
    auto [x, y] = make_vars("x", "y");

    {
        REQUIRE(diff(kepE(x, y), x) == sin(kepE(x, y)) / (1_dbl - x * cos(kepE(x, y))));
        REQUIRE(diff(kepE(x, y), y) == 1_dbl / (1_dbl - x * cos(kepE(x, y))));
        auto E = kepE(x * x, x * y);
        REQUIRE(diff(E, x) == (2_dbl * x * sin(E) + y) / (1_dbl - x * x * cos(E)));
        REQUIRE(diff(E, y) == x / (1_dbl - x * x * cos(E)));
    }

    {
        REQUIRE(diff(kepE(par[0], y), par[0]) == sin(kepE(par[0], y)) / (1_dbl - par[0] * cos(kepE(par[0], y))));
        REQUIRE(diff(kepE(x, par[1]), par[1]) == 1_dbl / (1_dbl - x * cos(kepE(x, par[1]))));
        auto E = kepE(par[0] * par[0], par[0] * par[1]);
        REQUIRE(diff(E, par[0]) == (2_dbl * par[0] * sin(E) + par[1]) / (1_dbl - par[0] * par[0] * cos(E)));
        REQUIRE(diff(E, par[1]) == par[0] / (1_dbl - par[0] * par[0] * cos(E)));
    }
}

TEST_CASE("kepE decompose")
{
    {
        auto [u0, u1] = make_vars("u_0", "u_1");

        taylor_dc_t dec;
        dec.emplace_back("e"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("M"_var, std::vector<std::uint32_t>{});
        taylor_decompose(kepE(u0, u1), dec);

        REQUIRE(dec.size() == 6u);

        REQUIRE(dec[2].first == kepE(u0, u1));
        REQUIRE(dec[2].second == std::vector<std::uint32_t>{5, 3});

        REQUIRE(dec[3].first == sin("u_2"_var));
        REQUIRE(dec[3].second == std::vector<std::uint32_t>{4});

        REQUIRE(dec[4].first == cos("u_2"_var));
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{3});

        REQUIRE(dec[5].first == "u_0"_var * "u_4"_var);
        REQUIRE(dec[5].second.empty());
    }

    {
        auto [u0, u1] = make_vars("u_0", "u_1");

        taylor_dc_t dec;
        dec.emplace_back("e"_var, std::vector<std::uint32_t>{});
        dec.emplace_back("M"_var, std::vector<std::uint32_t>{});
        taylor_decompose(kepE(u0 + u1, u1 - u0), dec);

        REQUIRE(dec.size() == 8u);

        REQUIRE(dec[2].first == "u_0"_var + "u_1"_var);
        REQUIRE(dec[2].second.empty());

        REQUIRE(dec[3].first == "u_1"_var - "u_0"_var);
        REQUIRE(dec[3].second.empty());

        REQUIRE(dec[4].first == kepE("u_2"_var, "u_3"_var));
        REQUIRE(dec[4].second == std::vector<std::uint32_t>{7, 5});

        REQUIRE(dec[5].first == sin("u_4"_var));
        REQUIRE(dec[5].second == std::vector<std::uint32_t>{6});

        REQUIRE(dec[6].first == cos("u_4"_var));
        REQUIRE(dec[6].second == std::vector<std::uint32_t>{5});

        REQUIRE(dec[7].first == "u_2"_var * "u_6"_var);
        REQUIRE(dec[7].second.empty());
    }
}

TEST_CASE("kepE overloads")
{
    auto k = kepE("x"_var, 1.1);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1});

    k = kepE("x"_var, 1.1l);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = kepE("x"_var, mppp::real128{"1.1"});
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{mppp::real128{"1.1"}});
#endif

    k = kepE(1.1, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1});

    k = kepE(1.1l, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = kepE(mppp::real128{"1.1"}, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{mppp::real128{"1.1"}});
#endif
}

TEST_CASE("kepE cse")
{
    auto x = "x"_var, y = "y"_var;

    llvm_state s;

    auto dc = taylor_add_jet<double>(s, "jet", {cos(kepE(x, y)) + sin(kepE(x, y)) + kepE(x, y), x}, 1, 1, false, false);

    REQUIRE(dc.size() == 10u);
}

// NOTE: this test checks a numerical integration of the Stark problem using kepE vs
// the implementation in the Python notebook, which does not use kepE. This test is useful
// to check both the symbolic and automatic derivatives of kepE.
TEST_CASE("kepE stark")
{
    using std::cos;
    using std::sin;
    using std::sqrt;

    auto [L, G, H, l, g, h] = make_vars("L", "G", "H", "l", "g", "h");

    const auto eps = 1e-3;

    auto E = kepE(sqrt(1. - G * G / (L * L)), l);

    auto Ham = -0.5 * pow(L, -2.)
               - eps * L * sqrt(1. - H * H / (G * G))
                     * (L * (cos(E) - sqrt(1. - G * G / (L * L))) * sin(g) + G * sin(E) * cos(g));

    auto ta = taylor_adaptive<double>{{prime(L) = -diff(Ham, l), prime(G) = -diff(Ham, g), prime(H) = -diff(Ham, h),
                                       prime(l) = diff(Ham, L), prime(g) = diff(Ham, G), prime(h) = diff(Ham, H)},
                                      std::vector{1.0045488165591647, 0.9731906288081488, -0.9683287292736491,
                                                  2.8485929090946436, 4.314274521695855, 3.3415926535897924}};
    auto ic_L = ta.get_state()[0];
    auto ic_G = ta.get_state()[1];
    auto ic_E = ta.get_state()[3];
    ta.get_state_data()[3] = ic_E - sqrt(1 - ic_G * ic_G / (ic_L * ic_L)) * sin(ic_E);

    auto [oc, _1, _2, _3, _4] = ta.propagate_until(250.);

    REQUIRE(oc == taylor_outcome::time_limit);

    REQUIRE(ta.get_state()[0] == approximately(1.0046255890340732));
    REQUIRE(ta.get_state()[1] == approximately(0.9802027040286941));
    REQUIRE(ta.get_state()[2] == approximately(-0.9683287292736491));
    REQUIRE(ta.get_state()[4] == approximately(3.8714912951286484));
    REQUIRE(ta.get_state()[5] == approximately(2.745578227312136));

    auto f_L = ta.get_state()[0];
    auto f_G = ta.get_state()[1];
    auto f_E = -2.2193101195959493;

    // NOTE: slightly less precise because we don't reduce the angle via callback here.
    REQUIRE(sin(ta.get_state()[3]) == approximately(sin(f_E - sqrt(1 - f_G * f_G / (f_L * f_L)) * sin(f_E)), 10000.));
    REQUIRE(cos(ta.get_state()[3]) == approximately(cos(f_E - sqrt(1 - f_G * f_G / (f_L * f_L)) * sin(f_E)), 10000.));
}

TEST_CASE("kepE s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = kepE(x, y);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == kepE(x, y));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(0., 0.9);

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

            add_cfunc<fp_t>(
                s, "cfunc", {kepE(x, y), kepE(x, par[0]), kepE(x, .5_dbl), kepE(par[0], y), kepE(.5_dbl, y)},
                kw::batch_size = batch_size, kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepE."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(bmt_inv_kep_E(ins[i], ins[i + batch_size]), fp_t(1000)));
                REQUIRE(outs[i + batch_size] == approximately(bmt_inv_kep_E(ins[i], pars[i]), fp_t(1000)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(bmt_inv_kep_E(ins[i], fp_t(.5)), fp_t(1000)));
                REQUIRE(outs[i + 3u * batch_size]
                        == approximately(bmt_inv_kep_E(pars[i], ins[i + batch_size]), fp_t(1000)));
                REQUIRE(outs[i + 4u * batch_size]
                        == approximately(bmt_inv_kep_E(fp_t(.5), ins[i + batch_size]), fp_t(1000)));
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
