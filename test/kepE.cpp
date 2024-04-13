// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
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

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
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

using namespace heyoka;
using namespace heyoka_test;

#if defined(HEYOKA_HAVE_REAL128) || defined(HEYOKA_HAVE_REAL)

using namespace mppp::literals;

#endif

const auto fp_types = std::tuple<float, double
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
#if LLVM_VERSION_MAJOR <= 17
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

TEST_CASE("kepE overloads")
{
    auto k = kepE("x"_var, 1.1f);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1f});

    k = kepE("x"_var, 1.1);
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

#if defined(HEYOKA_HAVE_REAL)
    k = kepE("x"_var, 1.1_r256);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1_r256});
#endif

    k = kepE(1.1f, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1f});

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

#if defined(HEYOKA_HAVE_REAL)
    k = kepE(1.1_r256, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1_r256});
#endif
}

TEST_CASE("kepE cse")
{
    auto x = "x"_var, y = "y"_var;

    auto ta = taylor_adaptive<double>{
        {prime(x) = cos(kepE(x, y)) + sin(kepE(x, y)) + kepE(x, y), prime(y) = x}, {0., 0.}, kw::tol = 1.};

    REQUIRE(ta.get_decomposition().size() == 9u);
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

    auto [oc, _1, _2, _3, _4, _5] = ta.propagate_until(250.);

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
                s, "cfunc", {kepE(x, y), kepE(x, par[0]), kepE(x, .5_dbl), kepE(par[0], y), kepE(.5_dbl, y)}, {x, y},
                kw::batch_size = batch_size, kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepE."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc mp")
{
    using fp_t = mppp::real;

    const auto prec = 237u;

    auto [x, y] = make_vars("x", "y");

    std::uniform_real_distribution<double> rdist(0., 0.9);

    auto gen = [&]() { return mppp::real(rdist(rng), static_cast<int>(prec)); };

    std::vector<fp_t> outs, ins, pars;

    outs.resize(5u);
    ins.resize(2u);
    pars.resize(1u);

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            std::generate(ins.begin(), ins.end(), gen);
            std::generate(outs.begin(), outs.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc",
                            {kepE(x, y), kepE(x, par[0]), kepE(x, .5_dbl), kepE(par[0], y), kepE(.5_dbl, y)}, {x, y},
                            kw::prec = prec, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepE."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            REQUIRE(outs[0] - ins[0] * sin(outs[0]) == approximately(ins[1], fp_t(1000)));
            REQUIRE(outs[1] - ins[0] * sin(outs[1]) == approximately(pars[0], fp_t(1000)));
            REQUIRE(outs[2] - ins[0] * sin(outs[2]) == approximately(mppp::real{0.5, prec}, fp_t(1000)));
            REQUIRE(outs[3] - pars[0] * sin(outs[3]) == approximately(ins[1], fp_t(1000)));
            REQUIRE(outs[4] - mppp::real{0.5, prec} * sin(outs[4]) == approximately(ins[1], fp_t(1000)));
        }
    }
}

#endif
