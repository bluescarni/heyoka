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
#include <variant>
#include <vector>

#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

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

TEST_CASE("atan2 overloads")
{
    auto k = atan2("x"_var, 1.1f);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1f});

    k = atan2("x"_var, 1.1);
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

#if defined(HEYOKA_HAVE_REAL)
    k = atan2("x"_var, 1.1_r256);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1_r256});
#endif

    k = atan2(1.1f, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1f});

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

#if defined(HEYOKA_HAVE_REAL)
    k = atan2(1.1_r256, "x"_var);
    REQUIRE(std::get<func>(k.value()).args()[1] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{1.1_r256});
#endif
}

TEST_CASE("atan2 cse")
{
    auto x = "x"_var, y = "y"_var;

    llvm_state s;

    auto dc = taylor_add_jet<double>(s, "jet", {atan2(y, x) + (x * x + y * y), x}, 1, 1, false, false);

    REQUIRE(dc.size() == 9u);
}

TEST_CASE("atan2 const fold")
{
    REQUIRE(atan2(1.1_dbl, 2.2_dbl) == expression{std::atan2(1.1, 2.2)});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(atan2(mppp::real{"1.1", 123}, 2_dbl) == expression{mppp::atan2(mppp::real{"1.1", 123}, 2.)});

#endif
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

            add_cfunc<fp_t>(
                s, "cfunc", {atan2(x, y), atan2(x, par[0]), atan2(x, 3_dbl), atan2(par[0], y), atan2(1_dbl, y)}, {x, y},
                kw::batch_size = batch_size, kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.atan2."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc_mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(
                s, "cfunc",
                {atan2(x, y), atan2(x, par[0]), atan2(par[0], x), atan2(x, 3. / 2_dbl), atan2(3. / 2_dbl, x)}, {x, y},
                kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{"1.1", prec}, mppp::real{"2.1", prec}};
            const std::vector pars{mppp::real{"3.1", prec}};
            std::vector<mppp::real> outs(5u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == atan2(ins[i], ins[i + 1u]));
            REQUIRE(outs[i + 1u] == atan2(ins[i], pars[i]));
            REQUIRE(outs[i + 2u * 1u] == atan2(pars[i], ins[i]));
            REQUIRE(outs[i + 3u * 1u] == atan2(ins[i], 3. / 2));
            REQUIRE(outs[i + 4u * 1u] == atan2(3. / 2, ins[i]));
        }
    }
}

#endif

TEST_CASE("normalise")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(normalise(atan2(x, y)) == atan2(x, y));
    REQUIRE(normalise(subs(atan2(x, y), {{x, .1_dbl}, {y, .2_dbl}})) == atan2(.1_dbl, .2_dbl));
}

// Tests to check vectorisation via the vector-function-abi-variant machinery.
TEST_CASE("vfabi double")
{
    for (auto fast_math : {false, true}) {
        llvm_state s{kw::slp_vectorize = true, kw::fast_math = fast_math};

        auto [a, b] = make_vars("a", "b");

        add_cfunc<double>(s, "cfunc", {atan2(a, .3), atan2(b, .4)}, {a, b});
        add_cfunc<double>(s, "cfuncs", {atan2(a, .3), atan2(b, .4)}, {a, b}, kw::strided = true);

        s.compile();

        auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("cfunc"));

        const std::vector ins{.1, .2};
        std::vector<double> outs(2u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::atan2(.1, .3)));
        REQUIRE(outs[1] == approximately(std::atan2(.2, .4)));

        // NOTE: autovec with external scalar functions seems to work
        // only since LLVM 16.
#if defined(HEYOKA_WITH_SLEEF) && LLVM_VERSION_MAJOR >= 16

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@atan2", boost::is_iequal()));
             it != string_find_iterator(); ++it) {
            ++count;
        }

        // NOTE: at the moment we have comprehensive coverage of LLVM versions
        // in the CI only for x86_64.
        if (tf.sse2) {
            // NOTE: occurrences of the scalar version:
            // - 2 calls in the strided cfunc,
            // - 1 declaration.
            REQUIRE(count == 3u);
        }

        if (tf.aarch64) {
            REQUIRE(count == 3u);
        }

        // NOTE: currently no auto-vectorization happens on ppc64 due apparently
        // to the way the target machine is being set up by orc/lljit (it works
        // fine with the opt tool). When this is resolved, we can test ppc64 too.

        // if (tf.vsx) {
        //     REQUIRE(count == 3u);
        // }

#endif
    }
}

TEST_CASE("vfabi float")
{
    for (auto fast_math : {false, true}) {
        llvm_state s{kw::slp_vectorize = true, kw::fast_math = fast_math};

        auto [a, b, c, d] = make_vars("a", "b", "c", "d");

        add_cfunc<float>(s, "cfunc", {atan2(a, .5f), atan2(b, .6f), atan2(c, .7f), atan2(d, .8f)}, {a, b, c, d});
        add_cfunc<float>(s, "cfuncs", {atan2(a, .5f), atan2(b, .6f), atan2(c, .7f), atan2(d, .8f)}, {a, b, c, d},
                         kw::strided = true);

        s.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(float *, const float *, const float *, const float *)>(s.jit_lookup("cfunc"));

        const std::vector<float> ins{.1f, .2f, .3f, .4f};
        std::vector<float> outs(4u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::atan2(.1f, .5f)));
        REQUIRE(outs[1] == approximately(std::atan2(.2f, .6f)));
        REQUIRE(outs[2] == approximately(std::atan2(.3f, .7f)));
        REQUIRE(outs[3] == approximately(std::atan2(.4f, .8f)));

        // NOTE: autovec with external scalar functions seems to work
        // only since LLVM 16.
#if defined(HEYOKA_WITH_SLEEF) && LLVM_VERSION_MAJOR >= 16

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@atan2f", boost::is_iequal()));
             it != string_find_iterator(); ++it) {
            ++count;
        }

        // NOTE: at the moment we have comprehensive coverage of LLVM versions
        // in the CI only for x86_64.
        if (tf.sse2) {
            // NOTE: occurrences of the scalar version:
            // - 4 calls in the strided cfunc,
            // - 1 declaration.
            REQUIRE(count == 5u);
        }

        if (tf.aarch64) {
            REQUIRE(count == 5u);
        }

        // NOTE: currently no auto-vectorization happens on ppc64 due apparently
        // to the way the target machine is being set up by orc/lljit (it works
        // fine with the opt tool). When this is resolved, we can test ppc64 too.

        // if (tf.vsx) {
        //     REQUIRE(count == 5u);
        // }

#endif
    }
}
