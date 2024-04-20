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
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const expression &x)
{
    return x * x;
}

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

TEST_CASE("pow stream")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        oss << pow(x, y);

        REQUIRE(oss.str() == "x**y");
    }

    {
        std::ostringstream oss;

        oss << pow(prod({-1_dbl, x}), y);

        REQUIRE(oss.str() == "(-x)**y");
    }

    {
        std::ostringstream oss;

        oss << pow(pow(x, y), z);

        REQUIRE(oss.str() == "(x**y)**z");
    }

    {
        std::ostringstream oss;

        oss << pow(x, pow(y, z));

        REQUIRE(oss.str() == "x**y**z");
    }

    {
        std::ostringstream oss;

        oss << pow(log(x), y);

        REQUIRE(oss.str() == "log(x)**y");
    }
}

TEST_CASE("pow expo 0")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 0.) == 1_dbl);
    REQUIRE(heyoka::pow(x, 0.l) == 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 0._rq) == 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, 1.) != 1_dbl);
    REQUIRE(heyoka::pow(x, 1.l) != 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1._rq) != 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, expression{0.}) == 1_dbl);
    REQUIRE(heyoka::pow(x, expression{0.l}) == 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, expression{0._rq}) == 1_dbl);

#endif

    REQUIRE(heyoka::pow(x, expression{1.}) != 1_dbl);
    REQUIRE(heyoka::pow(x, expression{1.l}) != 1_dbl);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, expression{1._rq}) != 1_dbl);

#endif
}

TEST_CASE("pow expo 1")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 1.) == x);
    REQUIRE(heyoka::pow(x, 1.l) == x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1._rq) == x);

#endif

    REQUIRE(heyoka::pow(x, 1.1) != x);
    REQUIRE(heyoka::pow(x, 1.1l) != x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 1.1_rq) != x);

#endif
}

TEST_CASE("pow diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(pow(3_dbl, x * x + y), "x") == (pow(3_dbl, x * x + y) * log(3_dbl)) * (x + x));
    REQUIRE(diff(pow(x * x + y, 1.2345_dbl), "y") == 1.2345_dbl * pow(x * x + y, 1.2345_dbl - 1_dbl));

    REQUIRE(diff(pow(3_dbl, par[0] * par[0] + y), par[0])
            == (pow(3_dbl, par[0] * par[0] + y) * log(3_dbl)) * (par[0] + par[0]));
    REQUIRE(diff(pow(x * x + par[1], 1.2345_dbl), par[1]) == 1.2345_dbl * pow(x * x + par[1], 1.2345_dbl - 1_dbl));
}

TEST_CASE("pow s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = pow(x, y);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == pow(x, y));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::pow;
        using std::isfinite;
        using std::isnan;

        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(.1, 10.);

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

            add_cfunc<fp_t>(s, "cfunc1", {pow(x, y), pow(x, par[0]), pow(x, fp_t{3} / 2)}, {x, y},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            add_cfunc<fp_t>(s, "cfunc2",
                            {expression{func{detail::pow_impl(x, 0_dbl)}}, pow(x, fp_t{-3}),
                             pow(x, -std::numeric_limits<fp_t>::infinity())},
                            {x, y}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            add_cfunc<fp_t>(s, "cfunc3", {pow(x, fp_t(7)), pow(x, fp_t{-8}), pow(x, fp_t{11} / 2)}, {x, y},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow_pos_small_half_3."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow_pos_small_int_0."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow_neg_small_int_3."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow_pos_small_half_11."));
            }

            s.compile();

            auto *cf_ptr1
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc1"));

            auto *cf_ptr2
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc2"));

            auto *cf_ptr3
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc3"));

            cf_ptr1(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(pow(ins[i], ins[i + batch_size]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(pow(ins[i], pars[i]), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pow(ins[i], fp_t(3) / 2), fp_t(100)));
            }

            cf_ptr2(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(fp_t(1), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(pow(ins[i], fp_t{-3}), fp_t(100)));
                REQUIRE((outs[i + 2u * batch_size] == approximately(fp_t(0), fp_t(100))
                         || (!isfinite(outs[i + 2u * batch_size]) && !isnan(outs[i + 2u * batch_size]))));
            }

            cf_ptr3(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(pow(ins[i], fp_t(7)), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(pow(ins[i], fp_t(-8)), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pow(ins[i], fp_t{11} / 2), fp_t(100)));
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

    // Small test to check that pow() is never invoked for small powers.
    {
        auto x = make_vars("x");

        llvm_state s{kw::opt_level = 0u};

        add_cfunc<double>(s, "cfunc1", {pow(x, 2_dbl), pow(x, -3_dbl), pow(x, -3_dbl / 2.), pow(x, 5_dbl / 2.)}, {x});

        REQUIRE(!boost::contains(s.get_ir(), "llvm.pow"));
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
                s, "cfunc1", {pow(x, y), pow(x, par[0]), pow(x, mppp::real{3. / 2, prec}), pow(x, mppp::real{6, prec})},
                {x, y}, kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr1
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc1"));

            const std::vector ins{mppp::real{"1.1", prec}, mppp::real{"2.1", prec}};
            const std::vector pars{mppp::real{"3.1", prec}};
            std::vector<mppp::real> outs(4u, mppp::real{0, prec});

            cf_ptr1(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == pow(ins[i], ins[i + 1u]));
            REQUIRE(outs[i + 1u] == pow(ins[i], pars[i]));
            REQUIRE(outs[i + 2u * 1u] == approximately(pow(ins[i], 3. / 2)));
            REQUIRE(outs[i + 3u * 1u] == approximately(pow(ins[i], mppp::real{6, prec})));
        }
    }
}

#endif

TEST_CASE("pow special cases")
{
    REQUIRE(pow(1.1_dbl, 2.2_dbl) == expression{std::pow(1.1, 2.2)});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(pow(2_dbl, mppp::real{"1.1", 123}) == expression{mppp::pow(2., mppp::real{"1.1", 123})});

#endif

    REQUIRE(pow(pow("x"_var, 3_dbl), 2_dbl) == pow("x"_var, 6_dbl));
    REQUIRE(pow(pow("x"_var, .5_dbl), 2_dbl) == "x"_var);

    auto [x, y] = make_vars("x", "y");

    REQUIRE(pow(pow(x, y), -5.) == pow(x, prod({-5._dbl, y})));
}

TEST_CASE("pow overloads")
{
    auto k = pow("x"_var, 1.1f);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1f});

    k = pow("x"_var, 1.1);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1});

    k = pow("x"_var, 1.1l);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1l});

#if defined(HEYOKA_HAVE_REAL128)
    k = pow("x"_var, mppp::real128{"1.1"});
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{mppp::real128{"1.1"}});
#endif

#if defined(HEYOKA_HAVE_REAL)
    k = pow("x"_var, 1.1_r256);
    REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);
    REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{1.1_r256});
#endif
}

// Tests to check vectorisation via the vector-function-abi-variant machinery.
TEST_CASE("vfabi double")
{
    for (auto fast_math : {false, true}) {
        llvm_state s{kw::slp_vectorize = true, kw::fast_math = fast_math};

        auto [a, b] = make_vars("a", "b");

        add_cfunc<double>(s, "cfunc", {pow(a, .1), pow(b, .2)}, {a, b});
        add_cfunc<double>(s, "cfuncs", {pow(a, .1), pow(b, .2)}, {a, b}, kw::strided = true);

        s.compile();

        auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("cfunc"));

        const std::vector ins{1., 2.};
        std::vector<double> outs(2u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::pow(1., .1)));
        REQUIRE(outs[1] == approximately(std::pow(2., .2)));

#if defined(HEYOKA_WITH_SLEEF)

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@llvm.pow.f64", boost::is_iequal()));
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

#if LLVM_VERSION_MAJOR >= 16

        // NOTE: LLVM16 is currently the version tested in the CI on arm64.
        if (tf.aarch64) {
            REQUIRE(count == 3u);
        }

#endif

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

        add_cfunc<float>(s, "cfunc", {pow(a, .6f), pow(b, .7f), pow(c, .8f), pow(d, .9f)}, {a, b, c, d});
        add_cfunc<float>(s, "cfuncs", {pow(a, .6f), pow(b, .7f), pow(c, .8f), pow(d, .9f)}, {a, b, c, d},
                         kw::strided = true);

        s.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(float *, const float *, const float *, const float *)>(s.jit_lookup("cfunc"));

        const std::vector<float> ins{.1f, .2f, .3f, .4f};
        std::vector<float> outs(4u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::pow(.1f, .6f)));
        REQUIRE(outs[1] == approximately(std::pow(.2f, .7f)));
        REQUIRE(outs[2] == approximately(std::pow(.3f, .8f)));
        REQUIRE(outs[3] == approximately(std::pow(.4f, .9f)));

#if defined(HEYOKA_WITH_SLEEF)

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@llvm.pow.f32", boost::is_iequal()));
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

#if LLVM_VERSION_MAJOR >= 16

        if (tf.aarch64) {
            REQUIRE(count == 5u);
        }

#endif

        // NOTE: currently no auto-vectorization happens on ppc64 due apparently
        // to the way the target machine is being set up by orc/lljit (it works
        // fine with the opt tool). When this is resolved, we can test ppc64 too.

        // if (tf.vsx) {
        //     REQUIRE(count == 5u);
        // }

#endif
    }
}
