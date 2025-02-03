// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/asinh.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

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

// NOTE: this wrapper is here only to ease the transition
// of old test code to the new implementation of square
// as a special case of multiplication.
auto square_wrapper(const expression &x)
{
    return pow(x, 2_dbl);
}

TEST_CASE("asinh diff var")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(asinh(square_wrapper(x) - y), x)
            == pow(square_wrapper(square_wrapper(x) - y) + 1., -.5) * (2_dbl * x));
    REQUIRE(diff(asinh(square_wrapper(x) + y), y) == pow(square_wrapper(square_wrapper(x) + y) + 1., -.5));
}

TEST_CASE("asinh diff par")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(asinh(square_wrapper(par[0]) - y), par[0])
            == pow(square_wrapper(square_wrapper(par[0]) - y) + 1., -.5) * (2_dbl * par[0]));
    REQUIRE(diff(asinh(square_wrapper(x) + par[1]), par[1])
            == pow(square_wrapper(square_wrapper(x) + par[1]) + 1., -.5));
}

TEST_CASE("asinh s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = asinh(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == asinh(x));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::asinh;

        using fp_t = decltype(fp_x);

        auto x = make_vars("x");

        std::uniform_real_distribution<double> rdist(-1., 1.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 3u);
            ins.resize(batch_size);
            pars.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {asinh(x), asinh(expression{fp_t(-.5)}), asinh(par[0])}, {x},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.asinh."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(asinh(ins[i]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(asinh(static_cast<fp_t>(-.5)), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(asinh(pars[i]), fp_t(100)));
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
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {asinh(x), asinh(expression{mppp::real{1.5, prec}}), asinh(par[0])}, {x},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{"1.7", prec}};
            const std::vector pars{mppp::real{"2.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == asinh(ins[i]));
            REQUIRE(outs[i + 1u] == asinh(mppp::real{1.5, prec}));
            REQUIRE(outs[i + 2u * 1u] == asinh(pars[i]));
        }
    }
}

#endif

// Tests to check vectorisation via the vector-function-abi-variant machinery.
TEST_CASE("vfabi double")
{
    for (auto fast_math : {false, true}) {
        llvm_state s{kw::slp_vectorize = true, kw::fast_math = fast_math};

        auto [a, b] = make_vars("a", "b");

        add_cfunc<double>(s, "cfunc", {asinh(a), asinh(b)}, {a, b});
        add_cfunc<double>(s, "cfuncs", {asinh(a), asinh(b)}, {a, b}, kw::strided = true);

        s.compile();

        auto *cf_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            s.jit_lookup("cfunc"));

        const std::vector ins{1.1, 1.2};
        std::vector<double> outs(2u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::asinh(1.1)));
        REQUIRE(outs[1] == approximately(std::asinh(1.2)));

        // NOTE: autovec with external scalar functions seems to work
        // only since LLVM 16.
#if defined(HEYOKA_WITH_SLEEF) && LLVM_VERSION_MAJOR >= 16

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@asinh", boost::is_iequal()));
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

        add_cfunc<float>(s, "cfunc", {asinh(a), asinh(b), asinh(c), asinh(d)}, {a, b, c, d});
        add_cfunc<float>(s, "cfuncs", {asinh(a), asinh(b), asinh(c), asinh(d)}, {a, b, c, d}, kw::strided = true);

        s.compile();

        auto *cf_ptr
            = reinterpret_cast<void (*)(float *, const float *, const float *, const float *)>(s.jit_lookup("cfunc"));

        const std::vector<float> ins{1.1f, 1.2f, 1.3f, 1.4f};
        std::vector<float> outs(4u, 0.);

        cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

        REQUIRE(outs[0] == approximately(std::asinh(1.1f)));
        REQUIRE(outs[1] == approximately(std::asinh(1.2f)));
        REQUIRE(outs[2] == approximately(std::asinh(1.3f)));
        REQUIRE(outs[3] == approximately(std::asinh(1.4f)));

        // NOTE: autovec with external scalar functions seems to work
        // only since LLVM 16.
#if defined(HEYOKA_WITH_SLEEF) && LLVM_VERSION_MAJOR >= 16

        const auto &tf = detail::get_target_features();

        auto ir = s.get_ir();

        using string_find_iterator = boost::find_iterator<std::string::iterator>;

        auto count = 0u;
        for (auto it = boost::make_find_iterator(ir, boost::first_finder("@asinhf", boost::is_iequal()));
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

        // NOTE: LLVM16 is currently the version tested in the CI on arm64.
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
