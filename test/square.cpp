// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/square.hpp>
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
#if LLVM_VERSION_MAJOR >= 13 && LLVM_VERSION_MAJOR <= 16
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("square diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(square(x * x - y), "x") == 2_dbl * (x * x - y) * (2_dbl * x));
    REQUIRE(diff(square(x * x - y), "y") == -(2_dbl * (x * x - y)));

    REQUIRE(diff(square(par[0] * par[0] - y), par[0]) == 2_dbl * (par[0] * par[0] - y) * (2_dbl * par[0]));
    REQUIRE(diff(square(x * x - par[1]), par[1]) == -(2_dbl * (x * x - par[1])));
}

TEST_CASE("square s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = square(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == square(x));
}

TEST_CASE("square stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;
        oss << square(x);

        REQUIRE(oss.str() == "x**2");
    }

    {
        std::ostringstream oss;
        oss << square(x + y);

        REQUIRE(oss.str() == "(x + y)**2");
    }

    {
        std::ostringstream oss;
        oss << square(2_dbl);

        REQUIRE(oss.str() == "2.0000000000000000**2");
    }

    {
        std::ostringstream oss;
        oss << square(par[0]);

        REQUIRE(oss.str() == "p0**2");
    }

    {
        std::ostringstream oss;
        oss << square(x + par[0]);

        REQUIRE(oss.str() == "(x + p0)**2");
    }

    {
        std::ostringstream oss;
        oss << square(cos(x + par[0]));

        REQUIRE(oss.str() == "cos((x + p0))**2");
    }
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x] = make_vars("x");

        std::uniform_real_distribution<double> rdist(.1, 10.);

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

            add_cfunc<fp_t>(s, "cfunc", {square(x), square(expression{fp_t(.5)}), square(par[0])},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.square."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(ins[i] * ins[i], fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(static_cast<fp_t>(.5) * .5, fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pars[i] * pars[i], fp_t(100)));
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
    auto [x] = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {square(x), square(expression{.5}), square(par[0])},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{".7", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == ins[i] * ins[i]);
            REQUIRE(outs[i + 1u] == mppp::real{-.5, prec} * mppp::real{-.5, prec});
            REQUIRE(outs[i + 2u * 1u] == pars[i] * pars[i]);
        }
    }
}

#endif
