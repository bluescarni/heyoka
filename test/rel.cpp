// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
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
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/time.hpp>
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
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("stream")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::ostringstream oss;
        oss << eq(x, y);
        REQUIRE(oss.str() == "(x == y)");
    }

    {
        std::ostringstream oss;
        oss << neq(x, y);
        REQUIRE(oss.str() == "(x != y)");
    }

    {
        std::ostringstream oss;
        oss << lt(x, y);
        REQUIRE(oss.str() == "(x < y)");
    }

    {
        std::ostringstream oss;
        oss << gt(x, y);
        REQUIRE(oss.str() == "(x > y)");
    }

    {
        std::ostringstream oss;
        oss << lte(x, y);
        REQUIRE(oss.str() == "(x <= y)");
    }

    {
        std::ostringstream oss;
        oss << gte(x, y);
        REQUIRE(oss.str() == "(x >= y)");
    }
}

TEST_CASE("diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(eq(x, y), "x") == 0_dbl);
    REQUIRE(diff(neq(x, y), "y") == 0_dbl);
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = eq(x, y);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = neq(x, y);

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == eq(x, y));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-1., 1.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars, time;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 8u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);
            time.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);
            std::generate(time.begin(), time.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc",
                            {eq(x, y), neq(x, par[0]), lt(y, 1_dbl), gt(x + y, y - x), lte(x * x, heyoka::time),
                             gte(par[0], .4_dbl), lte(x, x), gte(y, y)},
                            {x, y}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_eq."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_neq."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_lt."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_gt."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_lte."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.rel_gte."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == (ins[i] == ins[i + batch_size]));
                REQUIRE(outs[i + batch_size] == (ins[i] != pars[i]));
                REQUIRE(outs[i + 2u * batch_size] == (ins[i + batch_size] < 1));
                REQUIRE(outs[i + 3u * batch_size] == ((ins[i] + ins[i + batch_size]) > (ins[i + batch_size] - ins[i])));
                REQUIRE(outs[i + 4u * batch_size] == ((ins[i] * ins[i]) <= time[i]));
                REQUIRE(outs[i + 5u * batch_size] == (pars[i] >= .4));
                REQUIRE(outs[i + 6u * batch_size] == true);
                REQUIRE(outs[i + 7u * batch_size] == true);
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

            add_cfunc<mppp::real>(s, "cfunc",
                                  {eq(x, y), neq(x, par[0]), lt(y, 1_dbl), gt(x + y, y - x), lte(x * x, heyoka::time),
                                   gte(par[0], .4_dbl), lte(x, x), gte(y, y)},
                                  {x, y}, kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{".7", prec}, mppp::real{"-.1", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            const std::vector time{mppp::real{".3", prec}};
            std::vector<mppp::real> outs(8u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            auto i = 0u;
            auto batch_size = 1u;
            REQUIRE(outs[i] == (ins[i] == ins[i + batch_size]));
            REQUIRE(outs[i + batch_size] == (ins[i] != pars[i]));
            REQUIRE(outs[i + 2u * batch_size] == (ins[i + batch_size] < 1));
            REQUIRE(outs[i + 3u * batch_size] == ((ins[i] + ins[i + batch_size]) > (ins[i + batch_size] - ins[i])));
            REQUIRE(outs[i + 4u * batch_size] == ((ins[i] * ins[i]) <= time[i]));
            REQUIRE(outs[i + 5u * batch_size] == (pars[i] >= .4));
            REQUIRE(outs[i + 6u * batch_size] == true);
            REQUIRE(outs[i + 7u * batch_size] == true);
        }
    }
}

#endif
