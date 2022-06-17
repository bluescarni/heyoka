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

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
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

TEST_CASE("sin diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(sin(x * x - y), x) == cos(x * x - y) * (2. * x));
    REQUIRE(diff(sin(x * x - y), y) == -cos(x * x - y));

    REQUIRE(diff(sin(par[0] * par[0] - y), par[0]) == cos(par[0] * par[0] - y) * (2. * par[0]));
    REQUIRE(diff(sin(x * x - par[1]), par[1]) == -cos(x * x - par[1]));
}

TEST_CASE("sin s11n")
{
    std::stringstream ss;

    auto [x] = make_vars("x");

    auto ex = sin(x);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sin(x));
}

TEST_CASE("sin number simpl")
{
    using std::sin;

    auto [x] = make_vars("x");

    REQUIRE(sin(x * 0.) == 0_dbl);
    REQUIRE(sin(0.123_dbl) == expression{sin(0.123)});
    REQUIRE(sin(-0.123_ldbl) == expression{sin(-0.123l)});

#if defined(HEYOKA_HAVE_REAL128)
    using namespace mppp::literals;

    REQUIRE(sin(-0.123_f128) == expression{sin(-0.123_rq)});
#endif
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using std::sin;

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

            add_cfunc<fp_t>(s, "cfunc", {sin(x), sin(expression{fp_t(.5)}), sin(par[0])}, batch_size, high_accuracy,
                            compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sin."));
            }

            s.compile();

            auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(sin(ins[i]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(sin(static_cast<fp_t>(.5)), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(sin(pars[i]), fp_t(100)));
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
