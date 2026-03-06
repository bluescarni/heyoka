// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
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

TEST_CASE("time stream")
{
    std::ostringstream oss;

    oss << heyoka::time;

    REQUIRE(oss.str() == "t");
}

TEST_CASE("time diff")
{
    REQUIRE(diff(heyoka::time, "x") == 0_dbl);

    auto x = "x"_var;

    REQUIRE(diff(heyoka::time * cos(2. * x + 2. * heyoka::time), "x")
            == heyoka::time * (2. * -sin(2. * x + 2. * heyoka::time)));

    REQUIRE(diff(heyoka::time * cos(2. * par[0] + 2. * heyoka::time), par[0])
            == heyoka::time * (2. * -sin(2. * par[0] + 2. * heyoka::time)));
}

TEST_CASE("time s11n")
{
    std::stringstream ss;

    auto x = make_vars("x");

    auto ex = heyoka::time + x;

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == heyoka::time + x);
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        using std::cos;

        auto x = make_vars("x");

        std::uniform_real_distribution<double> rdist(-1., 1.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        std::vector<fp_t> outs, ins, pars, time;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 3u);
            ins.resize(batch_size);
            pars.resize(batch_size);
            time.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);
            std::generate(time.begin(), time.end(), gen);

            cfunc<fp_t> cf(
                {cos(x - heyoka::time), heyoka::time + cos(expression{fp_t(-.5)}), cos(par[0]) * heyoka::time}, {x},
                kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                kw::compact_mode = compact_mode, kw::opt_level = opt_level);

            if (opt_level == 0u && compact_mode) {
                const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                REQUIRE(std::ranges::any_of(irs, [](const auto &ir) {
                    return boost::contains(ir, "heyoka.llvm_c_eval.time.");
                }));
            }

            cf(mdspan<fp_t, dextents<std::size_t, 2>>(outs.data(), 3u, batch_size),
               mdspan<const fp_t, dextents<std::size_t, 2>>(ins.data(), 1u, batch_size),
               kw::pars = mdspan<const fp_t, dextents<std::size_t, 2>>(pars.data(), 1u, batch_size),
               kw::time = mdspan<const fp_t, dextents<std::size_t, 1>>(time.data(), batch_size));

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(cos(ins[i] - time[i]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(time[i] + cos(static_cast<fp_t>(-.5)), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(cos(pars[i]) * time[i], fp_t(100)));
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
            cfunc<mppp::real> cf(
                {cos(x - heyoka::time), heyoka::time + cos(expression{mppp::real{-.5, prec}}),
                 cos(par[0]) * heyoka::time},
                {x}, kw::compact_mode = compact_mode, kw::prec = prec,
                kw::opt_level = opt_level);

            const std::vector ins{mppp::real{".7", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            const std::vector time{mppp::real{".3", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf(outs, ins, kw::pars = pars, kw::time = time[0]);

            REQUIRE(outs[0] == cos(ins[0] - time[0]));
            REQUIRE(outs[1] == time[0] + cos(mppp::real{-.5, prec}));
            REQUIRE(outs[2] == cos(pars[0]) * time[0]);
        }
    }
}

#endif
