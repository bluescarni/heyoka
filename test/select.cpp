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
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/logical.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/time.hpp>
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
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("basic")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(expression{func{detail::select_impl{}}} == expression{func{detail::select_impl{0_dbl, 0_dbl, 0_dbl}}});

    // A couple of tests for the numeric overloads.
    REQUIRE(select(1., x, 2.) == select(1_dbl, x, 2_dbl));
    REQUIRE(select(x, par[0], 2.f) == select(x, par[0], 2_flt));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(select(mppp::real128{"3.1"}, x, mppp::real128{"2.1"}) == select(3.1_f128, x, 2.1_f128));

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(select(mppp::real{"3.1", 14}, x, mppp::real{"2.1", 14})
            == select(expression{mppp::real{"3.1", 14}}, x, expression{mppp::real{"2.1", 14}}));

#endif
}

TEST_CASE("stream")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;
        oss << select(x, y, z);
        REQUIRE(oss.str() == "select(x, y, z)");
    }

    {
        std::ostringstream oss;
        oss << select(x, y + z, y - z);
        REQUIRE(oss.str() == "select(x, (y + z), (y - z))");
    }
}

TEST_CASE("diff")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(diff(select(x, y * z, y / z), x) == 0_dbl);
    REQUIRE(diff(select(x, y * z, y / z), y) == ((select(x, 1_dbl, 0_dbl) * z) + (select(x, 0_dbl, 1_dbl) / z)));
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = select(x, y * z, y / z);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 1_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == select(x, y * z, y / z));
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

            outs.resize(batch_size * 2u);
            ins.resize(batch_size * 2u);
            pars.resize(batch_size);
            time.resize(batch_size);

            std::generate(ins.begin(), ins.end(), gen);
            std::generate(pars.begin(), pars.end(), gen);
            std::generate(time.begin(), time.end(), gen);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc",
                            {select(logical_and({gt(x, par[0]), lt(x, 2. * y)}), x * x, y * y),
                             select(logical_or({lte(x, 0_dbl), lt(y, heyoka::time)}), x / y, y / x)},
                            {x, y}, kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.select."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i]
                        == ((ins[i] > pars[i] && ins[i] < 2 * ins[i + batch_size])
                                ? ins[i] * ins[i]
                                : ins[i + batch_size] * ins[i + batch_size]));
                REQUIRE(outs[i + batch_size]
                        == ((ins[i] <= 0 || ins[i + batch_size] < time[i]) ? ins[i] / ins[i + batch_size]
                                                                           : ins[i + batch_size] / ins[i]));
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
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc",
                                  {select(logical_and({gt(x, par[0]), lt(x, 2. * y)}), x * x, y * y),
                                   select(logical_or({lte(x, 0_dbl), lt(y, heyoka::time)}), x / y, y / x)},
                                  {x, y}, kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{".7", prec}, mppp::real{"-.1", prec}};
            const std::vector pars{mppp::real{"0", prec}};
            const std::vector time{mppp::real{".3", prec}};
            std::vector<mppp::real> outs(8u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            auto i = 0u;
            auto batch_size = 1u;
            REQUIRE(outs[i] == mppp::real{"-.1", prec} * mppp::real{"-.1", prec});
            REQUIRE(outs[i + batch_size] == mppp::real{".7", prec} / mppp::real{"-.1", prec});
        }
    }
}

#endif

TEST_CASE("taylor_adaptive")
{
    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            auto ta1 = taylor_adaptive{
                {prime(x) = v, prime(v) = -sin(x)}, {1.23, 0.}, kw::compact_mode = cm, kw::opt_level = opt_level};

            auto ta2 = taylor_adaptive{
                {prime(x) = v, prime(v) = -(1. + select(gt(x, 1.24_dbl), 1. - par[0], 0_dbl)) * sin(x)},
                {1.23, 0.},
                kw::compact_mode = cm,
                kw::opt_level = opt_level};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.var_var_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));

            ta1 = taylor_adaptive{
                {prime(x) = v, prime(v) = -2. * sin(x)}, {1.23, 0.}, kw::compact_mode = cm, kw::opt_level = opt_level};
            ta2 = taylor_adaptive{{prime(x) = v, prime(v) = -(1. + select(lt(x, 1.24_dbl), par[0], 0_dbl)) * sin(x)},
                                  {1.23, 0.},
                                  kw::compact_mode = cm,
                                  kw::opt_level = opt_level,
                                  kw::pars = {1.}};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.var_par_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));

            ta1 = taylor_adaptive{
                {prime(x) = v, prime(v) = -2. * sin(x)}, {1.23, 0.}, kw::compact_mode = cm, kw::opt_level = opt_level};
            ta2 = taylor_adaptive{{prime(x) = v, prime(v) = -(1. + select(par[0], par[0], 0_dbl)) * sin(x)},
                                  {1.23, 0.},
                                  kw::compact_mode = cm,
                                  kw::opt_level = opt_level,
                                  kw::pars = {1.}};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.par_par_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
        }
    }
}

TEST_CASE("taylor_adaptive_batch")
{
    auto [x, v] = make_vars("x", "v");

    for (auto opt_level : {0u, 3u}) {
        for (auto cm : {false, true}) {
            auto ta1 = taylor_adaptive_batch{{prime(x) = v, prime(v) = -sin(x)},
                                             {1.23, 1.22, 0., 0.},
                                             2u,
                                             kw::compact_mode = cm,
                                             kw::opt_level = opt_level};

            auto ta2 = taylor_adaptive_batch{
                {prime(x) = v, prime(v) = -(1. + select(gt(x, 1.24_dbl), 1. - par[0], 0_dbl)) * sin(x)},
                {1.23, 1.22, 0., 0.},
                2u,
                kw::compact_mode = cm,
                kw::opt_level = opt_level};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.var_var_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
            REQUIRE(ta1.get_state()[2] == approximately(ta2.get_state()[2]));
            REQUIRE(ta1.get_state()[3] == approximately(ta2.get_state()[3]));

            ta1 = taylor_adaptive_batch{{prime(x) = v, prime(v) = -2. * sin(x)},
                                        {1.23, 1.22, 0., 0.},
                                        2u,
                                        kw::compact_mode = cm,
                                        kw::opt_level = opt_level};
            ta2 = taylor_adaptive_batch{
                {prime(x) = v, prime(v) = -(1. + select(lt(x, 1.24_dbl), par[0], 0_dbl)) * sin(x)},
                {1.23, 1.22, 0., 0.},
                2u,
                kw::compact_mode = cm,
                kw::opt_level = opt_level,
                kw::pars = {1., 1.}};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.var_par_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
            REQUIRE(ta1.get_state()[2] == approximately(ta2.get_state()[2]));
            REQUIRE(ta1.get_state()[3] == approximately(ta2.get_state()[3]));

            ta1 = taylor_adaptive_batch{{prime(x) = v, prime(v) = -2. * sin(x)},
                                        {1.23, 1.22, 0., 0.},
                                        2u,
                                        kw::compact_mode = cm,
                                        kw::opt_level = opt_level};
            ta2 = taylor_adaptive_batch{{prime(x) = v, prime(v) = -(1. + select(par[0], par[0], 0_dbl)) * sin(x)},
                                        {1.23, 1.22, 0., 0.},
                                        2u,
                                        kw::compact_mode = cm,
                                        kw::opt_level = opt_level,
                                        kw::pars = {1., 1.}};

            if (opt_level == 0u && cm) {
                REQUIRE(boost::contains(ta2.get_llvm_state().get_ir(), "heyoka.taylor_c_diff.select.par_par_num."));
            }

            ta1.propagate_until(5.);
            ta2.propagate_until(5.);

            REQUIRE(ta1.get_state()[0] == approximately(ta2.get_state()[0]));
            REQUIRE(ta1.get_state()[1] == approximately(ta2.get_state()[1]));
            REQUIRE(ta1.get_state()[2] == approximately(ta2.get_state()[2]));
            REQUIRE(ta1.get_state()[3] == approximately(ta2.get_state()[3]));
        }
    }
}
