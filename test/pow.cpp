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
#include <heyoka/math/log.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/square.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

#if defined(HEYOKA_HAVE_REAL128) || defined(HEYOKA_HAVE_REAL)

using namespace mppp::literals;

#endif

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

TEST_CASE("pow expo 2")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 2.) == square(x));
    REQUIRE(heyoka::pow(x, 2.l) == square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 2._rq) == square(x));

#endif

    REQUIRE(heyoka::pow(x, 2.1) != square(x));
    REQUIRE(heyoka::pow(x, 2.1l) != square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 21._rq) != square(x));

#endif
}

TEST_CASE("pow expo 3")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 3.) == square(x) * x);
    REQUIRE(heyoka::pow(x, 3.l) == square(x) * x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 3._rq) == square(x) * x);

#endif

    REQUIRE(heyoka::pow(x, 3.1) != square(x) * x);
    REQUIRE(heyoka::pow(x, 3.1l) != square(x) * x);

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 31._rq) != square(x) * x);

#endif
}

TEST_CASE("pow expo 4")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, 4.) == square(x) * square(x));
    REQUIRE(heyoka::pow(x, 4.l) == square(x) * square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 4._rq) == square(x) * square(x));

#endif

    REQUIRE(heyoka::pow(x, 4.1) != square(x) * square(x));
    REQUIRE(heyoka::pow(x, 4.1l) != square(x) * square(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, 41._rq) != square(x) * square(x));

#endif

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(heyoka::pow(x, 1.1_r256) != square(x) * square(x));
    REQUIRE(heyoka::pow(x, 1.1_r256) == heyoka::pow(x, expression{1.1_r256}));

#endif
}

TEST_CASE("pow expo .5")
{
    auto x = "x"_var;

    REQUIRE(heyoka::pow(x, .5) == sqrt(x));
    REQUIRE(heyoka::pow(x, .5l) == sqrt(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, .5_rq) == sqrt(x));

#endif

    REQUIRE(heyoka::pow(x, .51) != sqrt(x));
    REQUIRE(heyoka::pow(x, .51l) != sqrt(x));

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(heyoka::pow(x, .51_rq) != sqrt(x));

#endif
}

TEST_CASE("powi")
{
    auto x = "x"_var;

    REQUIRE(powi(x, 0) == 1_dbl);
    REQUIRE(powi(x + 1., 1) == x + 1.);
    REQUIRE(powi(x + 1., 2) == square(x + 1.));
    REQUIRE(powi(x + 1., 3) == square(x + 1.) * (x + 1.));
    REQUIRE(powi(x + 1., 4) == square(x + 1.) * square(x + 1.));
    REQUIRE(powi(x + 1., 5) == square(x + 1.) * square(x + 1.) * (x + 1.));
    REQUIRE(powi(x + 1., 6) == square(x + 1.) * square(x + 1.) * square(x + 1.));
    REQUIRE(powi(x + 1., 7) == square(x + 1.) * square(x + 1.) * (square(x + 1.) * (x + 1.)));
}

TEST_CASE("pow diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(pow(3_dbl, x * x + y), "x") == (pow(3_dbl, x * x + y) * log(3_dbl)) * (2_dbl * x));
    REQUIRE(diff(pow(x * x + y, 1.2345_dbl), "y") == 1.2345_dbl * pow(x * x + y, 1.2345_dbl - 1_dbl));

    REQUIRE(diff(pow(3_dbl, par[0] * par[0] + y), par[0])
            == (pow(3_dbl, par[0] * par[0] + y) * log(3_dbl)) * (2_dbl * par[0]));
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

            add_cfunc<fp_t>(s, "cfunc", {pow(x, y), pow(x, par[0]), pow(x, 3. / 2_dbl)}, kw::batch_size = batch_size,
                            kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.pow_approx."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(pow(ins[i], ins[i + batch_size]), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(pow(ins[i], pars[i]), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pow(ins[i], fp_t(3) / 2), fp_t(100)));
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

            add_cfunc<mppp::real>(s, "cfunc", {pow(x, y), pow(x, par[0]), pow(x, 3. / 2_dbl)},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{"1.1", prec}, mppp::real{"2.1", prec}};
            const std::vector pars{mppp::real{"3.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == pow(ins[i], ins[i + 1u]));
            REQUIRE(outs[i + 1u] == pow(ins[i], pars[i]));
            REQUIRE(outs[i + 2u * 1u] == pow(ins[i], 3. / 2));
        }
    }
}

#endif

TEST_CASE("pow const fold")
{
    REQUIRE(pow(1.1_dbl, 2.2_dbl) == expression{std::pow(1.1, 2.2)});

#if defined(HEYOKA_HAVE_REAL)

    REQUIRE(pow(2_dbl, mppp::real{"1.1", 123}) == expression{mppp::pow(2., mppp::real{"1.1", 123})});

#endif
}
