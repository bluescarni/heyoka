// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <fmt/core.h>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/div.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/prod.hpp>
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
#if LLVM_VERSION_MAJOR >= 13 && LLVM_VERSION_MAJOR <= 16
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::prod_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "prod");
    }

    {
        detail::prod_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::prod_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::prod_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{}", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{}", -1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_ldbl, x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-x");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y, x + y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y * (x + y))");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y, pow(x, y)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y * x**y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y, pow(x, 2_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-(x * y * x**2.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y * z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y * z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, -1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("({} / (x * y))", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, -1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("-({} / (x * y))", 1_dbl));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x * y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x / y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-(x / y)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -2_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "(x / y**2.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -2_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-(x / y**2.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "((x * y) / z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -1_dbl)});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "-((x * y) / z)");
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "((x * y) / z**3.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, 1_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-((x * y) / z**3.000"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({pow(x, 1_dbl), pow(y, -2_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "(x / (y**2.0000"));
        REQUIRE(boost::contains(oss.str(), "z**3.0000"));
        REQUIRE(boost::contains(oss.str(), "0000))"));
    }

    {
        std::ostringstream oss;

        detail::prod_impl ss({-1_dbl, pow(x, 1_dbl), pow(y, -2_dbl), pow(z, -3_dbl)});
        ss.to_stream(oss);

        REQUIRE(boost::contains(oss.str(), "-(x / (y**2.0000"));
        REQUIRE(boost::contains(oss.str(), "z**3.0000"));
        REQUIRE(boost::contains(oss.str(), "0000))"));
    }

    {
        std::ostringstream oss;

        oss << prod({pow(pow(x, y), -2_dbl), z});

        REQUIRE(boost::starts_with(oss.str(), "(z / x**(2.00000000"));
        REQUIRE(boost::ends_with(oss.str(), "0000000000 * y))"));
    }

    {
        std::ostringstream oss;

        oss << prod({x, pow(x, expression{func{detail::prod_impl({y})}})});

        REQUIRE(oss.str() == "(x * x**y)");
    }
}

TEST_CASE("args simpl")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    // Flattening.
    REQUIRE(prod({prod({x, y}), z}) == prod({x, y, z}));

    // Gathering of common bases with numerical exponents.
    REQUIRE(prod({x, pow(y, 3.), pow(x, 2.), pow(y, 4.)}) == prod({pow(x, 3.), pow(y, 7.)}));
    REQUIRE(prod({pow(y, 3.), pow(x, 2.), x, pow(y, 4.)}) == prod({pow(x, 3.), pow(y, 7.)}));

    // Constant folding.
    REQUIRE(prod({3_dbl, 4_dbl}) == 12_dbl);
    REQUIRE(prod({3_dbl, 4_dbl, x, -2_dbl}) == prod({x, -24_dbl}));
    REQUIRE(prod({.5_dbl, 2_dbl, x}) == x);
    REQUIRE(prod({pow(y, 3.), pow(x, -1.), x, pow(y, -3.)}) == 1_dbl);
    REQUIRE(prod({pow(y, 3.), pow(x, -1.), x, pow(y, -3.), 0_dbl}) == 0_dbl);

    // Special cases.
    REQUIRE(prod({}) == 1_dbl);
    REQUIRE(prod({x}) == x);

    // Sorting.
    REQUIRE(prod({y, z, x, 1_dbl}) == prod({1_dbl, x, y, z}));
}

TEST_CASE("diff")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE(diff(prod({}), x) == 0_dbl);

    REQUIRE(diff(prod({3_dbl}), x) == 0_dbl);
    REQUIRE(diff(prod({x}), x) == 1_dbl);
    REQUIRE(diff(prod({x, 3_dbl}), x) == 3_dbl);
    REQUIRE(diff(prod({x, y, 3_dbl}), x) == prod({3_dbl, y}));
    REQUIRE(diff(prod({x, y, 3_dbl}), y) == prod({3_dbl, x}));
    REQUIRE(diff(prod({x, y, 3_dbl}), z) == 0_dbl);
    REQUIRE(diff(prod({x, y, 3_dbl}), par[0]) == 0_dbl);
}

TEST_CASE("s11n")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = prod({x, y});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == prod({x, y}));
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-1., 1.);

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

            add_cfunc<fp_t>(s, "cfunc",
                            {prod({expression{fp_t{-1}}, x}), prod({par[0], x, y}), prod({expression{fp_t{5}}, x, y}),
                             // NOTE: test a couple of corner cases as well.
                             expression{func{detail::prod_impl({x})}}, expression{func{detail::prod_impl{}}}},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.prod."));
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.prod_neg."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(-ins[i], fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(pars[i] * ins[i] * ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(fp_t(5) * ins[i] * ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + 3u * batch_size] == ins[i]);
                REQUIRE(outs[i + 4u * batch_size] == fp_t(1));
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
                                  {prod({expression{mppp::real{-1, prec}}, x}), prod({par[0], x, y}),
                                   prod({expression{mppp::real{5, prec}}, x, y})},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{".7", prec}, mppp::real{"-.3", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == approximately(-ins[i]));
            REQUIRE(outs[i + 1u] == approximately(pars[i] * ins[i] * ins[i + 1u]));
            REQUIRE(outs[i + 2u] == approximately(mppp::real(5, prec) * ins[i] * ins[i + 1u]));
        }
    }
}

#endif

TEST_CASE("prod split")
{
    auto prod_wrapper = [](std::vector<expression> v) { return expression{func{detail::prod_impl(std::move(v))}}; };

    auto [x, y] = make_vars("x", "y");

    auto s = prod_wrapper({x, x, x, x, y, y, y});

    auto ss1 = detail::prod_split(s, 2u);

    REQUIRE(ss1
            == prod_wrapper(
                {prod_wrapper({prod_wrapper({x, x}), prod_wrapper({x, x})}), prod_wrapper({prod_wrapper({y, y}), y})}));

    ss1 = detail::prod_split(s, 3u);

    REQUIRE(ss1 == prod_wrapper({prod_wrapper({x, x, x}), prod_wrapper({x, y, y}), y}));

    ss1 = detail::prod_split(s, 4u);

    REQUIRE(ss1 == prod_wrapper({prod_wrapper({x, x, x, x}), prod_wrapper({y, y, y})}));

    ss1 = detail::prod_split(s, 8u);

    REQUIRE(s == ss1);

    ss1 = detail::prod_split(s, 9u);

    REQUIRE(s == ss1);

    ss1 = detail::prod_split(s, 10u);

    REQUIRE(s == ss1);

    // Check with a non-prod expression.
    auto ns = x + y;
    REQUIRE(detail::prod_split(ns, 8u) == ns);
    REQUIRE(std::get<func>(detail::prod_split(ns, 8u).value()).get_ptr() == std::get<func>(ns.value()).get_ptr());

    // A cfunc test with nested prods.
    std::vector<expression> x_vars;
    for (auto i = 0; i < 10; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
    }

    s = prod_wrapper({x_vars[0], x_vars[1], x_vars[2], x_vars[3], x_vars[4], x_vars[5], x_vars[6], x_vars[7], x_vars[8],
                      x_vars[9], cos(prod_wrapper(x_vars))});

    llvm_state ls;
    const auto dc = add_cfunc<double>(ls, "cfunc", {s});

    for (const auto &ex : dc) {
        if (const auto *fptr = std::get_if<func>(&ex.value())) {
            if (const auto *sptr = fptr->extract<detail::prod_impl>()) {
                REQUIRE(sptr->args().size() <= 8u);
            }
        }
    }

    ls.optimise();
    ls.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ls.jit_lookup("cfunc"));

    std::vector<double> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double output = 0;

    cf_ptr(&output, inputs.data(), nullptr, nullptr);

    REQUIRE(output
            == approximately((1. * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
                             * std::cos(1. * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)));
}

TEST_CASE("prod_to_div")
{
    auto [x, y] = make_vars("x", "y");

    // cfunc.
    auto ret = detail::prod_to_div_llvm_eval({prod({x, pow(y, -1_dbl)})});

    REQUIRE(ret.size() == 1u);
    REQUIRE(ret[0] == detail::div(x, y));

    ret = detail::prod_to_div_llvm_eval({prod({2_dbl, pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(2_dbl, y));

    ret = detail::prod_to_div_llvm_eval({prod({2_dbl, cos(x), pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(prod({2_dbl, cos(x)}), y));

    ret = detail::prod_to_div_llvm_eval({prod({2_dbl, cos(x), pow(y, -1_dbl), pow(x, -1.5_dbl)})});
    REQUIRE(ret[0] == detail::div(prod({2_dbl, cos(x)}), prod({y, pow(x, 1.5_dbl)})));

    ret = detail::prod_to_div_llvm_eval({cos(prod({2_dbl, cos(x), pow(y, -1_dbl), pow(x, -1.5_dbl)}))});
    REQUIRE(ret[0] == cos(detail::div(prod({2_dbl, cos(x)}), prod({y, pow(x, 1.5_dbl)}))));

    ret = detail::prod_to_div_llvm_eval({prod({pow(x, -.5_dbl), pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(1_dbl, prod({pow(x, .5_dbl), y})));

    ret = detail::prod_to_div_llvm_eval({prod({pow(x, -.5_dbl), pow(y, 1.5_dbl)})});
    REQUIRE(ret[0] == detail::div(pow(y, 1.5_dbl), pow(x, .5_dbl)));

    ret = detail::prod_to_div_llvm_eval({prod({pow(x, -.5_dbl), pow(y, 2_dbl)})});
    REQUIRE(ret[0] == detail::div(pow(y, 2_dbl), pow(x, .5_dbl)));

    ret = detail::prod_to_div_llvm_eval({prod({x, y})});
    REQUIRE(ret[0] == prod({x, y}));

    ret = detail::prod_to_div_llvm_eval(
        {prod({2_dbl, cos(prod({pow(x, -.5_dbl), pow(y, -1_dbl)})), sin(prod({pow(x, -.5_dbl), pow(y, -1_dbl)}))})});
    REQUIRE(ret[0]
            == prod({2_dbl, cos(detail::div(1_dbl, prod({pow(x, .5_dbl), pow(y, 1_dbl)}))),
                     sin(detail::div(1_dbl, prod({pow(x, .5_dbl), pow(y, 1_dbl)})))}));

    // Taylor diff.
    ret = detail::prod_to_div_taylor_diff({prod({x, pow(y, -1_dbl)})});

    REQUIRE(ret.size() == 1u);
    REQUIRE(ret[0] == detail::div(x, y));

    ret = detail::prod_to_div_taylor_diff({prod({2_dbl, pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(2_dbl, y));

    ret = detail::prod_to_div_taylor_diff({prod({2_dbl, cos(x), pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(prod({2_dbl, cos(x)}), y));

    ret = detail::prod_to_div_taylor_diff({prod({2_dbl, cos(x), pow(y, -1_dbl), pow(x, -1.5_dbl)})});
    REQUIRE(ret[0] == detail::div(prod({2_dbl, cos(x), pow(x, -1.5_dbl)}), prod({y})));

    ret = detail::prod_to_div_taylor_diff({cos(prod({2_dbl, cos(x), pow(y, -1_dbl), pow(x, -1.5_dbl)}))});
    REQUIRE(ret[0] == cos(detail::div(prod({2_dbl, cos(x), pow(x, -1.5_dbl)}), prod({y}))));

    ret = detail::prod_to_div_taylor_diff({prod({pow(x, -.5_dbl), pow(y, -1_dbl)})});
    REQUIRE(ret[0] == detail::div(pow(x, -.5_dbl), prod({y})));

    ret = detail::prod_to_div_taylor_diff({prod({pow(x, -.5_dbl), pow(y, 1.5_dbl)})});
    REQUIRE(ret[0] == prod({pow(x, -.5_dbl), pow(y, 1.5_dbl)}));

    ret = detail::prod_to_div_taylor_diff({prod({pow(x, -.5_dbl), pow(y, 2_dbl)})});
    REQUIRE(ret[0] == prod({pow(x, -.5_dbl), pow(y, 2_dbl)}));

    ret = detail::prod_to_div_taylor_diff({prod({x, y})});
    REQUIRE(ret[0] == prod({x, y}));

    ret = detail::prod_to_div_taylor_diff(
        {prod({2_dbl, cos(prod({pow(x, -.5_dbl), pow(y, -1_dbl)})), sin(prod({pow(x, -.5_dbl), pow(y, -1_dbl)}))})});
    REQUIRE(ret[0]
            == prod({2_dbl, cos(detail::div(pow(x, -.5_dbl), prod({pow(y, 1_dbl)}))),
                     sin(detail::div(pow(x, -.5_dbl), prod({pow(y, 1_dbl)})))}));
}
