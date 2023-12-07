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
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>
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

#include <heyoka/detail/sub.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/prod.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum.hpp>
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

TEST_CASE("basic test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum");
    }

    {
        detail::sum_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("{}", 0_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y)");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x + y + z)");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({prod({x, -1_dbl}), y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(y + z - x)");
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({prod({x, -2_dbl}), y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("(y + z - ({} * x))", 2_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({prod({x, -2_dbl}), -3_dbl, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("(z - ({} * x) - {})", 2_dbl, 3_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({prod({x, -2_dbl}), -3_dbl});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("((-{} * x) - {})", 2_dbl, 3_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({-3_dbl, prod({x, -2_dbl})});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("(-{} - ({} * x))", 3_dbl, 2_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({-3_dbl, -2_dbl});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("(-{} - {})", 3_dbl, 2_dbl));
    }

    {
        std::ostringstream oss;

        detail::sum_impl ss({prod({-3_dbl, x}), prod({x, -2_dbl})});
        ss.to_stream(oss);

        REQUIRE(oss.str() == fmt::format("((-{} * x) - ({} * x))", 3_dbl, 2_dbl));
    }
}

TEST_CASE("diff test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        detail::sum_impl ss;
        REQUIRE(ss.gradient().empty());
    }

    {
        detail::sum_impl ss({x, y, z});
        REQUIRE(ss.gradient() == std::vector{1_dbl, 1_dbl, 1_dbl});
    }

    {
        REQUIRE(diff(sum({x, y, z}), "x") == 1_dbl);
        REQUIRE(diff(sum({x, x * x, z}), "x") == sum({1_dbl, 2_dbl * x}));
        REQUIRE(diff(sum({x, x * x, -z}), "z") == -1_dbl);
    }
}

TEST_CASE("sum function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum({}) == 0_dbl);
    REQUIRE(sum({x}) == x);
}

TEST_CASE("sum s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = sum({x, y, z});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sum({x, y, z}));
}

TEST_CASE("sum number compress")
{
    REQUIRE(sum({1_dbl, 2_dbl, 3_dbl}) == 6_dbl);
    REQUIRE(sum({1_dbl, -2_dbl, 1_dbl}) == 0_dbl);

    REQUIRE(sum({1_dbl, 2_dbl, "x"_var}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({2_dbl, "x"_var}) == sum({"x"_var, 2_dbl}));
    REQUIRE(sum({1_dbl, "x"_var, 2_dbl}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({1_dbl, 2_dbl, "x"_var}) == sum({"x"_var, 3_dbl}));
    REQUIRE(sum({"x"_var, 2_dbl}) == sum({"x"_var, 2_dbl}));
    REQUIRE(sum({-1_dbl, "x"_var, 2_dbl, -1_dbl}) == "x"_var);

    REQUIRE(sum({"y"_var, 1_dbl, -21_dbl, "x"_var}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({1_dbl, "y"_var, -21_dbl, "x"_var}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({"y"_var, 1_dbl, "x"_var, -21_dbl}) == sum({"y"_var, "x"_var, -20_dbl}));
    REQUIRE(sum({"x"_var, 1_dbl, "y"_var, -21_dbl}) == sum({"x"_var, "y"_var, -20_dbl}));
    REQUIRE(sum({"x"_var, 1_dbl, "y"_var, par[0]}) == sum({"x"_var, "y"_var, par[0], 1_dbl}));
    REQUIRE(sum({"x"_var, "y"_var, par[0]}) == sum({"x"_var, "y"_var, par[0]}));
    REQUIRE(sum({1_dbl, "y"_var, -21_dbl, "x"_var, 20_dbl}) == sum({"y"_var, "x"_var}));

    REQUIRE(std::get<func>(sum({"y"_var, 1_dbl, "x"_var, -21_dbl}).value()).args()
            == std::vector{-20_dbl, "x"_var, "y"_var});
}

TEST_CASE("cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-10., 10.);

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

            add_cfunc<fp_t>(s, "cfunc", {sum({x, y}), sum({x, expression{fp_t(.5)}}), sum({par[0], y})},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sum."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == approximately(ins[i] + ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(ins[i] + static_cast<fp_t>(.5), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(pars[i] + ins[i + batch_size], fp_t(100)));
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
    using fp_t = mppp::real;

    const auto prec = 237u;

    auto [x, y] = make_vars("x", "y");

    std::uniform_real_distribution<double> rdist(-10., 10.);

    auto gen = [&]() { return mppp::real(rdist(rng), prec); };

    std::vector<fp_t> outs, ins, pars;

    const auto batch_size = 1u;

    outs.resize(3u);
    ins.resize(2u);
    pars.resize(1u);

    for (auto high_accuracy : {false, true}) {
        for (auto compact_mode : {false, true}) {
            for (auto opt_level : {0u, 1u, 2u, 3u}) {
                std::generate(ins.begin(), ins.end(), gen);
                std::generate(outs.begin(), outs.end(), gen);
                std::generate(pars.begin(), pars.end(), gen);

                llvm_state s{kw::opt_level = opt_level};

                add_cfunc<fp_t>(s, "cfunc", {sum({x, y}), sum({x, expression{fp_t(.5)}}), sum({par[0], y})},
                                kw::prec = prec, kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

                if (opt_level == 0u && compact_mode) {
                    REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sum."));
                }

                s.compile();

                auto *cf_ptr = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(
                    s.jit_lookup("cfunc"));

                cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

                for (auto i = 0u; i < batch_size; ++i) {
                    REQUIRE(outs[i] == approximately(ins[i] + ins[i + batch_size], fp_t(100)));
                    REQUIRE(outs[i + batch_size] == approximately(ins[i] + static_cast<fp_t>(.5), fp_t(100)));
                    REQUIRE(outs[i + 2u * batch_size] == approximately(pars[i] + ins[i + batch_size], fp_t(100)));
                }
            }
        }
    }
}

#endif

TEST_CASE("sum split")
{
    auto sum_wrapper = [](const std::vector<expression> &args) { return expression{func{detail::sum_impl(args)}}; };

    auto [x, y] = make_vars("x", "y");

    auto s = sum_wrapper({x, x, x, x, y, y, y});

    auto ss1 = detail::sum_split(s, 2u);

    REQUIRE(ss1
            == sum_wrapper({sum_wrapper({sum_wrapper({x, x}), sum_wrapper({x, x})}),
                            expression(func(detail::sum_impl({sum_wrapper({y, y}), y})))}));

    ss1 = detail::sum_split(s, 3u);

    REQUIRE(ss1 == expression(func(detail::sum_impl({sum_wrapper({x, x, x}), sum_wrapper({x, y, y}), y}))));

    ss1 = detail::sum_split(s, 4u);

    REQUIRE(ss1 == sum_wrapper({sum_wrapper({x, x, x, x}), sum_wrapper({y, y, y})}));

    ss1 = detail::sum_split(s, 8u);

    REQUIRE(s == ss1);

    ss1 = detail::sum_split(s, 9u);

    REQUIRE(s == ss1);

    ss1 = detail::sum_split(s, 10u);

    REQUIRE(s == ss1);

    // Check with a non-sum expression.
    auto ns = x * y;
    REQUIRE(detail::sum_split(ns, 8u) == ns);
    REQUIRE(std::get<func>(detail::sum_split(ns, 8u).value()).get_ptr() == std::get<func>(ns.value()).get_ptr());

    // A cfunc test with nested sums.
    std::vector<expression> x_vars;
    for (auto i = 0; i < 10; ++i) {
        x_vars.emplace_back(fmt::format("x_{}", i));
    }

    s = sum_wrapper({x_vars[0], x_vars[1], x_vars[2], x_vars[3], x_vars[4], x_vars[5], x_vars[6], x_vars[7], x_vars[8],
                     x_vars[9], cos(sum_wrapper(x_vars))});

    llvm_state ls;
    const auto dc = add_cfunc<double>(ls, "cfunc", {s});

    for (const auto &ex : dc) {
        if (const auto *fptr = std::get_if<func>(&ex.value())) {
            if (const auto *sptr = fptr->extract<detail::sum_impl>()) {
                REQUIRE(sptr->args().size() <= 8u);
            }
        }
    }

    ls.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ls.jit_lookup("cfunc"));

    std::vector<double> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double output = 0;

    cf_ptr(&output, inputs.data(), nullptr, nullptr);

    REQUIRE(output
            == approximately((1. + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10)
                             + std::cos(1. + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10)));
}

TEST_CASE("sum simpls")
{
    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum({sum({x, y}), sum({z, t})}) == sum({x, y, z, t}));
    REQUIRE(sum({x, x}) == prod({2_dbl, x}));

    REQUIRE(sum({sum({x, y}), y, z, t, prod({2_dbl, x}), y, z, t})
            == sum({prod({2_dbl, t}), prod({3_dbl, x}), prod({3_dbl, y}), prod({2_dbl, z})}));
}

TEST_CASE("sum_to_sub")
{
    auto [x, y] = make_vars("x", "y");

    auto ret = detail::sum_to_sub({sum({x, prod({-1_dbl, y})})});
    REQUIRE(ret[0] == detail::sub(x, y));

    ret = detail::sum_to_sub({sum({-5_dbl, x, prod({-1_dbl, y})})});
    REQUIRE(ret[0] == detail::sub(x - 5_dbl, y));

    ret = detail::sum_to_sub({sum({5_dbl, x, cos(y)})});
    REQUIRE(ret[0] == sum({5_dbl, x, cos(y)}));

    ret = detail::sum_to_sub({sum({-5_dbl, prod({-2_dbl, x}), prod({-1_dbl, cos(y)})})});
    REQUIRE(ret[0] == detail::sub((-5. - (2. * x)), cos(y)));

    ret = detail::sum_to_sub({cos(sum({-5_dbl, prod({-2_dbl, x}), prod({-1_dbl, cos(y)})}))});
    REQUIRE(ret[0] == cos(detail::sub((-5. - (2. * x)), cos(y))));

    auto tmp = sum({x, prod({-1_dbl, cos(x)})});

    ret = detail::sum_to_sub({sum({prod({-1_dbl, y}), sin(tmp), cos(tmp)})});
    REQUIRE(ret[0] == detail::sub(sum({sin(detail::sub(x, cos(x))), cos(detail::sub(x, cos(x)))}), y));
}

TEST_CASE("normalise")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(normalise(x + y) == x + y);
    REQUIRE(normalise(subs(x + y, {{x, 2_dbl * y}})) == 3. * y);
}
