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
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/sum_sq.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;

// Wrapper to ease the transition of old test code
// after the removal of sum_sq() from the public API.
auto sum_sq(const std::vector<expression> &args)
{
    std::vector<expression> new_args;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        new_args.push_back(arg * arg);
    }

    return sum(new_args);
}

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
        detail::sum_sq_impl ss;

        REQUIRE(ss.args().empty());
        REQUIRE(ss.get_name() == "sum_sq");
    }

    {
        detail::sum_sq_impl ss({x, y, z});

        REQUIRE(ss.args() == std::vector{x, y, z});
    }

    {
        detail::sum_sq_impl ss({par[0], x, y, 2_dbl, z});

        REQUIRE(ss.args() == std::vector{par[0], x, y, 2_dbl, z});
    }

    {
        auto ss = sum_sq({0_dbl, 0_dbl, x, 0_dbl});

        REQUIRE(ss == x * x);
    }
}

TEST_CASE("stream test")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss;
        ss.to_stream(oss);

        REQUIRE(oss.str() == "()");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "x**2");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2)");
    }

    {
        std::ostringstream oss;

        detail::sum_sq_impl ss({x, y, z});
        ss.to_stream(oss);

        REQUIRE(oss.str() == "(x**2 + y**2 + z**2)");
    }
}

TEST_CASE("sum_sq function")
{
    using Catch::Matchers::Message;

    auto [x, y, z, t] = make_vars("x", "y", "z", "t");

    REQUIRE(sum_sq({}) == 0_dbl);
    REQUIRE(sum_sq({x}) == x * x);
}

TEST_CASE("sum_sq s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = sum_sq({x, y, z});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == sum_sq({x, y, z}));
}

// TODO restore eventually.
#if 0

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

            add_cfunc<fp_t>(s, "cfunc", {sum_sq({x, y}), sum_sq({x, expression{fp_t(.5)}}), sum_sq({par[0], y})},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.sum_sq."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i]
                        == approximately(ins[i] * ins[i] + ins[i + batch_size] * ins[i + batch_size], fp_t(100)));
                REQUIRE(outs[i + batch_size]
                        == approximately(ins[i] * ins[i] + static_cast<fp_t>(.5) * 0.5, fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size]
                        == approximately(pars[i] * pars[i] + ins[i + batch_size] * ins[i + batch_size], fp_t(100)));
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

#endif

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc_mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {sum_sq({x, y}), sum_sq({x, expression{.5}}), sum_sq({par[0], y})},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{".7", prec}, mppp::real{".1", prec}};
            const std::vector pars{mppp::real{"-.1", prec}};
            std::vector<mppp::real> outs(3u, mppp::real{0, prec});

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            auto i = 0u;
            REQUIRE(outs[i] == ins[i] * ins[i] + ins[i + 1u] * ins[i + 1u]);
            REQUIRE(outs[i + 1u] == ins[i] * ins[i] + .5 * .5);
            REQUIRE(outs[i + 2u * 1u] == pars[i] * pars[i] + ins[i + 1u] * ins[i + 1u]);
        }
    }
}

#endif

TEST_CASE("commutativity")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(std::get<func>(sum_sq({x, y}).value()).is_commutative());
}
