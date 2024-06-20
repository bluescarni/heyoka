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

#if defined(__GNUC__) || defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"

#endif

TEST_CASE("basic")
{
    auto x = make_vars("x");

    REQUIRE(expression{func{detail::logical_and_impl{}}} == expression{func{detail::logical_and_impl{{1_dbl}}}});
    REQUIRE(logical_and({}) == 1_dbl);
    REQUIRE(logical_and({x}) == x);

    REQUIRE(expression{func{detail::logical_or_impl{}}} == expression{func{detail::logical_or_impl{{1_dbl}}}});
    REQUIRE(logical_or({}) == 0_dbl);
    REQUIRE(logical_or({x}) == x);
}

TEST_CASE("stream")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        std::ostringstream oss;
        oss << logical_and({x, y});
        REQUIRE(oss.str() == "logical_and(x, y)");
    }

    {
        std::ostringstream oss;
        oss << logical_and({x, y + z});
        REQUIRE(oss.str() == "logical_and(x, (y + z))");
    }

    {
        std::ostringstream oss;
        oss << logical_or({x, y});
        REQUIRE(oss.str() == "logical_or(x, y)");
    }

    {
        std::ostringstream oss;
        oss << logical_or({x, y + z});
        REQUIRE(oss.str() == "logical_or(x, (y + z))");
    }
}

TEST_CASE("diff")
{
    auto [x, y] = make_vars("x", "y");

    REQUIRE(diff(logical_and({x, y}), "x") == 0_dbl);
    REQUIRE(diff(logical_and({x * y, y - x}), "x") == 0_dbl);

    REQUIRE(diff(logical_or({x, y}), "x") == 0_dbl);
    REQUIRE(diff(logical_or({x * y, y - x}), "x") == 0_dbl);
}

TEST_CASE("s11n logical_and")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = logical_and({x, y});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 1_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == logical_and({x, y}));
}

TEST_CASE("s11n logical_or")
{
    std::stringstream ss;

    auto [x, y] = make_vars("x", "y");

    auto ex = logical_or({x, y});

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 1_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == logical_or({x, y}));
}

TEST_CASE("cfunc logical_and")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_int_distribution<int> idist(-3, 3);

        auto gen = [&idist]() { return static_cast<fp_t>(idist(rng)); };

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

            add_cfunc<fp_t>(s, "cfunc", {logical_and({x, y}), logical_and({par[0], heyoka::time, y})}, {x, y},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.logical_and."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == (ins[i] && ins[i + batch_size]));
                REQUIRE(outs[i + batch_size] == (pars[i] && time[i] && ins[i + batch_size]));
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

    // A test specific for NaN handling.
    auto [x, y] = make_vars("x", "y");
    llvm_state s;
    std::vector<double> outs, ins{1., std::numeric_limits<double>::quiet_NaN()};
    outs.resize(1);

    add_cfunc<double>(s, "cfunc", {logical_and({x, y})}, {x, y});

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

    REQUIRE(outs[0] == 1.);
}

TEST_CASE("cfunc logical_or")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_int_distribution<int> idist(-3, 3);

        auto gen = [&idist]() { return static_cast<fp_t>(idist(rng)); };

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

            add_cfunc<fp_t>(s, "cfunc", {logical_or({x, y}), logical_or({par[0], heyoka::time, y})}, {x, y},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.logical_or."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            cf_ptr(outs.data(), ins.data(), pars.data(), time.data());

            for (auto i = 0u; i < batch_size; ++i) {
                REQUIRE(outs[i] == (ins[i] || ins[i + batch_size]));
                REQUIRE(outs[i + batch_size] == (pars[i] || time[i] || ins[i + batch_size]));
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

    // A test specific for NaN handling.
    auto [x, y] = make_vars("x", "y");
    llvm_state s;
    std::vector<double> outs, ins{0., std::numeric_limits<double>::quiet_NaN()};
    outs.resize(1);

    add_cfunc<double>(s, "cfunc", {logical_or({x, y})}, {x, y});

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

    REQUIRE(outs[0] == 1.);
}

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc logical_and mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {logical_and({x, y}), logical_and({par[0], heyoka::time, y})}, {x, y},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

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
            REQUIRE(outs[i] == (ins[i] && ins[i + batch_size]));
            REQUIRE(static_cast<bool>(outs[i]));
            REQUIRE(outs[i + batch_size] == (pars[i] && time[i] && ins[i + batch_size]));
            REQUIRE(!static_cast<bool>(outs[i + batch_size]));
        }
    }

    // A test specific for NaN handling.
    llvm_state s;
    std::vector<mppp::real> outs, ins{mppp::real{1., prec}, mppp::real{std::numeric_limits<double>::quiet_NaN(), prec}};
    outs.resize(1, mppp::real{0., prec});

    add_cfunc<mppp::real>(s, "cfunc", {logical_and({x, y})}, {x, y}, kw::prec = prec);

    s.compile();

    auto *cf_ptr = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
        s.jit_lookup("cfunc"));

    cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

    REQUIRE(outs[0] == 1);
}

TEST_CASE("cfunc logical_or mp")
{
    auto [x, y] = make_vars("x", "y");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {logical_or({x, y}), logical_or({par[0], heyoka::time, y})}, {x, y},
                                  kw::compact_mode = compact_mode, kw::prec = prec);

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
            REQUIRE(outs[i] == (ins[i] || ins[i + batch_size]));
            REQUIRE(static_cast<bool>(outs[i]));
            REQUIRE(outs[i + batch_size] == (pars[i] || time[i] || ins[i + batch_size]));
            REQUIRE(static_cast<bool>(outs[i + batch_size]));
        }
    }

    // A test specific for NaN handling.
    llvm_state s;
    std::vector<mppp::real> outs, ins{mppp::real{0., prec}, mppp::real{std::numeric_limits<double>::quiet_NaN(), prec}};
    outs.resize(1, mppp::real{0., prec});

    add_cfunc<mppp::real>(s, "cfunc", {logical_or({x, y})}, {x, y}, kw::prec = prec);

    s.compile();

    auto *cf_ptr = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
        s.jit_lookup("cfunc"));

    cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

    REQUIRE(outs[0] == 1);
}

#endif

#if defined(__GNUC__) || defined(__clang__)

#pragma GCC diagnostic pop

#endif
