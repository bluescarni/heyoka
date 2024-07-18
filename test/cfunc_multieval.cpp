// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <heyoka/config.hpp>

#include <random>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

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

std::mt19937 rng;

// NOTE: ICE on MSVC.
#if !defined(_MSC_VER) || defined(__clang__)

// Single-thread test.
TEST_CASE("multieval st")
{
    using Catch::Matchers::Message;

    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        // Setup the buffers.
        std::vector<fp_t> obuf, ibuf, pbuf, tbuf;

        using out_2d = typename cfunc<fp_t>::out_2d;
        using in_2d = typename cfunc<fp_t>::in_2d;
        using in_1d = typename cfunc<fp_t>::in_1d;

        // Def cted failure.
        auto cf0 = cfunc<fp_t>{};

        CHECK_THROWS_AS(cf0(out_2d{obuf.data(), 0, 0}, in_2d{ibuf.data(), 0, 0}), std::invalid_argument);

        cf0 = cfunc<fp_t>{{x + y, x - y},
                          {x, y},
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        // Error checking.
        REQUIRE_THROWS_MATCHES(cf0(out_2d{obuf.data(), 0, 0}, in_2d{ibuf.data(), 0, 0}), std::invalid_argument,
                               Message("Invalid outputs array passed to a cfunc: the number of function "
                                       "outputs is 2, but the number of rows in the outputs array is 0"));

        // Check with zero nevals.
        REQUIRE_NOTHROW(cf0(out_2d{obuf.data(), 2, 0}, in_2d{ibuf.data(), 2, 0}));

        // Check no error on zero nevals with null outputs span.
        REQUIRE_NOTHROW(cf0(out_2d{nullptr, 2, 0}, in_2d{ibuf.data(), 2, 0}));

        obuf.resize(20u);

        REQUIRE_THROWS_MATCHES(cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 0, 0}), std::invalid_argument,
                               Message("Invalid inputs array passed to a cfunc: the number of function "
                                       "inputs is 2, but the number of rows in the inputs array is 0"));

        ibuf.resize(20u);

        REQUIRE_THROWS_MATCHES(cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 1, 10}), std::invalid_argument,
                               Message("Invalid inputs array passed to a cfunc: the number of function "
                                       "inputs is 2, but the number of rows in the inputs array is 1"));
        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 5}), std::invalid_argument,
            Message("Invalid inputs array passed to a cfunc: the expected number of columns deduced from the "
                    "outputs array is 10, but the number of columns in the inputs array is 5"));

        cf0 = cfunc<fp_t>{{x + y + par[0], x - y + heyoka::time},
                          {x, y},
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}), std::invalid_argument,
            Message("An array of parameter values must be passed in order to evaluate a function with parameters"));

        pbuf.resize(10u);

        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 0, 0}),
            std::invalid_argument,
            Message("The array of parameter values provided for the evaluation "
                    "of a compiled function has 0 row(s), "
                    "but the number of parameters in the function is 1"));

        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 0, 3}),
            std::invalid_argument,
            Message("The array of parameter values provided for the evaluation "
                    "of a compiled function has 0 row(s), "
                    "but the number of parameters in the function is 1"));

        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 5}),
            std::invalid_argument,
            Message("The array of parameter values provided for the evaluation "
                    "of a compiled function has 5 column(s), "
                    "but the expected number of columns deduced from the "
                    "outputs array is 10"));

        REQUIRE_THROWS_MATCHES(
            cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10}),
            std::invalid_argument,
            Message("An array of time values must be provided in order to evaluate a time-dependent function"));

        tbuf.resize(10u);

        REQUIRE_THROWS_MATCHES(cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10},
                                   kw::pars = in_2d{pbuf.data(), 1, 10}, kw::time = in_1d{tbuf.data(), 5}),
                               std::invalid_argument,
                               Message("The array of time values provided for the evaluation "
                                       "of a compiled function has a size of 5, "
                                       "but the expected size deduced from the "
                                       "outputs array is 10"));

        // Functional testing.
        cf0 = cfunc<fp_t>{{x + y, x - y},
                          {x, y},
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        auto ospan = out_2d{obuf.data(), 2, 10};
        auto ispan = in_2d{ibuf.data(), 2, 10};
        auto pspan = in_2d{pbuf.data(), 1, 10};
        auto tspan = in_1d{tbuf.data(), 10};

        auto gen = [rdist = std::uniform_real_distribution<double>{10., 20.}]() mutable {
            return static_cast<fp_t>(rdist(rng));
        };

        std::ranges::generate(ibuf, gen);

        cf0(ospan, ispan);

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == ispan(0, j) + ispan(1, j));
            REQUIRE(ospan(1, j) == ispan(0, j) - ispan(1, j));
        }

        // Try also with empty par span.
        std::ranges::fill(obuf, fp_t(0));
        cf0(ospan, ispan, kw::pars = in_2d{nullptr, 0, 10});

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == ispan(0, j) + ispan(1, j));
            REQUIRE(ospan(1, j) == ispan(0, j) - ispan(1, j));
        }

        // Function with no inputs and no pars, but provided with empty outputs/pars.
        cf0 = cfunc<fp_t>{{3_dbl, 4_dbl},
                          {},
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        cf0(ospan, in_2d{nullptr, 0, 10}, kw::pars = in_2d{nullptr, 0, 10});

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == 3);
            REQUIRE(ospan(1, j) == 4);
        }

        // Example with pars and times.

        std::ranges::generate(pbuf, gen);
        std::ranges::generate(tbuf, gen);

        cf0 = cfunc<fp_t>{{x + y + par[0], x - y + heyoka::time},
                          {x, y},
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        cf0(ospan, ispan, kw::pars = pspan, kw::time = tspan);

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == approximately(ispan(0, j) + ispan(1, j) + pspan(0, j)));
            REQUIRE(ospan(1, j) == approximately(ispan(0, j) - ispan(1, j) + tspan[j]));
        }

        // Try all null spans, zero nevals.
        REQUIRE_NOTHROW(cf0(out_2d{nullptr, 2, 0}, in_2d{nullptr, 2, 0}, kw::pars = in_2d{nullptr, 1, 0},
                            kw::time = in_1d{nullptr, 0}));

        // Try function with no inputs.
        cf0 = cfunc<fp_t>({par[0], 2_dbl - heyoka::time}, {}, kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode);

        cf0(ospan, in_2d{nullptr, 0, 10}, kw::pars = pspan, kw::time = tspan);

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == pspan(0, j));
            REQUIRE(ospan(1, j) == 2 - tspan(j));
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

// Multi-thread tests.
TEST_CASE("multieval mt double")
{
    auto [x, y] = make_vars("x", "y");

    std::uniform_real_distribution<double> rdist{10, 20};

    const auto nevals = 1'000'003ull;

    // Setup the buffers.
    std::vector<double> obuf(2u * nevals), ibuf(2u * nevals), pbuf(nevals), tbuf(nevals);

    auto gen = [&rdist]() { return rdist(rng); };

    std::ranges::generate(obuf, gen);
    std::ranges::generate(ibuf, gen);
    std::ranges::generate(pbuf, gen);
    std::ranges::generate(tbuf, gen);

    auto ospan = cfunc<double>::out_2d{obuf.data(), 2, nevals};
    auto ispan = cfunc<double>::in_2d{ibuf.data(), 2, nevals};
    auto pspan = cfunc<double>::in_2d{pbuf.data(), 1, nevals};
    auto tspan = cfunc<double>::in_1d{tbuf.data(), nevals};

    for (auto cm : {false, true}) {
        for (auto batch_size : {0, 2, 4}) {
            auto cf0 = cfunc<double>{
                {x + y + par[0], x - y + heyoka::time}, {x, y}, kw::compact_mode = cm, kw::batch_size = batch_size};

            cf0(ospan, ispan, kw::pars = pspan, kw::time = tspan);

            for (std::size_t j = 0; j < ospan.extent(1); ++j) {
                REQUIRE(ospan(0, j) == approximately(ispan(0, j) + ispan(1, j) + pspan(0, j)));
                REQUIRE(ospan(1, j) == approximately(ispan(0, j) - ispan(1, j) + tspan[j]));
            }
        }
    }
}

#if defined(HEYOKA_HAVE_REAL128)

TEST_CASE("multieval mt real128")
{
    auto [x, y] = make_vars("x", "y");

    std::uniform_real_distribution<double> rdist{10, 20};

    const auto nevals = 10003ull;

    // Setup the buffers.
    std::vector<mppp::real128> obuf(2u * nevals), ibuf(2u * nevals), pbuf(nevals), tbuf(nevals);

    auto gen = [&rdist]() { return static_cast<mppp::real128>(rdist(rng)); };

    std::ranges::generate(obuf, gen);
    std::ranges::generate(ibuf, gen);
    std::ranges::generate(pbuf, gen);
    std::ranges::generate(tbuf, gen);

    auto ospan = cfunc<mppp::real128>::out_2d{obuf.data(), 2, nevals};
    auto ispan = cfunc<mppp::real128>::in_2d{ibuf.data(), 2, nevals};
    auto pspan = cfunc<mppp::real128>::in_2d{pbuf.data(), 1, nevals};
    auto tspan = cfunc<mppp::real128>::in_1d{tbuf.data(), nevals};

    for (auto cm : {false, true}) {
        auto cf0 = cfunc<mppp::real128>{{x + y + par[0], x - y + heyoka::time}, {x, y}, kw::compact_mode = cm};

        cf0(ospan, ispan, kw::pars = pspan, kw::time = tspan);

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == approximately(ispan(0, j) + ispan(1, j) + pspan(0, j)));
            REQUIRE(ospan(1, j) == approximately(ispan(0, j) - ispan(1, j) + tspan[j]));
        }
    }
}

#endif

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("multieval st mp")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    // Setup the buffers.
    std::vector<mppp::real> obuf, ibuf, pbuf, tbuf;

    using out_2d = cfunc<mppp::real>::out_2d;
    using in_2d = cfunc<mppp::real>::in_2d;
    using in_1d = cfunc<mppp::real>::in_1d;

    const auto prec = 31;

    auto cf0 = cfunc<mppp::real>{{x + y + par[0], x - y + heyoka::time}, {x, y}, kw::prec = prec};

    obuf.resize(20u, mppp::real{0, 30});
    ibuf.resize(20u, mppp::real{0, 29});
    pbuf.resize(10u, mppp::real{0, 28});
    tbuf.resize(10u, mppp::real{0, 27});

    REQUIRE_THROWS_MATCHES(
        cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10},
            kw::time = in_1d{tbuf.data(), 10}),
        std::invalid_argument,
        Message("An mppp::real with an invalid precision of 30 was detected in the arguments to the evaluation "
                "of a compiled function - the expected precision value is 31"));

    std::ranges::fill(obuf, mppp::real{0, prec});

    REQUIRE_THROWS_MATCHES(
        cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10},
            kw::time = in_1d{tbuf.data(), 10}),
        std::invalid_argument,
        Message("An mppp::real with an invalid precision of 29 was detected in the arguments to the evaluation "
                "of a compiled function - the expected precision value is 31"));

    std::ranges::fill(ibuf, mppp::real{2, prec});

    REQUIRE_THROWS_MATCHES(
        cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10},
            kw::time = in_1d{tbuf.data(), 10}),
        std::invalid_argument,
        Message("An mppp::real with an invalid precision of 28 was detected in the arguments to the evaluation "
                "of a compiled function - the expected precision value is 31"));

    std::ranges::fill(pbuf, mppp::real{3, prec});

    REQUIRE_THROWS_MATCHES(
        cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10},
            kw::time = in_1d{tbuf.data(), 10}),
        std::invalid_argument,
        Message("An mppp::real with an invalid precision of 27 was detected in the arguments to the evaluation "
                "of a compiled function - the expected precision value is 31"));

    std::ranges::fill(tbuf, mppp::real{4, prec});

    cf0(out_2d{obuf.data(), 2, 10}, in_2d{ibuf.data(), 2, 10}, kw::pars = in_2d{pbuf.data(), 1, 10},
        kw::time = in_1d{tbuf.data(), 10});

    auto ospan = out_2d{obuf.data(), 2, 10};
    auto ispan = in_2d{ibuf.data(), 2, 10};
    auto pspan = in_2d{pbuf.data(), 1, 10};
    auto tspan = in_1d{tbuf.data(), 10};

    for (std::size_t j = 0; j < ospan.extent(1); ++j) {
        REQUIRE(ospan(0, j) == approximately(ispan(0, j) + ispan(1, j) + pspan(0, j)));
        REQUIRE(ospan(1, j) == approximately(ispan(0, j) - ispan(1, j) + tspan[j]));
    }

    // Try a case of a function without time which is provided with a time array
    // with invalid precision for evaluation.
    cf0 = cfunc<mppp::real>{{x + y, x - y}, {x, y}, kw::prec = prec};

    std::ranges::fill(tbuf, mppp::real(0, 16));

    REQUIRE_THROWS_MATCHES(
        cf0(ospan, ispan, kw::time = tspan), std::invalid_argument,
        Message("An mppp::real with an invalid precision of 16 was detected in the arguments to the evaluation "
                "of a compiled function - the expected precision value is 31"));
}

TEST_CASE("multieval mt real")
{
    auto [x, y] = make_vars("x", "y");

    std::uniform_real_distribution<double> rdist{10, 20};

    const auto nevals = 10003ull;

    // Setup the buffers.
    std::vector<mppp::real> obuf(2u * nevals), ibuf(2u * nevals), pbuf(nevals), tbuf(nevals);

    const auto prec = 128;

    auto gen = [&]() { return mppp::real(rdist(rng), prec); };

    std::ranges::generate(obuf, gen);
    std::ranges::generate(ibuf, gen);
    std::ranges::generate(pbuf, gen);
    std::ranges::generate(tbuf, gen);

    auto ospan = cfunc<mppp::real>::out_2d{obuf.data(), 2, nevals};
    auto ispan = cfunc<mppp::real>::in_2d{ibuf.data(), 2, nevals};
    auto pspan = cfunc<mppp::real>::in_2d{pbuf.data(), 1, nevals};
    auto tspan = cfunc<mppp::real>::in_1d{tbuf.data(), nevals};

    for (auto cm : {false, true}) {
        auto cf0
            = cfunc<mppp::real>{{x + y + par[0], x - y + heyoka::time}, {x, y}, kw::compact_mode = cm, kw::prec = prec};

        cf0(ospan, ispan, kw::pars = pspan, kw::time = tspan);

        for (std::size_t j = 0; j < ospan.extent(1); ++j) {
            REQUIRE(ospan(0, j) == approximately(ispan(0, j) + ispan(1, j) + pspan(0, j)));
            REQUIRE(ospan(1, j) == approximately(ispan(0, j) - ispan(1, j) + tspan[j]));
        }
    }
}

#endif
