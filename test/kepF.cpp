// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <sstream>
#include <tuple>
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
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepF.hpp>
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
#if LLVM_VERSION_MAJOR >= 13 && LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

TEST_CASE("kepF def ctor")
{
    detail::kepF_impl k;

    REQUIRE(k.args().size() == 3u);
    REQUIRE(k.args()[0] == 0_dbl);
    REQUIRE(k.args()[1] == 0_dbl);
    REQUIRE(k.args()[2] == 0_dbl);
}

TEST_CASE("kepF diff")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        REQUIRE(diff(kepF(x, y, z), x)
                == -cos(kepF(x, y, z)) / (1_dbl - x * sin(kepF(x, y, z)) - y * cos(kepF(x, y, z))));
        REQUIRE(diff(kepF(x, y, z), y)
                == sin(kepF(x, y, z)) / (1_dbl - x * sin(kepF(x, y, z)) - y * cos(kepF(x, y, z))));
        REQUIRE(diff(kepF(x, y, z), z) == 1_dbl / (1_dbl - x * sin(kepF(x, y, z)) - y * cos(kepF(x, y, z))));
        auto F = kepF(x * x, x * y, x * z);
        REQUIRE(diff(F, x) == (y * sin(F) - 2_dbl * x * cos(F) + z) / (1_dbl - x * x * sin(F) - x * y * cos(F)));
        REQUIRE(diff(F, y) == (x * sin(F)) / (1_dbl - x * x * sin(F) - x * y * cos(F)));
    }

    {
        REQUIRE(diff(kepF(par[0], y, z), par[0])
                == -cos(kepF(par[0], y, z)) / (1_dbl - par[0] * sin(kepF(par[0], y, z)) - y * cos(kepF(par[0], y, z))));
        REQUIRE(diff(kepF(x, par[1], z), par[1])
                == sin(kepF(x, par[1], z)) / (1_dbl - x * sin(kepF(x, par[1], z)) - par[1] * cos(kepF(x, par[1], z))));
        REQUIRE(diff(kepF(x, y, par[2]), par[2])
                == 1_dbl / (1_dbl - x * sin(kepF(x, y, par[2])) - y * cos(kepF(x, y, par[2]))));
        auto F = kepF(par[0] * par[0], par[0] * par[1], par[0] * par[2]);
        REQUIRE(diff(F, par[0])
                == (par[1] * sin(F) - 2_dbl * par[0] * cos(F) + par[2])
                       / (1_dbl - par[0] * par[0] * sin(F) - par[0] * par[1] * cos(F)));
        REQUIRE(diff(F, par[1]) == (par[0] * sin(F)) / (1_dbl - par[0] * par[0] * sin(F) - par[0] * par[1] * cos(F)));
    }
}

#define HEYOKA_TEST_KEPF_OVERLOAD(type)                                                                                \
    {                                                                                                                  \
        auto k = kepF("x"_var, static_cast<type>(1.1), static_cast<type>(1.3));                                        \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepF(static_cast<type>(1.1), "y"_var, static_cast<type>(1.3));                                        \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepF(static_cast<type>(1.1), static_cast<type>(1.3), "z"_var);                                        \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepF("x"_var, "y"_var, static_cast<type>(1.3));                                                       \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepF("x"_var, static_cast<type>(1.3), "z"_var);                                                       \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepF(static_cast<type>(1.3), "y"_var, "z"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }

TEST_CASE("kepF overloads")
{
    HEYOKA_TEST_KEPF_OVERLOAD(double);
    HEYOKA_TEST_KEPF_OVERLOAD(long double);

#if defined(HEYOKA_HAVE_REAL128)

    HEYOKA_TEST_KEPF_OVERLOAD(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL128)

    HEYOKA_TEST_KEPF_OVERLOAD(mppp::real);

#endif
}

#undef HEYOKA_TEST_KEPF_OVERLOAD

TEST_CASE("kepF s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = kepF(x, y, z);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == kepF(x, y, z));
}

TEST_CASE("cfunc")
{
    using std::isnan;

    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto eps_close = [](const fp_t &a, const fp_t &b) {
            using std::abs;
            return abs(a - b) <= std::numeric_limits<fp_t>::epsilon() * 10000;
        };

        auto [h, k, lam] = make_vars("h", "k", "lam");

        std::uniform_real_distribution<double> lam_dist(-1e5, 1e5), h_dist(std::nextafter(-1., 0.), 1.);

        auto generate_hk = [&h_dist]() {
            // Generate h.
            auto h_val = h_dist(rng);

            // Generate a k such that h**2+k**2<1.
            const auto max_abs_k = std::sqrt(1. - h_val * h_val);
            std::uniform_real_distribution<double> k_dist(std::nextafter(-max_abs_k, 0.), max_abs_k);
            auto k_val = static_cast<fp_t>(k_dist(rng));

            return std::make_pair(static_cast<fp_t>(h_val), std::move(k_val));
        };

        std::vector<fp_t> outs, ins, pars;

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            if (batch_size != 1u && std::is_same_v<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size * 3u);
            ins.resize(batch_size * 3u);
            pars.resize(batch_size * 2u);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {kepF(h, k, lam), kepF(par[0], par[1], lam), kepF(.5_dbl, .3_dbl, lam)},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepF."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            for (auto niter = 0; niter < 100; ++niter) {
                for (auto i = 0u; i < batch_size; ++i) {
                    // Generate the hs and ks.
                    auto [hval, kval] = generate_hk();
                    // Generate the lam.
                    auto lamval = lam_dist(rng);

                    ins[i] = hval;
                    ins[i + batch_size] = kval;
                    ins[i + 2u * batch_size] = lamval;

                    // Generate another pair of hs and ks for the pars.
                    std::tie(hval, kval) = generate_hk();
                    pars[i] = hval;
                    pars[i + batch_size] = kval;
                }

                cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

                for (auto i = 0u; i < batch_size; ++i) {
                    using std::cos;
                    using std::sin;

                    // First output.
                    REQUIRE(!isnan(outs[i]));
                    auto Fval = outs[i];
                    auto hval = ins[i];
                    auto kval = ins[i + batch_size];
                    auto lamval = ins[i + 2u * batch_size];
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * cos(Fval) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * cos(Fval) - kval * sin(Fval))));

                    // Second output.
                    REQUIRE(!isnan(outs[i + batch_size]));
                    Fval = outs[i + batch_size];
                    hval = pars[i];
                    kval = pars[i + batch_size];
                    lamval = ins[i + 2u * batch_size];
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * cos(Fval) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * cos(Fval) - kval * sin(Fval))));

                    // Third output.
                    REQUIRE(!isnan(outs[i + batch_size * 2u]));
                    Fval = outs[i + batch_size * 2u];
                    hval = .5;
                    kval = .3;
                    lamval = ins[i + 2u * batch_size];
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * cos(Fval) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * cos(Fval) - kval * sin(Fval))));
                }
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

    // Check nan/invalid values handling.
    auto [h, k, lam] = make_vars("h", "k", "lam");

    llvm_state s;

    add_cfunc<double>(s, "cfunc", {kepF(h, k, lam)});

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double out = 0;
    double ins[3] = {.1, .2, std::numeric_limits<double>::quiet_NaN()};
    cf_ptr(&out, ins, nullptr, nullptr);

    REQUIRE(isnan(out));

    ins[0] = std::numeric_limits<double>::quiet_NaN();
    ins[1] = .2;
    ins[2] = 1.;

    cf_ptr(&out, ins, nullptr, nullptr);

    REQUIRE(isnan(out));

    ins[0] = .2;
    ins[1] = std::numeric_limits<double>::quiet_NaN();
    ins[2] = 1.;

    cf_ptr(&out, ins, nullptr, nullptr);

    REQUIRE(isnan(out));

    ins[0] = .2;
    ins[1] = 1.;
    ins[2] = 1.;

    cf_ptr(&out, ins, nullptr, nullptr);

    REQUIRE(isnan(out));
}

// A numerically-difficult case in which the shrinking bounding range tolerance
// check is necessary to prevent the NR from bouncing around the root endlessly.
TEST_CASE("cfunc bound")
{
    using std::cos;
    using std::isnan;
    using std::sin;

    auto [h, k, lam] = make_vars("h", "k", "lam");

    llvm_state s;

    add_cfunc<double>(s, "cfunc", {kepF(h, k, lam)});

    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("cfunc"));

    double Fval = 0;
    const auto hval = 0.9796044983076618306583327466796618;
    const auto kval = 0.1800214955091904156514459600657574;
    const auto lamval = 93548.66355109098367393016815185547;
    double ins[3] = {hval, kval, lamval};

    cf_ptr(&Fval, ins, nullptr, nullptr);

    auto eps_close = [](double a, double b) {
        using std::abs;
        return abs(a - b) <= std::numeric_limits<double>::epsilon() * 100;
    };

    REQUIRE(!isnan(Fval));
    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * cos(Fval) - kval * sin(Fval))));
}
