// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
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
#include <heyoka/math/kepDE.hpp>
#include <heyoka/math/sin.hpp>
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

TEST_CASE("kepDE def ctor")
{
    detail::kepDE_impl k;

    REQUIRE(k.args().size() == 3u);
    REQUIRE(k.args()[0] == 0_dbl);
    REQUIRE(k.args()[1] == 0_dbl);
    REQUIRE(k.args()[2] == 0_dbl);
}

TEST_CASE("kepDE diff")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    {
        REQUIRE(diff(kepDE(x, y, z), x)
                == (cos(kepDE(x, y, z)) - 1_dbl) / (1_dbl + x * sin(kepDE(x, y, z)) - y * cos(kepDE(x, y, z))));
        REQUIRE(diff(kepDE(x, y, z), y)
                == sin(kepDE(x, y, z)) / (1_dbl + x * sin(kepDE(x, y, z)) - y * cos(kepDE(x, y, z))));
        REQUIRE(diff(kepDE(x, y, z), z) == 1_dbl / (1_dbl + x * sin(kepDE(x, y, z)) - y * cos(kepDE(x, y, z))));
        auto DE = kepDE(x * x, x * y, x * z);
        REQUIRE(diff(DE, x)
                == (2_dbl * x * (cos(DE) - 1_dbl) + y * sin(DE) + z) / (1_dbl + x * x * sin(DE) - x * y * cos(DE)));
        REQUIRE(diff(DE, y) == (x * sin(DE)) / (1_dbl + x * x * sin(DE) - x * y * cos(DE)));
    }

    {
        REQUIRE(diff(kepDE(par[0], y, z), par[0])
                == (cos(kepDE(par[0], y, z)) - 1_dbl)
                       / (1_dbl + par[0] * sin(kepDE(par[0], y, z)) - y * cos(kepDE(par[0], y, z))));
        REQUIRE(diff(kepDE(x, par[1], z), par[1])
                == sin(kepDE(x, par[1], z))
                       / (1_dbl + x * sin(kepDE(x, par[1], z)) - par[1] * cos(kepDE(x, par[1], z))));
        REQUIRE(diff(kepDE(x, y, par[2]), par[2])
                == 1_dbl / (1_dbl + x * sin(kepDE(x, y, par[2])) - y * cos(kepDE(x, y, par[2]))));
        auto DE = kepDE(par[0] * par[0], par[0] * par[1], par[0] * par[2]);
        REQUIRE(diff(DE, par[0])
                == (2_dbl * par[0] * (cos(DE) - 1_dbl) + par[1] * sin(DE) + par[2])
                       / (1_dbl + par[0] * par[0] * sin(DE) - par[0] * par[1] * cos(DE)));
        REQUIRE(diff(DE, par[1])
                == (par[0] * sin(DE)) / (1_dbl + par[0] * par[0] * sin(DE) - par[0] * par[1] * cos(DE)));
    }

    // Use numerical differencing as a further check.
    {
        llvm_state s;

        auto der = diff(kepDE(x, y, z), x);
        add_cfunc<double>(s, "der", {der});
        add_cfunc<double>(s, "f", {kepDE(x, y, z)});
        s.compile();

        auto *der_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("der"));
        auto *f_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("f"));

        double out_der{};
        const std::array in_der = {.1, .1, 1.1};
        der_ptr(&out_der, in_der.data(), nullptr, nullptr);

        double out_f[2] = {};
        const auto in_f1 = in_der;
        auto in_f2 = in_der;
        in_f2[0] += 1e-8;
        f_ptr(out_f, in_f1.data(), nullptr, nullptr);
        f_ptr(out_f + 1, in_f2.data(), nullptr, nullptr);

        REQUIRE(out_der == approximately((out_f[1] - out_f[0]) / 1e-8, 1e9));
    }

    {
        llvm_state s;

        auto der = diff(kepDE(x, y, z), y);
        add_cfunc<double>(s, "der", {der});
        add_cfunc<double>(s, "f", {kepDE(x, y, z)});
        s.compile();

        auto *der_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("der"));
        auto *f_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("f"));

        double out_der{};
        const std::array in_der = {.1, .1, 1.1};
        der_ptr(&out_der, in_der.data(), nullptr, nullptr);

        double out_f[2] = {};
        const auto in_f1 = in_der;
        auto in_f2 = in_der;
        in_f2[1] += 1e-8;
        f_ptr(out_f, in_f1.data(), nullptr, nullptr);
        f_ptr(out_f + 1, in_f2.data(), nullptr, nullptr);

        REQUIRE(out_der == approximately((out_f[1] - out_f[0]) / 1e-8, 1e9));
    }

    {
        llvm_state s;

        auto der = diff(kepDE(x, y, z), z);
        add_cfunc<double>(s, "der", {der});
        add_cfunc<double>(s, "f", {kepDE(x, y, z)});
        s.compile();

        auto *der_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("der"));
        auto *f_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("f"));

        double out_der{};
        const std::array in_der = {.1, .1, 1.1};
        der_ptr(&out_der, in_der.data(), nullptr, nullptr);

        double out_f[2] = {};
        const auto in_f1 = in_der;
        auto in_f2 = in_der;
        in_f2[2] += 1e-8;
        f_ptr(out_f, in_f1.data(), nullptr, nullptr);
        f_ptr(out_f + 1, in_f2.data(), nullptr, nullptr);

        REQUIRE(out_der == approximately((out_f[1] - out_f[0]) / 1e-8, 1e9));
    }
}

#define HEYOKA_TEST_KEPDE_OVERLOAD(type)                                                                               \
    {                                                                                                                  \
        auto k = kepDE("x"_var, static_cast<type>(1.1), static_cast<type>(1.3));                                       \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepDE(static_cast<type>(1.1), "y"_var, static_cast<type>(1.3));                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepDE(static_cast<type>(1.1), static_cast<type>(1.3), "z"_var);                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.1)});      \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepDE("x"_var, "y"_var, static_cast<type>(1.3));                                                      \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[2].value()) == number{static_cast<type>(1.3)});      \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepDE("x"_var, static_cast<type>(1.3), "z"_var);                                                      \
        REQUIRE(std::get<func>(k.value()).args()[0] == "x"_var);                                                       \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[1].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }                                                                                                                  \
    {                                                                                                                  \
        auto k = kepDE(static_cast<type>(1.3), "y"_var, "z"_var);                                                      \
        REQUIRE(std::get<number>(std::get<func>(k.value()).args()[0].value()) == number{static_cast<type>(1.3)});      \
        REQUIRE(std::get<func>(k.value()).args()[1] == "y"_var);                                                       \
        REQUIRE(std::get<func>(k.value()).args()[2] == "z"_var);                                                       \
    }

TEST_CASE("kepDE overloads")
{
    HEYOKA_TEST_KEPDE_OVERLOAD(float);
    HEYOKA_TEST_KEPDE_OVERLOAD(double);
    HEYOKA_TEST_KEPDE_OVERLOAD(long double);

#if defined(HEYOKA_HAVE_REAL128)

    HEYOKA_TEST_KEPDE_OVERLOAD(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

    HEYOKA_TEST_KEPDE_OVERLOAD(mppp::real);

#endif
}

#undef HEYOKA_TEST_KEPDE_OVERLOAD

TEST_CASE("kepDE s11n")
{
    std::stringstream ss;

    auto [x, y, z] = make_vars("x", "y", "z");

    auto ex = kepDE(x, y, z);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << ex;
    }

    ex = 0_dbl;

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> ex;
    }

    REQUIRE(ex == kepDE(x, y, z));
}

TEST_CASE("cfunc")
{
    using std::isnan;
    using std::nextafter;
    using std::sqrt;

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
            auto h_val = static_cast<fp_t>(h_dist(rng));

            // Generate a k such that h**2+k**2<1.
            const auto max_abs_k = sqrt(fp_t(1) - h_val * h_val);
            std::uniform_real_distribution<double> k_dist(static_cast<double>(nextafter(-max_abs_k, fp_t(0))),
                                                          static_cast<double>(max_abs_k));
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

            add_cfunc<fp_t>(s, "cfunc", {kepDE(h, k, lam), kepDE(par[0], par[1], lam), kepDE(.5_dbl, .3_dbl, lam)},
                            kw::batch_size = batch_size, kw::high_accuracy = high_accuracy,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepDE."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            for (auto niter = 0; niter < 100; ++niter) {
                for (auto i = 0u; i < batch_size; ++i) {
                    // Generate the hs and ks.
                    auto [hval, kval] = generate_hk();
                    // Generate the lam.
                    auto lamval = static_cast<fp_t>(lam_dist(rng));

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
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));

                    // Second output.
                    REQUIRE(!isnan(outs[i + batch_size]));
                    Fval = outs[i + batch_size];
                    hval = pars[i];
                    kval = pars[i + batch_size];
                    lamval = ins[i + 2u * batch_size];
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));

                    // Third output.
                    REQUIRE(!isnan(outs[i + batch_size * 2u]));
                    Fval = outs[i + batch_size * 2u];
                    hval = fp_t(.5);
                    kval = fp_t(.3);
                    lamval = ins[i + 2u * batch_size];
                    REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));
                    REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (fp_t(1) - cos(Fval)) - kval * sin(Fval))));
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

    add_cfunc<double>(s, "cfunc", {kepDE(h, k, lam)});

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

#if defined(HEYOKA_HAVE_REAL)

TEST_CASE("cfunc mp")
{
    using fp_t = mppp::real;

    const auto prec = 237;

    auto eps_close = [&](const fp_t &a, const fp_t &b) {
        using std::abs;
        return abs(a - b) <= mppp::real{1ul, -(prec - 1), prec} * 10000;
    };

    auto [h, k, lam] = make_vars("h", "k", "lam");

    std::uniform_real_distribution<double> lam_dist(-1e5, 1e5), h_dist(std::nextafter(-1., 0.), 1.);

    auto generate_hk = [&]() {
        // Generate h.
        auto h_val = h_dist(rng);

        // Generate a k such that h**2+k**2<1.
        const auto max_abs_k = std::sqrt(1. - h_val * h_val);
        std::uniform_real_distribution<double> k_dist(std::nextafter(-max_abs_k, 0.), max_abs_k);
        auto k_val = mppp::real(k_dist(rng), prec);

        return std::make_pair(mppp::real(h_val, prec), std::move(k_val));
    };

    std::vector<fp_t> outs, ins, pars;

    outs.resize(3u, mppp::real{0, prec});
    ins.resize(3u);
    pars.resize(2u);

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {kepDE(h, k, lam), kepDE(par[0], par[1], lam), kepDE(.5_dbl, .3_dbl, lam)},
                            kw::compact_mode = compact_mode, kw::prec = prec);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.kepDE."));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            // Generate the hs and ks.
            auto [hval, kval] = generate_hk();
            // Generate the lam.
            auto lamval = mppp::real(lam_dist(rng), prec);

            ins[0] = hval;
            ins[1] = kval;
            ins[2] = lamval;

            // Generate another pair of hs and ks for the pars.
            std::tie(hval, kval) = generate_hk();
            pars[0] = hval;
            pars[1] = kval;

            cf_ptr(outs.data(), ins.data(), pars.data(), nullptr);

            using std::cos;
            using std::sin;

            // First output.
            REQUIRE(!isnan(outs[0]));
            auto Fval = outs[0];
            hval = ins[0];
            kval = ins[1];
            lamval = ins[2];
            REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));
            REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));

            // Second output.
            REQUIRE(!isnan(outs[1]));
            Fval = outs[1];
            hval = pars[0];
            kval = pars[1];
            lamval = ins[2];
            REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));
            REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));

            // Third output.
            REQUIRE(!isnan(outs[2]));
            Fval = outs[2];
            hval = mppp::real(.5, prec);
            kval = mppp::real(.3, prec);
            lamval = ins[2];
            REQUIRE(eps_close(cos(lamval), cos(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));
            REQUIRE(eps_close(sin(lamval), sin(Fval + hval * (1. - cos(Fval)) - kval * sin(Fval))));
        }
    }
}

#endif
