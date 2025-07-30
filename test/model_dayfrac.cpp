// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
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
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/dayfrac.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// NOTE: declare the implementation function called from within LLVM, so that we can test it.
extern "C" void heyoka_tt_to_dayfrac(double *, std::uint32_t) noexcept;

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

TEST_CASE("impl")
{
    // J2000 test.
    double inout = 0.;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(0.49925712962962965, 100000.));

    // January 1st test.
    // Time('2007-01-01 00:00:00', format='iso', scale='utc').
    inout = 2556.5007544444443;
    heyoka_tt_to_dayfrac(&inout, 1);
    // NOTE: ideally out should be zero, but it could also be 365. This happens because rounding errors in the time
    // conversions may lead to interpreting the original date as the very end of 2006 (rather than the very beginning of
    // 2007).
    REQUIRE((inout == approximately(365.) || inout == approximately(0.)));

    // Contemporary test.
    // Time('2025-07-20 01:23:45.6789', format='iso', scale='utc').
    inout = 9331.558968320602;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(200.0581675798611));

    // Negative time test.
    // Time('1998-03-04 18:23:45.6789', format='iso', scale='utc').
    inout = -667.7327677905092;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(62.76650091319444, 10000.));

    // Leap year test.
    // Time('2008-12-31 22:23:45.6789', format='iso', scale='utc').
    inout = 3287.4339220243055;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(365.9331675798611));

    // Leap second test.
    // Time('1987-12-31 23:59:60', format='iso', scale='utc').
    inout = -4383.499361296296;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(365.));

    // Leap second + leap year test.
    // Time('2016-12-31 23:59:60', format='iso', scale='utc').
    inout = 6209.500789166666;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(366.));

    // Leap year with two leap seconds test.
    // Time('1972-12-31 23:59:60', format='iso', scale='utc').
    inout = -9861.499500185186;
    heyoka_tt_to_dayfrac(&inout, 1);
    REQUIRE(inout == approximately(366.00001157407405));
}

TEST_CASE("dayfrac basics")
{
    REQUIRE(model::dayfrac() == model::dayfrac(kw::time_expr = heyoka::time));
}

TEST_CASE("dayfrac s11n")
{
    {
        std::stringstream ss;

        auto x = make_vars("x");

        auto ex = model::dayfrac(kw::time_expr = x);

        {
            boost::archive::binary_oarchive oa(ss);

            oa << ex;
        }

        ex = 0_dbl;

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> ex;
        }

        REQUIRE(ex == model::dayfrac(kw::time_expr = x));
    }
}

TEST_CASE("dayfrac cfunc")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = make_vars("x");

        std::vector<fp_t> outs, ins;

        for (auto batch_size : {1u, 2u, 3u, 8u}) {
            if (batch_size != 1u && std::same_as<fp_t, long double> && skip_batch_ld) {
                continue;
            }

            outs.resize(batch_size);
            ins.resize(batch_size);

            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<fp_t>(s, "cfunc", {model::dayfrac(kw::time_expr = x)}, {x}, kw::batch_size = batch_size,
                            kw::compact_mode = compact_mode);

            if (opt_level == 0u && compact_mode) {
                REQUIRE(boost::contains(s.get_ir(), "heyoka.llvm_c_eval.dayfrac"));
            }

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(fp_t *, const fp_t *, const fp_t *, const fp_t *)>(s.jit_lookup("cfunc"));

            std::ranges::fill(ins, fp_t(9331.558968320602));
            cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

            for (auto i = 0u; i < batch_size; ++i) {
                using std::abs;
                REQUIRE(abs(outs[i] - static_cast<fp_t>(200.0581675798611))
                        < ((std::same_as<fp_t, float>) ? 1e-3 : 1e-11));
            }
        }
    };

    for (auto cm : {false, true}) {
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 0, cm); });
        tuple_for_each(fp_types, [&tester, cm](auto x) { tester(x, 3, cm); });
    }
}

#if defined(HEYOKA_HAVE_REAL)

// NOTE: the point of the multiprecision test is just to check we used the correct llvm primitives in the
// implementation.
TEST_CASE("dayfrac cfunc_mp")
{
    auto x = make_vars("x");

    const auto prec = 237u;

    for (auto compact_mode : {false, true}) {
        for (auto opt_level : {0u, 3u}) {
            llvm_state s{kw::opt_level = opt_level};

            add_cfunc<mppp::real>(s, "cfunc", {model::dayfrac(kw::time_expr = x)}, {x}, kw::compact_mode = compact_mode,
                                  kw::prec = prec);

            s.compile();

            auto *cf_ptr
                = reinterpret_cast<void (*)(mppp::real *, const mppp::real *, const mppp::real *, const mppp::real *)>(
                    s.jit_lookup("cfunc"));

            const std::vector ins{mppp::real{0, prec}};
            std::vector<mppp::real> outs{mppp::real{0, prec}};

            cf_ptr(outs.data(), ins.data(), nullptr, nullptr);

            REQUIRE(abs(outs[0] - 0.49925712962962965) < 1e-11);
        }
    }
}

#endif
