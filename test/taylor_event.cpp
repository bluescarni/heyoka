// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstddef>
#include <initializer_list>
#include <tuple>

#include <boost/lexical_cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

TEST_CASE("taylor nte basic")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, v] = make_vars("x", "v");

        using ev_t = typename taylor_adaptive<fp_t>::nt_event;

        auto counter = 0u;

        auto ta = taylor_adaptive<fp_t>{
            {prime(x) = v, prime(v) = -9.8 * sin(x)},
            {fp_t(-0.25), fp_t(0.)},
            kw::opt_level = opt_level,
            kw::high_accuracy = high_accuracy,
            kw::compact_mode = compact_mode,
            kw::nt_events = {ev_t(v, [&counter](taylor_adaptive<fp_t> &, fp_t t, std::uint32_t) {
                if (counter == 0u) {
                    REQUIRE(t == 0);
                }

                if (counter == 2u) {
#if defined(HEYOKA_HAVE_REAL128)
                    if constexpr (std::is_same_v<fp_t, mppp::real128>) {
                        using namespace mppp::literals;

                        REQUIRE(t == approximately(2.01495830729551199828007207119092374_rq, fp_t(1000)));
                    } else {
#endif
                        REQUIRE(t
                                == approximately(boost::lexical_cast<fp_t>("2.01495830729551199828007207119092374"),
                                                 fp_t(1000)));
#if defined(HEYOKA_HAVE_REAL128)
                    }
#endif
                }

                ++counter;
            })}};

        for (auto i = 0; i < 20; ++i) {
            ta.step();
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
