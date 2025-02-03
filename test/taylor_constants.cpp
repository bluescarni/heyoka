// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <initializer_list>
#include <random>
#include <tuple>

#include <boost/math/constants/constants.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/constants.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

static std::mt19937 rng;

using namespace heyoka;
using namespace heyoka_test;
namespace hy = heyoka;

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

// Variable template for the pi constant at different levels of precision.
template <typename T>
const auto pi_const = boost::math::constants::pi<T>();

#if defined(HEYOKA_HAVE_REAL128)

template <>
const mppp::real128 pi_const<mppp::real128> = mppp::pi_128;

#endif

TEST_CASE("taylor constant")
{
    auto tester = [](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto x = "x"_var, y = "y"_var;

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = hy::pi, prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = 1,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == pi_const<fp_t>);
            REQUIRE(jet[3] == 5);
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = hy::pi, prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = 1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == pi_const<fp_t>);
            REQUIRE(jet[5] == pi_const<fp_t>);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);
        }

        {
            auto ta = taylor_adaptive<fp_t>{{prime(x) = hy::pi, prime(y) = x + y},
                                            {fp_t(2), fp_t(3)},
                                            kw::tol = .5,
                                            kw::high_accuracy = high_accuracy,
                                            kw::compact_mode = compact_mode,
                                            kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == 3);
            REQUIRE(jet[2] == pi_const<fp_t>);
            REQUIRE(jet[3] == 5);
            REQUIRE(jet[4] == 0);
            REQUIRE(jet[5] == approximately(fp_t{1} / 2 * (jet[3] + jet[2])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = hy::pi + x, prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{3}, fp_t{-3}},
                                                  2,
                                                  kw::tol = .5,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);

            REQUIRE(jet[2] == 3);
            REQUIRE(jet[3] == -3);

            REQUIRE(jet[4] == pi_const<fp_t> + jet[0]);
            REQUIRE(jet[5] == pi_const<fp_t> + jet[1]);

            REQUIRE(jet[6] == 5);
            REQUIRE(jet[7] == -5);

            REQUIRE(jet[8] == approximately(fp_t{1} / 2 * jet[4]));
            REQUIRE(jet[9] == approximately(fp_t{1} / 2 * jet[5]));

            REQUIRE(jet[10] == approximately(fp_t{1} / 2 * (jet[6] + jet[4])));
            REQUIRE(jet[11] == approximately(fp_t{1} / 2 * (jet[7] + jet[5])));
        }

        {
            auto ta = taylor_adaptive_batch<fp_t>{{prime(x) = hy::pi + x, prime(y) = x + y},
                                                  {fp_t{2}, fp_t{-2}, fp_t{1}, fp_t{3}, fp_t{-3}, fp_t{0}},
                                                  3,
                                                  kw::tol = .1,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode,
                                                  kw::opt_level = opt_level};

            ta.step(true);

            const auto jet = tc_to_jet(ta);

            REQUIRE(jet[0] == 2);
            REQUIRE(jet[1] == -2);
            REQUIRE(jet[2] == 1);

            REQUIRE(jet[3] == 3);
            REQUIRE(jet[4] == -3);
            REQUIRE(jet[5] == 0);

            REQUIRE(jet[6] == approximately(pi_const<fp_t> + jet[0]));
            REQUIRE(jet[7] == approximately(pi_const<fp_t> + jet[1]));
            REQUIRE(jet[8] == approximately(pi_const<fp_t> + jet[2]));

            REQUIRE(jet[9] == 5);
            REQUIRE(jet[10] == -5);
            REQUIRE(jet[11] == 1);

            REQUIRE(jet[12] == approximately(fp_t{1} / 2 * jet[6]));
            REQUIRE(jet[13] == approximately(fp_t{1} / 2 * jet[7]));
            REQUIRE(jet[14] == approximately(fp_t{1} / 2 * jet[8]));

            REQUIRE(jet[15] == approximately(fp_t{1} / 2 * (jet[9] + jet[6])));
            REQUIRE(jet[16] == approximately(fp_t{1} / 2 * (jet[10] + jet[7])));
            REQUIRE(jet[17] == approximately(fp_t{1} / 2 * (jet[11] + jet[8])));

            REQUIRE(jet[18] == approximately(fp_t{1} / 6 * 2 * jet[12]));
            REQUIRE(jet[19] == approximately(fp_t{1} / 6 * 2 * jet[13]));
            REQUIRE(jet[20] == approximately(fp_t{1} / 6 * 2 * jet[14]));

            REQUIRE(jet[21] == approximately(fp_t{1} / 6 * (2 * jet[15] + 2 * jet[12])));
            REQUIRE(jet[22] == approximately(fp_t{1} / 6 * (2 * jet[16] + 2 * jet[13])));
            REQUIRE(jet[23] == approximately(fp_t{1} / 6 * (2 * jet[17] + 2 * jet[14])));
        }

        // Do the batch/scalar comparison.
        compare_batch_scalar<fp_t>({prime(x) = x + hy::pi, prime(y) = x + y}, opt_level, high_accuracy, compact_mode,
                                   rng, .1f, 20.f);
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
