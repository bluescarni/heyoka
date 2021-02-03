// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <functional>
#include <limits>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/number.hpp>

#include "catch.hpp"

using namespace heyoka;

#if defined(HEYOKA_HAVE_REAL128)

using namespace mppp::literals;

#endif

TEST_CASE("number hash eq")
{
    auto hash_number = [](const number &n) { return hash(n); };

    REQUIRE(number{1.1} == number{1.1});
    REQUIRE(number{1.1} != number{1.2});

    REQUIRE(number{1.} == number{1.l});
    REQUIRE(number{1.l} == number{1.});
    REQUIRE(number{0.} == number{-0.l});
    REQUIRE(number{0.l} == number{-0.});
    REQUIRE(hash_number(number{1.l}) == hash_number(number{1.}));
    REQUIRE(hash_number(number{0.l}) == hash_number(number{-0.}));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} == number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<long double>::quiet_NaN()}
            == number{std::numeric_limits<long double>::quiet_NaN()});

    REQUIRE(number{std::numeric_limits<long double>::quiet_NaN()} == number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} == number{std::numeric_limits<long double>::quiet_NaN()});

    REQUIRE(hash_number(number{std::numeric_limits<long double>::quiet_NaN()})
            == hash_number(number{std::numeric_limits<double>::quiet_NaN()}));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{0.l});
    REQUIRE(number{0.l} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-0.l});
    REQUIRE(number{-0.l} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-1.23l});
    REQUIRE(number{1.23l} != number{std::numeric_limits<double>::quiet_NaN()});

#if defined(HEYOKA_HAVE_REAL128)

    REQUIRE(number{1.} == number{1._rq});
    REQUIRE(number{1._rq} == number{1.});
    REQUIRE(number{0.} == number{-0._rq});
    REQUIRE(number{0._rq} == number{-0.});
    REQUIRE(hash_number(number{1._rq}) == hash_number(number{1.}));
    REQUIRE(hash_number(number{0._rq}) == hash_number(number{-0.}));

    REQUIRE(number{1.1} != number{1.1_rq});
    REQUIRE(number{1.1_rq} != number{1.1});

    REQUIRE(number{std::numeric_limits<mppp::real128>::quiet_NaN()}
            == number{std::numeric_limits<mppp::real128>::quiet_NaN()});

    REQUIRE(number{std::numeric_limits<mppp::real128>::quiet_NaN()}
            == number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()}
            == number{std::numeric_limits<mppp::real128>::quiet_NaN()});

    REQUIRE(hash_number(number{std::numeric_limits<mppp::real128>::quiet_NaN()})
            == hash_number(number{std::numeric_limits<double>::quiet_NaN()}));

    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{0._rq});
    REQUIRE(number{0._rq} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-0._rq});
    REQUIRE(number{-0._rq} != number{std::numeric_limits<double>::quiet_NaN()});
    REQUIRE(number{std::numeric_limits<double>::quiet_NaN()} != number{-1.23_rq});
    REQUIRE(number{1.23_rq} != number{std::numeric_limits<double>::quiet_NaN()});

#endif
}
