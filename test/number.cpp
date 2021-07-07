// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <functional>
#include <initializer_list>
#include <limits>
#include <sstream>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

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
    REQUIRE(number{1.1} != number{1.2l});
    REQUIRE(number{1.2l} != number{1.1});
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
    REQUIRE(number{1.1} != number{1.2_rq});
    REQUIRE(number{1.2_rq} != number{1.1});
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

    // Verify that subexpressions which differ only
    // by the type of the constants (but not their values)
    // are correctly simplified.
    auto [x, y] = make_vars("x", "y");

    llvm_state s{kw::opt_level = 0u};

    auto dc = taylor_add_jet<double>(s, "jet", {prime(x) = (y + 1.) + (y + 1.l), prime(y) = x}, 1, 1, false, true);

    REQUIRE(dc.size() == 6u);

    // Make sure the vector of constants has been
    // optimised out because both constants are 1.
    REQUIRE(!boost::contains(s.get_ir(), "internal constant [2 x double]"));
}

TEST_CASE("number s11n")
{
    std::stringstream ss;

    number n{4.5l};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{4.5l});

    ss.str("");

    n = number{1.2};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.l};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.2});

#if defined(HEYOKA_HAVE_REAL128)
    ss.str("");

    n = number{1.1_rq};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0.};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.1_rq});

    ss.str("");

    n = number{1.1};

    {
        boost::archive::binary_oarchive oa(ss);

        oa << n;
    }

    n = number{0._rq};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> n;
    }

    REQUIRE(n == number{1.1});
#endif
}
