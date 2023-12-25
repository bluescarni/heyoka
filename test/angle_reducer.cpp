// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callback/angle_reducer.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

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

TEST_CASE("basics")
{
    using Catch::Matchers::Message;

    // Default construction.
    {
        callback::angle_reducer a0;
        auto a1 = a0;
        auto a2 = std::move(a1);
        a0 = a2;
        a2 = std::move(a0);
    }

    // Ctor from range.
    {
        const std::vector<std::string> vstr{"a", "b", "c"};

        callback::angle_reducer a0(vstr);
        auto a1 = a0;
        auto a2 = std::move(a1);
        a0 = a2;
        a0 = *&a0;
        a2 = std::move(a0);
    }

    // Ctor from init list.
    {
        callback::angle_reducer a0 = {"a", "b", "c"};
        auto a1 = a0;
        auto a2 = std::move(a1);
        a0 = a2;
        a0 = *&a0;
        a2 = std::move(a0);
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(
        callback::angle_reducer(std::vector<std::string>{}), std::invalid_argument,
        Message("The list of expressions passed to the constructor of angle_reducer cannot be empty"));
    REQUIRE_THROWS_MATCHES(
        callback::angle_reducer(std::vector{1_dbl}), std::invalid_argument,
        Message("The list of expressions passed to the constructor of angle_reducer can contain only variables"));
    REQUIRE_THROWS_MATCHES(
        callback::angle_reducer(std::vector{1.f}), std::invalid_argument,
        Message("The list of expressions passed to the constructor of angle_reducer can contain only variables"));
}

TEST_CASE("s11n")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        callback::angle_reducer a0 = {"a", "b", "c"};

        {
            std::stringstream ss;

            step_callback<fp_t> cb(a0);

            {
                boost::archive::binary_oarchive oa(ss);

                oa << cb;
            }

            cb = step_callback<fp_t>{};

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> cb;
            }

            REQUIRE(cb);
            REQUIRE(value_isa<callback::angle_reducer>(cb));
        }

        {
            std::stringstream ss;

            step_callback_batch<fp_t> cb(a0);

            {
                boost::archive::binary_oarchive oa(ss);

                oa << cb;
            }

            cb = step_callback_batch<fp_t>{};

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> cb;
            }

            REQUIRE(cb);
            REQUIRE(value_isa<callback::angle_reducer>(cb));
        }
    };

    tuple_for_each(fp_types, tester);

#if defined(HEYOKA_HAVE_REAL)

    std::stringstream ss;

    callback::angle_reducer a0 = {"a", "b", "c"};

    step_callback<mppp::real> cb(a0);

    {
        boost::archive::binary_oarchive oa(ss);

        oa << cb;
    }

    cb = step_callback<mppp::real>{};

    {
        boost::archive::binary_iarchive ia(ss);

        ia >> cb;
    }

    REQUIRE(cb);
    REQUIRE(value_isa<callback::angle_reducer>(cb));

#endif
}

TEST_CASE("scalar")
{
    using Catch::Matchers::Message;
    using std::cos;

    auto [x0, v0, x1, v1] = make_vars("x0", "v0", "x1", "v1");

    auto tester = [&](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto ta = taylor_adaptive<fp_t>{{prime(x0) = v0, prime(x1) = v1, prime(v0) = -sin(x0), prime(v1) = -sin(x1)},
                                        {fp_t(0.05), fp_t(0.05), fp_t(10), fp_t(10.1)}};

        const auto en0 = fp_t(0.5) * ta.get_state()[2] * ta.get_state()[2] + (1 - cos(ta.get_state()[0]));
        const auto en1 = fp_t(0.5) * ta.get_state()[3] * ta.get_state()[3] + (1 - cos(ta.get_state()[1]));

        step_callback_set<fp_t> scs{callback::angle_reducer({x0, x1}), [](const auto &ta) {
                                        REQUIRE((ta.get_state()[0] >= 0 && ta.get_state()[0] < 6.29));
                                        REQUIRE((ta.get_state()[1] >= 0 && ta.get_state()[1] < 6.29));
                                        return true;
                                    }};

        ta.propagate_until(fp_t(100), kw::callback = scs);

        const auto fen0 = fp_t(0.5) * ta.get_state()[2] * ta.get_state()[2] + (1 - cos(ta.get_state()[0]));
        const auto fen1 = fp_t(0.5) * ta.get_state()[3] * ta.get_state()[3] + (1 - cos(ta.get_state()[1]));

        REQUIRE(fen0 == approximately(en0, fp_t(1000)));
        REQUIRE(fen1 == approximately(en1, fp_t(1000)));

        // Failure modes.
        callback::angle_reducer ar;
        REQUIRE_THROWS_MATCHES(ta.propagate_until(fp_t(200), kw::callback = ar), std::invalid_argument,
                               Message("Cannot use an angle_reducer which was default-constructed or moved-from"));
        REQUIRE_THROWS_MATCHES(ar(ta), std::invalid_argument,
                               Message("Cannot use an angle_reducer which was default-constructed or moved-from"));

        ta = taylor_adaptive<fp_t>{{prime(x0) = v0, prime(x1) = v1, prime(v0) = -sin(x0), prime(v1) = -sin(x1)},
                                   {fp_t(0.05), fp_t(0.05), fp_t(10), fp_t(10.1)}};
        auto cb = std::get<5>(ta.propagate_until(fp_t(20), kw::callback = callback::angle_reducer({x1})));
        ta = taylor_adaptive<fp_t>{{prime(x0) = x0}, {fp_t(0.05)}};
        REQUIRE_THROWS_MATCHES(
            cb(ta), std::invalid_argument,
            Message("Inconsistent state detected in angle_reducer: the last index in the indices vector has a "
                    "value of 1, but the number of state variables is only 1"));
    };

    tuple_for_each(fp_types, tester);

#if defined(HEYOKA_HAVE_REAL)

    {
        using fp_t = mppp::real;

        const auto prec = 237;

        auto ta = taylor_adaptive<fp_t>{{prime(x0) = v0, prime(x1) = v1, prime(v0) = -sin(x0), prime(v1) = -sin(x1)},
                                        {fp_t(0.05, prec), fp_t(0.05, prec), fp_t(10, prec), fp_t(10.1, prec)},
                                        kw::prec = prec};

        const auto en0 = fp_t(0.5, prec) * ta.get_state()[2] * ta.get_state()[2] + (1 - cos(ta.get_state()[0]));
        const auto en1 = fp_t(0.5, prec) * ta.get_state()[3] * ta.get_state()[3] + (1 - cos(ta.get_state()[1]));

        step_callback_set<fp_t> scs{callback::angle_reducer({x0, x1}), [](const auto &ta) {
                                        REQUIRE((ta.get_state()[0] >= 0 && ta.get_state()[0] < 6.29));
                                        REQUIRE((ta.get_state()[1] >= 0 && ta.get_state()[1] < 6.29));
                                        return true;
                                    }};

        ta.propagate_until(fp_t(100, prec), kw::callback = scs);

        const auto fen0 = fp_t(0.5, prec) * ta.get_state()[2] * ta.get_state()[2] + (1 - cos(ta.get_state()[0]));
        const auto fen1 = fp_t(0.5, prec) * ta.get_state()[3] * ta.get_state()[3] + (1 - cos(ta.get_state()[1]));

        REQUIRE(fen0 == approximately(en0, fp_t(1000, prec)));
        REQUIRE(fen1 == approximately(en1, fp_t(1000, prec)));
    }

#endif
}

TEST_CASE("batch")
{
    using Catch::Matchers::Message;
    using std::cos;

    auto [x0, v0, x1, v1] = make_vars("x0", "v0", "x1", "v1");

    auto tester = [&](auto fp_x) {
        using fp_t = decltype(fp_x);

        auto ta = taylor_adaptive_batch<fp_t>{
            {prime(x0) = v0, prime(x1) = v1, prime(v0) = -sin(x0), prime(v1) = -sin(x1)},
            {fp_t(0.05), fp_t(0.06), fp_t(0.05), fp_t(0.05), fp_t(10), fp_t(10.01), fp_t(10.1), fp_t(10.11)},
            2u};

        const auto en0 = fp_t(0.5) * ta.get_state()[4] * ta.get_state()[4] + (1 - cos(ta.get_state()[0]));
        const auto en1 = fp_t(0.5) * ta.get_state()[6] * ta.get_state()[6] + (1 - cos(ta.get_state()[2]));

        step_callback_batch_set<fp_t> scs{callback::angle_reducer({x0, x1}), [](const auto &ta) {
                                              REQUIRE((ta.get_state()[0] >= 0 && ta.get_state()[0] < 6.29));
                                              REQUIRE((ta.get_state()[1] >= 0 && ta.get_state()[1] < 6.29));
                                              REQUIRE((ta.get_state()[2] >= 0 && ta.get_state()[2] < 6.29));
                                              REQUIRE((ta.get_state()[3] >= 0 && ta.get_state()[3] < 6.29));
                                              return true;
                                          }};

        ta.propagate_until(fp_t(100), kw::callback = scs);

        const auto fen0 = fp_t(0.5) * ta.get_state()[4] * ta.get_state()[4] + (1 - cos(ta.get_state()[0]));
        const auto fen1 = fp_t(0.5) * ta.get_state()[6] * ta.get_state()[6] + (1 - cos(ta.get_state()[2]));

        REQUIRE(fen0 == approximately(en0, fp_t(1000)));
        REQUIRE(fen1 == approximately(en1, fp_t(1000)));

        // Failure modes.
        callback::angle_reducer ar;
        REQUIRE_THROWS_MATCHES(ta.propagate_until(fp_t(200), kw::callback = ar), std::invalid_argument,
                               Message("Cannot use an angle_reducer which was default-constructed or moved-from"));
        REQUIRE_THROWS_MATCHES(ar(ta), std::invalid_argument,
                               Message("Cannot use an angle_reducer which was default-constructed or moved-from"));

        ta = taylor_adaptive_batch<fp_t>{
            {prime(x0) = v0, prime(x1) = v1, prime(v0) = -sin(x0), prime(v1) = -sin(x1)},
            {fp_t(0.05), fp_t(0.06), fp_t(0.05), fp_t(0.05), fp_t(10), fp_t(10.01), fp_t(10.1), fp_t(10.11)},
            2u};
        auto cb = std::get<1>(ta.propagate_until(fp_t(20), kw::callback = callback::angle_reducer({x1})));
        ta = taylor_adaptive_batch<fp_t>{{prime(x0) = x0}, {fp_t(0.05), fp_t(0.06)}, 2u};
        REQUIRE_THROWS_MATCHES(
            cb(ta), std::invalid_argument,
            Message("Inconsistent state detected in angle_reducer: the last index in the indices vector has a "
                    "value of 1, but the number of state variables is only 1"));
    };

    tuple_for_each(fp_types, tester);
}

TEST_CASE("stream operator")
{
    {
        std::ostringstream oss;
        oss << callback::angle_reducer{};

        REQUIRE(oss.str() == "Angle reducer (default constructed)");
    }

    {
        std::ostringstream oss;
        oss << callback::angle_reducer{{"x"_var, "y"_var, "z"_var}};

        REQUIRE(boost::algorithm::contains(oss.str(), "Angle reducer: "));
        REQUIRE(boost::algorithm::contains(oss.str(), "x"));
        REQUIRE(boost::algorithm::contains(oss.str(), "y"));
        REQUIRE(boost::algorithm::contains(oss.str(), "z"));
    }
}
