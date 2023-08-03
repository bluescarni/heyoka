// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <functional>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

template <typename TA>
bool cb0(TA &)
{
    return true;
}

struct cb1 {
    template <typename TA>
    bool operator()(TA &ta)
    {
        ta.get_state_data()[0] = 2;
        return false;
    }

    template <typename TA>
    void pre_hook(TA &ta)
    {
        ta.get_state_data()[0] = 1;
    }
};

TEST_CASE("step_callback basics")
{
    using Catch::Matchers::Message;

    taylor_adaptive<double> ta;

    {
        step_callback<double> step_cb;

        REQUIRE(!step_cb);

        REQUIRE_THROWS_AS(step_cb(ta), std::bad_function_call);
        REQUIRE_THROWS_AS(step_cb.pre_hook(ta), std::bad_function_call);

        REQUIRE(std::is_nothrow_swappable_v<step_callback<double>>);

        REQUIRE(!std::is_constructible_v<step_callback<double>, void>);
        REQUIRE(!std::is_constructible_v<step_callback<double>, int, int>);

#if !defined(_MSC_VER) || defined(__clang__)

        // NOTE: vanilla MSVC does not like these extraction.
        REQUIRE(step_cb.extract<int>() == nullptr);
        REQUIRE(std::as_const(step_cb).extract<int>() == nullptr);
#endif

        REQUIRE(step_cb.get_type_index() == typeid(void));

        // Copy construction of empty callback.
        auto step_cb2 = step_cb;
        REQUIRE(!step_cb2);

        // Move construction of empty callback.
        auto step_cb3 = std::move(step_cb);
        REQUIRE(!step_cb3);
    }

    {
        auto lam = [](auto &) { return true; };

        step_callback<double> step_cb(lam);

        REQUIRE(static_cast<bool>(step_cb));

        REQUIRE(step_cb(ta));
        REQUIRE_NOTHROW(step_cb.pre_hook(ta));

#if !defined(_MSC_VER) || defined(__clang__)

        REQUIRE(step_cb.extract<int>() == nullptr);
        REQUIRE(std::as_const(step_cb).extract<int>() == nullptr);

#endif

        REQUIRE(step_cb.extract<decltype(lam)>() != nullptr);
        REQUIRE(std::as_const(step_cb).extract<decltype(lam)>() != nullptr);

        // Copy construction.
        auto step_cb2 = step_cb;
        REQUIRE(step_cb2);
        REQUIRE(step_cb2.extract<decltype(lam)>() != nullptr);
        REQUIRE(step_cb2.extract<decltype(lam)>() != step_cb.extract<decltype(lam)>());

        // Move construction.
        auto step_cb3 = std::move(step_cb);
        REQUIRE(step_cb3);
        REQUIRE(step_cb3.extract<decltype(lam)>() != nullptr);
        REQUIRE(!step_cb);

        // Revive step_cb via copy assignment.
        step_cb = step_cb3;
        REQUIRE(step_cb);
        REQUIRE(step_cb.extract<decltype(lam)>() != nullptr);
        REQUIRE(step_cb.extract<decltype(lam)>() != step_cb3.extract<decltype(lam)>());

        // Revive step_cb via move assignment.
        const auto *orig_ptr = step_cb.extract<decltype(lam)>();
        auto step_cb4 = std::move(step_cb);
        step_cb = std::move(step_cb4);
        REQUIRE(!step_cb4);
        REQUIRE(step_cb.extract<decltype(lam)>() != nullptr);
        REQUIRE(step_cb.extract<decltype(lam)>() != step_cb3.extract<decltype(lam)>());
        REQUIRE(step_cb.extract<decltype(lam)>() == orig_ptr);
    }

    {
        step_callback<double> step_cb(&cb0<taylor_adaptive<double>>);

        REQUIRE(static_cast<bool>(step_cb));

        REQUIRE(step_cb(ta));
        REQUIRE_NOTHROW(step_cb.pre_hook(ta));
    }

    {
        step_callback<double> step_cb(cb0<taylor_adaptive<double>>);

        REQUIRE(static_cast<bool>(step_cb));

        REQUIRE(step_cb(ta));
        REQUIRE_NOTHROW(step_cb.pre_hook(ta));
    }

    {
        step_callback<double> step_cb(cb1{});

        REQUIRE(ta.get_state()[0] == 0.);

        REQUIRE(static_cast<bool>(step_cb));

        REQUIRE(!step_cb(ta));
        REQUIRE(ta.get_state()[0] == 2.);

        REQUIRE_NOTHROW(step_cb.pre_hook(ta));
        REQUIRE(ta.get_state()[0] == 1.);

        ta.get_state_data()[0] = 0;
    }

    {
        step_callback<double> step_cb([](auto &ta) {
            ta.get_state_data()[0] = 3;
            return true;
        });

        REQUIRE(ta.get_state()[0] == 0.);

        REQUIRE(static_cast<bool>(step_cb));

        REQUIRE(step_cb(ta));
        REQUIRE(ta.get_state()[0] == 3.);

        REQUIRE_NOTHROW(step_cb.pre_hook(ta));
        REQUIRE(ta.get_state()[0] == 3.);

        ta.get_state_data()[0] = 0;
    }

    {
        using std::swap;

        step_callback<double> step_cb1(cb1{}), step_cb2;

        swap(step_cb1, step_cb2);

        REQUIRE(static_cast<bool>(step_cb2));
        REQUIRE(!static_cast<bool>(step_cb1));

        REQUIRE(step_cb2.extract<cb1>() != nullptr);
        REQUIRE(step_cb1.extract<cb1>() == nullptr);
    }
}

struct cb2 {
    template <typename TA>
    bool operator()(TA &)
    {
        return false;
    }

    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

HEYOKA_S11N_STEP_CALLBACK_EXPORT(cb2, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(cb2, double)

TEST_CASE("step_callback s11n")
{
    {
        step_callback<double> step_cb(cb2{});

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << step_cb;
        }

        step_cb = step_callback<double>{};
        REQUIRE(!step_cb);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> step_cb;
        }

        REQUIRE(!!step_cb);
        REQUIRE(step_cb.extract<cb2>() != nullptr);
    }

    {
        step_callback<double> step_cb;

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << step_cb;
        }

        step_cb = step_callback<double>{cb2{}};
        REQUIRE(step_cb);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> step_cb;
        }

        REQUIRE(!step_cb);
    }

    {
        step_callback_batch<double> step_cb(cb2{});

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << step_cb;
        }

        step_cb = step_callback_batch<double>{};
        REQUIRE(!step_cb);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> step_cb;
        }

        REQUIRE(!!step_cb);
        REQUIRE(step_cb.extract<cb2>() != nullptr);
    }

    {
        step_callback_batch<double> step_cb;

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << step_cb;
        }

        step_cb = step_callback_batch<double>{cb2{}};
        REQUIRE(step_cb);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> step_cb;
        }

        REQUIRE(!step_cb);
    }
}
