// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/pendulum.hpp>
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

struct only_ph {
    template <typename TA>
    void pre_hook(TA &)
    {
    }
};

TEST_CASE("step_callback basics")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        taylor_adaptive<fp_t> ta;

        {
            step_callback<fp_t> step_cb;

            REQUIRE(!step_cb);

            REQUIRE_THROWS_AS(step_cb(ta), std::bad_function_call);

            REQUIRE(std::is_nothrow_swappable_v<step_callback<fp_t>>);

            REQUIRE(!std::is_constructible_v<step_callback<fp_t>, void>);
            REQUIRE(!std::is_constructible_v<step_callback<fp_t>, int, int>);
            REQUIRE(!std::is_constructible_v<step_callback<fp_t>, only_ph>);

            REQUIRE(step_cb.get_type_index() == typeid(detail::empty_callable));

            // Copy construction of empty callback.
            auto step_cb2 = step_cb;
            REQUIRE(!step_cb2);

            // Move construction of empty callback.
            auto step_cb3 = std::move(step_cb);
            REQUIRE(!step_cb3);

            // Empty init from nullptr.
            step_callback<fp_t> c6 = static_cast<bool (*)(taylor_adaptive<fp_t> &)>(nullptr);
            REQUIRE(!c6);

            // Empty init from empty std::function.
            step_callback<fp_t> c7 = std::function<bool(taylor_adaptive<fp_t> &)>{};
            REQUIRE(!c7);

            // Empty init from empty callable.
            step_callback<fp_t> c8 = callable<bool(taylor_adaptive<fp_t> &)>{};
            REQUIRE(!c8);
        }

        {
            auto lam = [](auto &) { return true; };

            step_callback<fp_t> step_cb(lam);

            REQUIRE(static_cast<bool>(step_cb));

            REQUIRE(step_cb(ta));
            REQUIRE_NOTHROW(step_cb.pre_hook(ta));

            REQUIRE(step_cb.get_type_index() == typeid(decltype(lam)));

            REQUIRE(step_cb.template extract<decltype(lam)>() != nullptr);
            REQUIRE(std::as_const(step_cb).template extract<decltype(lam)>() != nullptr);

            // Copy construction.
            auto step_cb2 = step_cb;
            REQUIRE(step_cb2);
            REQUIRE(step_cb2.template extract<decltype(lam)>() != nullptr);
            REQUIRE(step_cb2.template extract<decltype(lam)>() != step_cb.template extract<decltype(lam)>());

            // Move construction.
            auto step_cb3 = std::move(step_cb);
            REQUIRE(step_cb3);
            REQUIRE(step_cb3.template extract<decltype(lam)>() != nullptr);

            // Revive step_cb via copy assignment.
            step_cb = step_cb3;
            REQUIRE(step_cb);
            REQUIRE(step_cb.template extract<decltype(lam)>() != nullptr);
            REQUIRE(step_cb.template extract<decltype(lam)>() != step_cb3.template extract<decltype(lam)>());

            // Revive step_cb via move assignment.
            const auto *orig_ptr = step_cb.template extract<decltype(lam)>();
            auto step_cb4 = std::move(step_cb);
            step_cb = std::move(step_cb4);
            REQUIRE(step_cb.template extract<decltype(lam)>() != nullptr);
            REQUIRE(step_cb.template extract<decltype(lam)>() != step_cb3.template extract<decltype(lam)>());
            REQUIRE(step_cb.template extract<decltype(lam)>() == orig_ptr);
        }

        {
            step_callback<fp_t> step_cb(&cb0<taylor_adaptive<fp_t>>);

            REQUIRE(static_cast<bool>(step_cb));

            REQUIRE(step_cb.get_type_index() == typeid(decltype(&cb0<taylor_adaptive<fp_t>>)));

            REQUIRE(step_cb(ta));
            REQUIRE_NOTHROW(step_cb.pre_hook(ta));
        }

        {
            step_callback<fp_t> step_cb(cb0<taylor_adaptive<fp_t>>);

            REQUIRE(static_cast<bool>(step_cb));

            REQUIRE(step_cb(ta));
            REQUIRE_NOTHROW(step_cb.pre_hook(ta));
        }

        {
            step_callback<fp_t> step_cb(cb1{});

            REQUIRE(ta.get_state()[0] == 0.);

            REQUIRE(static_cast<bool>(step_cb));

            REQUIRE(!step_cb(ta));
            REQUIRE(ta.get_state()[0] == 2.);

            REQUIRE_NOTHROW(step_cb.pre_hook(ta));
            REQUIRE(ta.get_state()[0] == 1.);

            ta.get_state_data()[0] = 0;
        }

        // Same test as above, but using reference wrapper.
        {
            cb1 orig_cb1;

            step_callback<fp_t> step_cb(std::ref(orig_cb1));

            REQUIRE(ta.get_state()[0] == 0.);

            REQUIRE(static_cast<bool>(step_cb));

            REQUIRE(!step_cb(ta));
            REQUIRE(ta.get_state()[0] == 2.);

            REQUIRE_NOTHROW(step_cb.pre_hook(ta));
            REQUIRE(ta.get_state()[0] == 1.);

            ta.get_state_data()[0] = 0;
        }

        {
            step_callback<fp_t> step_cb([](auto &ta) {
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

            step_callback<fp_t> step_cb1(cb1{}), step_cb2;

            swap(step_cb1, step_cb2);

            REQUIRE(static_cast<bool>(step_cb2));
            REQUIRE(!static_cast<bool>(step_cb1));

            REQUIRE(step_cb2.template extract<cb1>() != nullptr);
            REQUIRE(step_cb1.template extract<cb1>() == nullptr);
        }
    };

    tuple_for_each(fp_types, tester);
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

HEYOKA_S11N_STEP_CALLBACK_EXPORT(cb2, float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(cb2, float)

HEYOKA_S11N_STEP_CALLBACK_EXPORT(cb2, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(cb2, double)

#if !defined(HEYOKA_ARCH_PPC)

HEYOKA_S11N_STEP_CALLBACK_EXPORT(cb2, long double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(cb2, long double)

#endif

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT(cb2, mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(cb2, mppp::real128)

#endif

TEST_CASE("step_callback s11n")
{
    auto tester = [](auto fp_x) {
        using fp_t = decltype(fp_x);

        {
            step_callback<fp_t> step_cb(cb2{});

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << step_cb;
            }

            step_cb = step_callback<fp_t>{};
            REQUIRE(!step_cb);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> step_cb;
            }

            REQUIRE(!!step_cb);
            REQUIRE(step_cb.template extract<cb2>() != nullptr);
        }

        // Empty step callback test.
        {
            step_callback<fp_t> step_cb;

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << step_cb;
            }

            step_cb = step_callback<fp_t>{cb2{}};
            REQUIRE(step_cb);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> step_cb;
            }

            REQUIRE(!step_cb);
        }

        {
            step_callback_batch<fp_t> step_cb(cb2{});

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << step_cb;
            }

            step_cb = step_callback_batch<fp_t>{};
            REQUIRE(!step_cb);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> step_cb;
            }

            REQUIRE(!!step_cb);
            REQUIRE(step_cb.template extract<cb2>() != nullptr);
        }

        // Empty step callback test.
        {
            step_callback_batch<fp_t> step_cb;

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << step_cb;
            }

            step_cb = step_callback_batch<fp_t>{cb2{}};
            REQUIRE(step_cb);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> step_cb;
            }

            REQUIRE(!step_cb);
        }
    };

    tuple_for_each(fp_types, tester);
}

struct pend_cb {
    template <typename TA>
    bool operator()(TA &)
    {
        return true;
    }

    void pre_hook(taylor_adaptive<double> &ta)
    {
        ta.get_pars_data()[0] = 1.5;
    }

    void pre_hook(taylor_adaptive_batch<double> &ta)
    {
        ta.get_pars_data()[0] = 1.5;
        ta.get_pars_data()[1] = 1.5;
    }
};

struct tm_cb {
    template <typename TA>
    bool operator()(TA &)
    {
        return true;
    }

    void pre_hook(taylor_adaptive<double> &ta)
    {
        ta.set_time(ta.get_time() + 1);
    }

    void pre_hook(taylor_adaptive_batch<double> &ta)
    {
        ta.set_time({ta.get_time()[0] + 1, ta.get_time()[1] + 1});
    }
};

TEST_CASE("step_callback pre_hook")
{
    using Catch::Matchers::Message;

    auto dyn = model::pendulum(kw::length = par[0]);

    {
        auto ta0 = taylor_adaptive<double>{dyn, {1., 0.}};
        auto ta1 = taylor_adaptive<double>{dyn, {1., 0.}, kw::pars = {1.5}};

        REQUIRE(ta0.get_pars()[0] == 0.);

        ta0.propagate_until(3., kw::callback = pend_cb{});
        ta1.propagate_until(3.);

        REQUIRE(ta0.get_pars()[0] == 1.5);

        REQUIRE(ta0.get_state() == ta1.get_state());

        REQUIRE_THROWS_MATCHES(
            ta0.propagate_until(6., kw::callback = tm_cb{}), std::runtime_error,
            Message("The invocation of the callback passed to propagate_until() resulted in the alteration of the "
                    "time coordinate of the integrator - this is not supported"));

        REQUIRE(ta0.get_time() == 4);

        ta0.set_time(0.);
        ta0.get_pars_data()[0] = 0.1;
        ta1.set_time(0.);

        auto res0 = ta0.propagate_grid({0., 1., 2.}, kw::callback = pend_cb{});
        auto res1 = ta1.propagate_grid({0., 1., 2.});

        REQUIRE(std::get<4>(res0));
        REQUIRE(!std::get<4>(res1));
        REQUIRE(std::get<5>(res0)[0] == std::get<5>(res1)[0]);

        REQUIRE(ta0.get_pars()[0] == 1.5);

        REQUIRE_THROWS_MATCHES(
            ta0.propagate_grid({ta0.get_time(), ta0.get_time() + 1}, kw::callback = tm_cb{}), std::runtime_error,
            Message("The invocation of the callback passed to propagate_grid() resulted in the alteration of the "
                    "time coordinate of the integrator - this is not supported"));
    }

    {
        auto ta0 = taylor_adaptive_batch<double>{dyn, {1., 1.1, 0., 0.1}, 2u};
        auto ta1 = taylor_adaptive_batch<double>{dyn, {1., 1.1, 0., 0.1}, 2u, kw::pars = {1.5, 1.5}};

        REQUIRE(ta0.get_pars()[0] == 0.);
        REQUIRE(ta0.get_pars()[1] == 0.);

        ta0.propagate_until(3., kw::callback = pend_cb{});
        ta1.propagate_until(3.);

        REQUIRE(ta0.get_pars()[0] == 1.5);
        REQUIRE(ta0.get_pars()[1] == 1.5);

        REQUIRE(ta0.get_state() == ta1.get_state());

        REQUIRE_THROWS_MATCHES(
            ta0.propagate_until(6., kw::callback = tm_cb{}), std::runtime_error,
            Message("The invocation of the callback passed to propagate_until() resulted in the alteration of the "
                    "time coordinate of the integrator - this is not supported"));

        REQUIRE(ta0.get_time() == std::vector{4., 4.});

        ta0.set_time(0.);
        ta0.get_pars_data()[0] = 0.1;
        ta0.get_pars_data()[1] = 0.1;
        ta1.set_time(0.);

        auto res0 = ta0.propagate_grid({0., 0., 1., 1., 2., 2.}, kw::callback = pend_cb{});
        auto res1 = ta1.propagate_grid({0., 0., 1., 1., 2., 2.});

        REQUIRE(res0 == res1);

        REQUIRE(ta0.get_pars()[0] == 1.5);
        REQUIRE(ta0.get_pars()[1] == 1.5);

        REQUIRE_THROWS_MATCHES(
            ta0.propagate_grid({ta0.get_time()[0], ta0.get_time()[1], 4., 4.}, kw::callback = tm_cb{}),
            std::runtime_error,
            Message("The invocation of the callback passed to propagate_grid() resulted in the alteration of the "
                    "time coordinate of the integrator - this is not supported"));
    }
}

TEST_CASE("step_callback_set")
{
    using Catch::Matchers::Message;
    using std::swap;

    auto dyn = model::pendulum();

    auto tester = [&](auto fp_x) {
        using fp_t = decltype(fp_x);

        {
            // Swappability.
            REQUIRE(std::is_nothrow_swappable_v<step_callback_set<fp_t>>);
            REQUIRE(std::is_nothrow_swappable_v<step_callback_batch_set<fp_t>>);

            // Basic API.
            step_callback_set<fp_t> scs;
            REQUIRE(scs.size() == 0u);
            REQUIRE_THROWS_MATCHES(scs[0], std::out_of_range,
                                   Message("Out of range index 0 when accessing a step callback set of size 0"));
            REQUIRE_THROWS_MATCHES(std::as_const(scs)[0], std::out_of_range,
                                   Message("Out of range index 0 when accessing a step callback set of size 0"));
            auto scs2 = step_callback_set<fp_t>{[](const auto &) { return true; }};
            REQUIRE(scs2.size() == 1u);
            REQUIRE_NOTHROW(scs2[0]);
            REQUIRE_NOTHROW(std::as_const(scs2)[0]);
            REQUIRE_THROWS_MATCHES(scs2[10], std::out_of_range,
                                   Message("Out of range index 10 when accessing a step callback set of size 1"));

            swap(scs, scs2);
            REQUIRE(scs.size() == 1u);
            REQUIRE(scs2.size() == 0u);
            REQUIRE_NOTHROW(scs[0]);
            REQUIRE_NOTHROW(std::as_const(scs)[0]);
            REQUIRE_THROWS_MATCHES(scs[10], std::out_of_range,
                                   Message("Out of range index 10 when accessing a step callback set of size 1"));

            auto scs3 = scs;
            REQUIRE(scs3.size() == 1u);

            auto scs4 = std::move(scs);
            REQUIRE(scs4.size() == 1u);

            scs = scs4;
            REQUIRE(scs.size() == 1u);

            scs2 = std::move(scs);
            REQUIRE(scs2.size() == 1u);
        }

        // Empty set.
        {
            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            const auto oc = std::get<0>(ta0.propagate_until(10., kw::callback = step_callback_set<fp_t>{}));

            REQUIRE(oc == taylor_outcome::time_limit);
        }

        // Check sequencing of callback invocations.
        {
            int c1 = 0;
            int c2 = 0;

            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            const auto oc
                = std::get<0>(ta0.propagate_until(10., kw::callback = step_callback_set<fp_t>{[&c1, &c2](const auto &) {
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  ++c1;
                                                                                                  return true;
                                                                                              },
                                                                                              [&c1, &c2](const auto &) {
                                                                                                  ++c2;
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  return true;
                                                                                              }}));

            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(c1 == c2);
        }

        // Check stopping.
        {
            int c1 = 0;
            int c2 = 0;

            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            const auto oc
                = std::get<0>(ta0.propagate_until(10., kw::callback = step_callback_set<fp_t>{[&c1, &c2](const auto &) {
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  ++c1;
                                                                                                  return false;
                                                                                              },
                                                                                              [&c1, &c2](const auto &) {
                                                                                                  ++c2;
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  return true;
                                                                                              }}));

            REQUIRE(oc == taylor_outcome::cb_stop);
            REQUIRE(c1 == c2);
        }

        {
            int c1 = 0;
            int c2 = 0;

            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            const auto oc
                = std::get<0>(ta0.propagate_until(10., kw::callback = step_callback_set<fp_t>{[&c1, &c2](const auto &) {
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  ++c1;
                                                                                                  return true;
                                                                                              },
                                                                                              [&c1, &c2](const auto &) {
                                                                                                  ++c2;
                                                                                                  REQUIRE(c1 == c2);
                                                                                                  return false;
                                                                                              }}));

            REQUIRE(oc == taylor_outcome::cb_stop);
            REQUIRE(c1 == c2);
        }

        // Pre-hook invocation.
        {
            int a = 0;
            int b = 0;

            int h1 = 0;
            int h2 = 0;

            struct my_cb1 {
                int &c1;
                int &c2;
                int &h;

                bool operator()(taylor_adaptive<fp_t> &)
                {
                    REQUIRE(c1 == c2);
                    ++c1;
                    return true;
                }

                void pre_hook(taylor_adaptive<fp_t> &)
                {
                    REQUIRE(h == 0);
                    ++h;
                }
            };

            struct my_cb2 {
                int &c1;
                int &c2;
                int &h;

                bool operator()(taylor_adaptive<fp_t> &)
                {
                    ++c2;
                    REQUIRE(c1 == c2);
                    return true;
                }

                void pre_hook(taylor_adaptive<fp_t> &)
                {
                    REQUIRE(h == 0);
                    ++h;
                }
            };

            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            const auto oc = std::get<0>(
                ta0.propagate_until(10., kw::callback = step_callback_set<fp_t>{my_cb1{a, b, h1}, my_cb2{a, b, h2}}));

            REQUIRE(oc == taylor_outcome::time_limit);
            REQUIRE(a == b);
            REQUIRE(h1 == 1);
            REQUIRE(h2 == 1);
        }

        // Error handling.
        {
            auto ta0 = taylor_adaptive<fp_t>{dyn, {1., 0.}};

            REQUIRE_THROWS_MATCHES(
                ta0.propagate_until(6., kw::callback = step_callback_set<fp_t>{step_callback<fp_t>{}}),
                std::invalid_argument,
                Message("Cannot construct a callback set containing one or more empty callbacks"));
            REQUIRE_THROWS_MATCHES(
                ta0.propagate_until(6., kw::callback = step_callback_set<fp_t>{step_callback<fp_t>{},
                                                                               [](const auto &) { return true; }}),
                std::invalid_argument,
                Message("Cannot construct a callback set containing one or more empty callbacks"));
            REQUIRE_THROWS_MATCHES(
                ta0.propagate_until(6., kw::callback = step_callback_set<fp_t>{[](const auto &) { return true; },
                                                                               step_callback<fp_t>{}}),
                std::invalid_argument,
                Message("Cannot construct a callback set containing one or more empty callbacks"));
        }

        // Serialisation.
        {
            step_callback<fp_t> scs{step_callback_set<fp_t>{cb2{}}};

            std::stringstream ss;

            {
                boost::archive::binary_oarchive oa(ss);

                oa << scs;
            }

            scs = step_callback<fp_t>{};
            REQUIRE(!scs);

            {
                boost::archive::binary_iarchive ia(ss);

                ia >> scs;
            }

            REQUIRE(static_cast<bool>(scs));
            REQUIRE(scs.template extract<step_callback_set<fp_t>>() != nullptr);
        }
    };

    tuple_for_each(fp_types, tester);
}
