// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <heyoka/callable.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

void blap() {}

int blop(int n)
{
    return n + 1;
}

int blup(int n)
{
    return n + 2;
}

std::vector<int> bar(std::vector<int> &&v)
{
    return std::move(v);
}

struct foo {
    template <typename T>
    auto operator()(T a, T b) const
    {
        return a + b + val;
    }
    int val = 0;
};

struct frob {
    template <typename T>
    auto operator()(T a, T b) const
    {
        return a + b;
    }
    std::vector<int> vec;
};

TEST_CASE("callable basics")
{
    callable<void()> c;
    REQUIRE(!c);

    callable<void()> c2 = c;
    REQUIRE(!c2);

    callable<void()> c3 = std::move(c2);
    REQUIRE(!c2);
    REQUIRE(!c3);

    callable<void()> c4(blap);
    REQUIRE(!!c4);
    c3 = c4;
    REQUIRE(!!c3);
    c = std::move(c4);
    REQUIRE(!!c);

    callable<void()> c5 = c;
    REQUIRE(!!c5);
    REQUIRE(!!c);

    REQUIRE(std::is_nothrow_move_constructible_v<callable<void()>>);
    REQUIRE(std::is_nothrow_move_assignable_v<callable<void()>>);

    // Empty init from nullptr.
    callable<void()> c6 = static_cast<void (*)()>(nullptr);
    REQUIRE(!c6);

    // Empty init from empty std::function.
    callable<void()> c7 = std::function<void()>{};
    REQUIRE(!c7);

    // Empty init from empty callable.
    callable<int(int)> c8 = callable<int(double)>{};
    REQUIRE(!c8);
}

TEST_CASE("callable call")
{
    // Simple test for call and test swappability.
    {
        using call_t = callable<int(int)>;

        call_t c0(blop);

        REQUIRE(c0(4) == 5);

        call_t c1(blup);

        REQUIRE(c1(4) == 6);

        REQUIRE(std::is_nothrow_swappable_v<call_t>);

        using std::swap;
        swap(c0, c1);
        REQUIRE(c1(4) == 5);
        REQUIRE(c0(4) == 6);
    }

    // Test move semantics in the arguments.
    {
        using call_t = callable<std::vector<int>(std::vector<int> &&)>;

        call_t c0(bar);

        std::vector v{1, 2, 3, 4};

        auto ret = c0(std::move(v));

        REQUIRE(v.empty());
        REQUIRE(ret == std::vector{1, 2, 3, 4});
    }

    // Test construction from lambda.
    {
        callable<int(int)> c = [](int n) { return n - 1; };

        REQUIRE(c(9) == 8);
    }

    // Test construction from function object.
    {
        callable<double(double, double)> c0 = foo{5}, c1 = foo{-2};

        REQUIRE(c0(3., 1.) == 9.);
        REQUIRE(c1(3., 1.) == 2.);

        using std::swap;
        swap(c0, c1);

        REQUIRE(c1(3., 1.) == 9.);
        REQUIRE(c0(3., 1.) == 2.);
    }

    // Test move construction from callable.
    {
        frob f{std::vector{1, 2, 3, 4, 5}};

        callable<double(double, double)> c0 = std::move(f);

        REQUIRE(f.vec.empty());
    }

    // Calling an empty callable.
    REQUIRE_THROWS_AS(callable<void()>{}(), std::bad_function_call);
    REQUIRE_THROWS_AS(callable<void()>{std::function<void()>{}}(), std::bad_function_call);

    // Calling an invalid callable.
    struct large_callable {
        std::array<int, 100> arr{};
        void operator()() const {}
    };

    callable<void()> clarge(large_callable{});
    auto cl2 = std::move(clarge);
    REQUIRE_THROWS_AS(clarge(), std::bad_function_call);
    REQUIRE(!clarge);
}

TEST_CASE("callable type idx")
{
    callable<void()> c;

    REQUIRE(c.get_type_index() == typeid(detail::empty_callable));

    auto f = []() {};

    c = callable<void()>{f};

    REQUIRE(c.get_type_index() == typeid(f));
}

struct foo_s11n {
    int addval = 0;

    int operator()(int n) const
    {
        return n + addval;
    }

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & addval;
    }
};

HEYOKA_S11N_CALLABLE_EXPORT(foo_s11n, int, int)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka::detail::empty_callable, int, int)

TEST_CASE("callable s11n")
{
    {
        callable<int(int)> c(foo_s11n{100});

        REQUIRE(c(1) == 101);

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << c;
        }

        c = callable<int(int)>{};
        REQUIRE(!c);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> c;
        }

        REQUIRE(!!c);
        REQUIRE(c(1) == 101);
    }

    // Test an empty callable too.
    {
        callable<int(int)> c;

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);

            oa << c;
        }

        c = callable<int(int)>(foo_s11n{100});
        REQUIRE(c);

        {
            boost::archive::binary_iarchive ia(ss);

            ia >> c;
        }

        REQUIRE(!c);
    }
}

struct vfunc {
    void operator()() const {}
    int n = 0;
};

TEST_CASE("callable extract")
{
    callable<void()> c;
    REQUIRE(c.extract<void (*)()>() == nullptr);
    REQUIRE(c.extract<detail::empty_callable>() != nullptr);
    REQUIRE(std::as_const(c).extract<void (*)()>() == nullptr);

    c = callable<void()>(blap);
    REQUIRE(c.extract<void (*)()>() != nullptr);
    REQUIRE(c.extract<vfunc>() == nullptr);
    REQUIRE(std::as_const(c).extract<void (*)()>() != nullptr);
    REQUIRE(std::as_const(c).extract<vfunc>() == nullptr);

    c = callable<void()>(vfunc{});
    REQUIRE(c.extract<void (*)()>() == nullptr);
    REQUIRE(c.extract<vfunc>() != nullptr);
    REQUIRE(std::as_const(c).extract<void (*)()>() == nullptr);
    REQUIRE(std::as_const(c).extract<vfunc>() != nullptr);

    c = callable<void()>(vfunc{42});
    ++(c.extract<vfunc>()->n);
    REQUIRE(c.extract<vfunc>()->n == 43);
}
