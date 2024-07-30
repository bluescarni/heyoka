// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/pendulum.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    REQUIRE(llvm_state::get_memcache_size() == 0u);
    REQUIRE(llvm_state::get_memcache_limit() > 0u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}};

    auto cache_size = llvm_state::get_memcache_size();
    REQUIRE(cache_size > 0u);

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}};

    REQUIRE(llvm_state::get_memcache_size() == cache_size);

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::opt_level = 2u};

    REQUIRE(llvm_state::get_memcache_size() > cache_size);
    cache_size = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::opt_level = 2u, kw::tol = 1e-12};
    REQUIRE(llvm_state::get_memcache_size() > cache_size);

    llvm_state::clear_memcache();
    REQUIRE(llvm_state::get_memcache_size() == 0u);

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}};
    cache_size = llvm_state::get_memcache_size();

    llvm_state::set_memcache_limit(cache_size);
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-12};

    REQUIRE(llvm_state::get_memcache_size() < cache_size);

    llvm_state::set_memcache_limit(0);

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};

    REQUIRE(llvm_state::get_memcache_size() == 0u);
}

TEST_CASE("priority")
{
    // Check that the least recently used items are evicted first.
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    const auto size11 = llvm_state::get_memcache_size();

    llvm_state::clear_memcache();
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-15};
    const auto size15 = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-12};
    const auto size12 = llvm_state::get_memcache_size() - size15;

    llvm_state::set_memcache_limit(llvm_state::get_memcache_size());
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    REQUIRE(llvm_state::get_memcache_size() == size12 + size11);

    // Check that cache hit moves element to the front.
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-15};
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-12};
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-15};

    llvm_state::set_memcache_limit(llvm_state::get_memcache_size());
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    REQUIRE(llvm_state::get_memcache_size() == size15 + size11);
}

// A test to check that the cache shrinks at the first
// insertion attempt after set_memcache_limit().
TEST_CASE("shrink test")
{
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    const auto size11 = llvm_state::get_memcache_size();

    llvm_state::clear_memcache();
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-15};
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-12};
    const auto cache_size = llvm_state::get_memcache_size();

    llvm_state::set_memcache_limit(size11);
    REQUIRE(llvm_state::get_memcache_size() == cache_size);
    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    REQUIRE(llvm_state::get_memcache_size() == size11);
}

// A test to check that the force_avx512 flag is taken
// into account when interacting with the cache.
TEST_CASE("force_avx512 test")
{
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    const auto size11 = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::force_avx512 = true};
    REQUIRE(llvm_state::get_memcache_size() > size11);
}

// Same test for the slp_vectorize option.
TEST_CASE("slp_vectorize test")
{
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    const auto size11 = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::slp_vectorize = true};
    REQUIRE(llvm_state::get_memcache_size() > size11);

    const auto new_size = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{
        model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::slp_vectorize = true, kw::force_avx512 = true};
    REQUIRE(llvm_state::get_memcache_size() > new_size);
}

// Same test for the slp_vectorize option.
TEST_CASE("code_model test")
{
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

    auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11};
    const auto size11 = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::code_model = code_model::medium};
    REQUIRE(llvm_state::get_memcache_size() > size11);

    const auto new_size = llvm_state::get_memcache_size();

    ta = taylor_adaptive<double>{
        model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::code_model = code_model::medium, kw::force_avx512 = true};
    REQUIRE(llvm_state::get_memcache_size() > new_size);
}

// Bug: in compact mode, global variables used to be created in random
// order, which would lead to logically-identical modules considered
// different by the cache machinery due to the different declaration order.
TEST_CASE("bug cache miss compact mode")
{
    {
        llvm_state::clear_memcache();
        llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

        auto ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::compact_mode = true};
        const auto orig_size = llvm_state::get_memcache_size();

        // Re-create the same ta several times and then check the cache size has not changed.
        for (auto i = 0; i < 100; ++i) {
            ta = taylor_adaptive<double>{model::pendulum(), {1., 0.}, kw::tol = 1e-11, kw::compact_mode = true};
        }

        REQUIRE(llvm_state::get_memcache_size() == orig_size);
    }

    {
        llvm_state::clear_memcache();
        llvm_state::set_memcache_limit(2048ull * 1024u * 1024u);

        {
            llvm_state s;
            add_cfunc<double>(s, "func", {model::pendulum_energy()}, {"v"_var, "x"_var}, kw::compact_mode = true);
            s.compile();
            (void)s.jit_lookup("func");
        }

        const auto orig_size = llvm_state::get_memcache_size();

        // Re-create the same cfunc several times and then check the cache size has not changed.
        for (auto i = 0; i < 100; ++i) {
            llvm_state s;
            add_cfunc<double>(s, "func", {model::pendulum_energy()}, {"v"_var, "x"_var}, kw::compact_mode = true);
            s.compile();
            (void)s.jit_lookup("func");
        }

        REQUIRE(llvm_state::get_memcache_size() == orig_size);
    }
}
