// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
