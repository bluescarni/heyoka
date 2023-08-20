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
