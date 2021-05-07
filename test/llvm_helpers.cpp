// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <stdexcept>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("while_loop")
{
    for (auto opt_level : {0u, 1u, 2u, 3u}) {
        llvm_state s{kw::opt_level = opt_level};

        detail::llvm_while_loop_test0(s);

        // Fetch the function pointerr
        auto f_ptr = reinterpret_cast<std::uint32_t (*)(std::uint32_t)>(s.jit_lookup("count_n"));

        REQUIRE(f_ptr(0) == 0u);
        REQUIRE(f_ptr(1) == 1u);
        REQUIRE(f_ptr(2) == 2u);
        REQUIRE(f_ptr(3) == 3u);
        REQUIRE(f_ptr(4) == 4u);
    }

#if 0
    // Error handling.
    {
        llvm_state s;

        REQUIRE_THROWS_AS(detail::llvm_while_loop_test1(s), std::runtime_error);
    }

    {
        llvm_state s;

        REQUIRE_THROWS_AS(detail::llvm_while_loop_test2(s), std::runtime_error);
    }
#endif
}
