// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdlib>
#include <filesystem>

#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("diskcache envvar")
{
    const auto *env_val = std::getenv("HEYOKA_CACHE_DIR");

    if (env_val == nullptr || *env_val == '\0') {
        // Not set externally — skip.
        WARN("HEYOKA_CACHE_DIR not set, skipping test");
        return;
    }

    // The default disk cache path should match the env variable.
    REQUIRE(llvm_state::get_diskcache_path() == std::filesystem::path(env_val));
}
