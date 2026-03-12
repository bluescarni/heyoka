// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <filesystem>
#include <stdexcept>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

namespace
{

// RAII helper to set up a temporary disk cache directory.
//
// On construction, it will generate a temporary directory and set the disk cache path to that directory. On
// destruction, it will restore the original disk cache path and remove the temp dir.
struct tmp_diskcache_dir {
    std::filesystem::path orig_path;
    std::filesystem::path tmp_path;

    tmp_diskcache_dir()
    {
        orig_path = llvm_state::get_diskcache_path();

        const auto bp = boost::filesystem::temp_directory_path()
                        / boost::filesystem::unique_path("heyoka_diskcache_test_%%%%_%%%%");
        tmp_path = bp.string();
        std::filesystem::create_directories(tmp_path);

        // NOTE: set_diskcache_path() could in principle throw, in which case the directory cleanup in the dtor would
        // not happen because the tmp_diskcache_dir has not been fully constructed yet. Thus, the explicit cleanup
        // before re-throwing.
        try {
            llvm_state::set_diskcache_path(tmp_path);
        } catch (...) {
            cleanup();

            throw;
        }
    }

    ~tmp_diskcache_dir()
    {
        // NOTE: make sure that exceptions raised here don't prevent the temp dir cleanup.
        try {
            llvm_state::set_diskcache_path(orig_path);
        } catch (...) {
            ;
        }

        cleanup();
    }

    void cleanup() const
    {
        std::filesystem::remove_all(tmp_path);
    }

    tmp_diskcache_dir(const tmp_diskcache_dir &) = delete;
    tmp_diskcache_dir(tmp_diskcache_dir &&) = delete;
    tmp_diskcache_dir &operator=(const tmp_diskcache_dir &) = delete;
    tmp_diskcache_dir &operator=(tmp_diskcache_dir &&) = delete;
};

} // namespace

// NOTE: in these tests it is imperative that we do not *ever* touch the default on-disk cache of the user (i.e., the
// one in /home/user/...). For this reason, we must always use the tmp_diskcache_dir machinery within each test, unless
// we are 100% sure we are invoking functions that cannot affect the cache state.

// NOTE: set/get_diskcache_path() do not touch the database. At most, set_diskcache_path() will reset the current
// db connection if existing.
TEST_CASE("diskcache path")
{
    const auto orig_path = llvm_state::get_diskcache_path();

    {
        tmp_diskcache_dir tdd;

        const auto new_path = llvm_state::get_diskcache_path();

        REQUIRE(new_path == tdd.tmp_path);
    }

    REQUIRE(llvm_state::get_diskcache_path() == orig_path);
}

// NOTE: get/set_diskcache_enabled() are simple boolean getter/setter, no interaction with the db.
TEST_CASE("diskcache enabled")
{
    llvm_state::set_diskcache_enabled(true);
    REQUIRE(llvm_state::get_diskcache_enabled());

    llvm_state::set_diskcache_enabled(false);
    REQUIRE(!llvm_state::get_diskcache_enabled());
}

TEST_CASE("diskcache limit")
{
    using Catch::Matchers::Message;

    {
        tmp_diskcache_dir tdd;

        // Default limit should be positive.
        const auto limit = llvm_state::get_diskcache_limit();
        REQUIRE(limit > 0);

        // Set a new limit.
        llvm_state::set_diskcache_limit(1000000);
        REQUIRE(llvm_state::get_diskcache_limit() == 1000000);

        // Restore.
        llvm_state::set_diskcache_limit(limit);
        REQUIRE(llvm_state::get_diskcache_limit() == limit);

        // Negative limit throws.
        REQUIRE_THROWS_MATCHES(llvm_state::set_diskcache_limit(-1), std::invalid_argument,
                               Message("Invalid negative size limit for the llvm_state on-disk cache: -1"));

        REQUIRE_THROWS_AS(llvm_state::set_diskcache_limit(-1), std::invalid_argument);
    }
}

TEST_CASE("diskcache size")
{
    {
        tmp_diskcache_dir tdd;

        // Current size should be zero (new cache).
        REQUIRE(llvm_state::get_diskcache_size() == 0);

        // Clearing just keeps zero.
        llvm_state::clear_diskcache();
        REQUIRE(llvm_state::get_diskcache_size() == 0);
    }
}

TEST_CASE("diskcache independent instances")
{
    tmp_diskcache_dir tdd1;

    // Record the default limit.
    const auto default_limit = llvm_state::get_diskcache_limit();
    REQUIRE(default_limit > 0);

    // Change the limit in tdd1's DB.
    llvm_state::set_diskcache_limit(12345);
    REQUIRE(llvm_state::get_diskcache_limit() == 12345);

    // Create tdc2 while tdc1 is still alive - this changes the path and resets the connection.
    tmp_diskcache_dir tdd2;

    // Fresh DB - limit should be back to default, not 12345.
    REQUIRE(llvm_state::get_diskcache_limit() == default_limit);

    // Size should be zero in the new DB.
    REQUIRE(llvm_state::get_diskcache_size() == 0);
}
