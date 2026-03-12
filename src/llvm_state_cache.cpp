// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/scope/scope_exit.hpp>
#include <boost/unordered_map.hpp>

#include <fmt/core.h>

#include <llvm/Config/llvm-config.h>

#include <sqlite3.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/llvm_state.hpp>

// The in-memory cache maps the bitcode of one or more LLVM modules and an integer flag (representing several
// compilation settings) to:
//
// - the optimised version of the bitcode,
// - the textual IR corresponding to the optimised bitcode,
// - the object code of the optimised bitcode.
//
// The cache invalidation policy is LRU, implemented by pairing a linked list to an unordered_map.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to compute the total size in bytes of the data contained in an llvm_mc_value. Will throw on overflow.
std::size_t llvm_mc_value::total_size() const
{
    assert(!opt_bc.empty());
    assert(opt_bc.size() == opt_ir.size());
    assert(opt_bc.size() == obj.size());

    boost::safe_numerics::safe<std::size_t> ret = 0;

    for (decltype(opt_bc.size()) i = 0; i < opt_bc.size(); ++i) {
        ret += opt_bc[i].size();
        ret += opt_ir[i].size();
        ret += obj[i].size();
    }

    return ret;
}

namespace
{

// Default sizes for the caches (in bytes).
//
// NOTE: 2GB.
constexpr std::uint64_t default_memcache_size = 2147483648ull;
// NOTE: make this 10 times bigger, needs to be represented as signed int due to the fact that this ends up in an sqlite
// table.
constexpr std::int64_t default_diskcache_size = default_memcache_size * 10;

// Global mutex for thread-safe operations on both the in-memory and on-disk caches.
//
// NOTE: std::mutex constructor not constexpr on MinGW:
//
// https://github.com/bluescarni/heyoka/issues/403
#if !defined(__MINGW32__)
HEYOKA_CONSTINIT
#endif
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex llvm_state_cache_mutex;

// Definition of the data structures for the in-memory cache.
using lru_queue_t = std::list<std::pair<std::vector<std::string>, std::uint32_t>>;

using lru_key_t = lru_queue_t::iterator;

// Implementation of hashing for std::pair<std::vector<std::string>, std::uint32_t> and its heterogeneous counterpart.
template <typename T>
auto cache_key_hasher(const T &k) noexcept
{
    assert(!k.first.empty());

    // Combine the bitcodes.
    auto seed = std::hash<std::string>{}(k.first[0]);
    for (decltype(k.first.size()) i = 1; i < k.first.size(); ++i) {
        boost::hash_combine(seed, k.first[i]);
    }

    // Combine with the compilation flag.
    boost::hash_combine(seed, k.second);

    return seed;
}

struct lru_hasher {
    std::size_t operator()(const lru_key_t &k) const noexcept
    {
        return cache_key_hasher(*k);
    }
};

struct lru_cmp {
    bool operator()(const lru_key_t &k1, const lru_key_t &k2) const noexcept
    {
        return *k1 == *k2;
    }
};

// NOTE: use boost::unordered_map because we need heterogeneous lookup.
using lru_map_t = boost::unordered_map<lru_key_t, llvm_mc_value, lru_hasher, lru_cmp>;

// Global variables for the implementation of the in-memory cache.
//
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
lru_queue_t lru_queue;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp,bugprone-throwing-static-initialization)
lru_map_t lru_map;

// Size of the in-memory cache.
//
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::size_t memcache_size = 0;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::uint64_t memcache_limit = default_memcache_size;

// Machinery for heterogeneous lookup into the in-memory cache.
//
// NOTE: this function MUST be invoked while holding the global lock.
auto llvm_state_memcache_hl(const std::vector<std::string> &bc, const std::uint32_t comp_flag)
{
    // NOTE: the heterogeneous version of the key replaces std::vector<std::string> with a const reference.
    using compat_key_t = std::pair<const std::vector<std::string> &, std::uint32_t>;

    struct compat_hasher {
        std::size_t operator()(const compat_key_t &k) const noexcept
        {
            return cache_key_hasher(k);
        }
    };

    struct compat_cmp {
        bool operator()(const lru_key_t &k1, const compat_key_t &k2) const noexcept
        {
            return k1->first == k2.first && k1->second == k2.second;
        }
        bool operator()(const compat_key_t &k1, const lru_key_t &k2) const noexcept
        {
            return operator()(k2, k1);
        }
    };

    return lru_map.find(std::make_pair(std::cref(bc), comp_flag), compat_hasher{}, compat_cmp{});
}

// Small helper to safely compute the total size (in bytes) of a vector of bitcodes.
auto llvm_state_memcache_bc_size(const std::vector<std::string> &bc)
{
    boost::safe_numerics::safe<std::size_t> size = 0;

    for (const auto &s : bc) {
        size += s.size();
    }

    return size;
}

// Debug function to run sanity checks on the in-memory cache.
//
// NOTE: this function MUST be invoked while holding the global lock.
void llvm_state_memcache_sanity_checks()
{
    assert(lru_queue.size() == lru_map.size());

#if !defined(NDEBUG)

    // Check that the computed size of the cache is consistent with memcache_size.
    boost::safe_numerics::safe<std::size_t> size = 0;
    for (const auto &[it, val] : std::as_const(lru_map)) {
        size += llvm_state_memcache_bc_size(it->first);
        size += val.total_size();
    }

    assert(size == memcache_size);

#endif
}

// Implementation of the insertion of an entry into the in-memory cache.
//
// NOTE: this function MUST be invoked while holding the global lock. It assumes that the entry is *not* present in the
// cache already.
void llvm_state_memcache_insert_impl(std::vector<std::string> bc, const std::uint32_t comp_flag, llvm_mc_value val)
{
    // Compute the new cache size (i.e., the size after insertion of the new entry).
    auto new_cache_size
        = boost::safe_numerics::safe<std::size_t>(memcache_size) + val.total_size() + llvm_state_memcache_bc_size(bc);

    // Remove items from the cache if we are exceeding the limit.
    while (new_cache_size > memcache_limit && !lru_queue.empty()) {
        // Fetch an iterator to the last item in the queue.
        const auto last_queue_it = std::prev(lru_queue.end());

        // Locate it in lru_map.
        const auto cur_it = lru_map.find(last_queue_it);
        assert(cur_it != lru_map.end());

        // Compute the total size of the last item in the queue.
        //
        // NOTE: cur_size is computed with safe arithmetics.
        const auto cur_size = cur_it->second.total_size() + llvm_state_memcache_bc_size(last_queue_it->first);

        // NOTE: the next 4 lines cannot throw, which ensures that the cache cannot be left in an inconsistent state.

        // Remove the last item in the queue.
        lru_map.erase(cur_it);
        lru_queue.pop_back();

        // Update new_cache_size and memcache_size.
        new_cache_size -= cur_size;
        memcache_size -= cur_size;
    }

    if (new_cache_size > memcache_limit) {
        // We cleared out the cache and yet insertion of bc would still exceed the limit. Exit.
        assert(lru_queue.empty());
        assert(memcache_size == 0u);

        return;
    }

    // Add the new item to the front of the queue.
    //
    // NOTE: if this throws, we have not modified lru_map yet, no cleanup needed.
    lru_queue.emplace_front(std::move(bc), comp_flag);

    // Add the new item to the map.
    try {
        const auto [new_it, ins_flag] = lru_map.emplace(lru_queue.begin(), std::move(val));
        assert(ins_flag);

        // Update memcache_size.
        memcache_size = new_cache_size;

        // LCOV_EXCL_START
    } catch (...) {
        // Emplacement in lru_map failed, make sure to remove the item we just added to lru_queue before re-throwing.
        lru_queue.pop_front();

        throw;
    }
    // LCOV_EXCL_STOP
}

// Environment fingerprint for the on-disk cache.
//
// This captures properties of the execution environment that affect the bitcode->object code transformation, but are
// NOT reflected in the bitcode itself. Things that affect bitcode *generation* should not need to be here - they
// produce different bitcode, which is already part of the cache key.
//
// For instance, if we are on a heyoka version *without* mppp::real support, we cannot possibly generate bitcode that
// references MPFR functions, so even if such bitcode exists somewhere in the cache originating from a heyoka version
// which does include mppp::real support, we should never get to the point of loading such binary code because we could
// never match the bitcode.
//
// Current ingredients:
//
// - HEYOKA_VERSION: heyoka controls the LLVM optimisation pipeline (which passes, in what order),
//   so a new heyoka version can produce different object code from the same bitcode.
// - LLVM_VERSION: LLVM performs the actual compilation, so even identical bitcode + identical pass
//   pipeline can produce different object code with a different LLVM version.
//
// The format is "KEY1=VALUE1;KEY2=VALUE2;..." to allow extension with additional ingredients in the future.
//
// NOTE: if we ever realise that we need additional fingerprinting (e.g., presence/absence of optional dependencies,
// and/or their specific version numbers), we can easily add new ingredients.
constexpr auto diskcache_fingerprint
    = std::string_view("HEYOKA_VERSION=" HEYOKA_VERSION_STRING ";LLVM_VERSION=" LLVM_VERSION_STRING);

// Version of the DB schema for the on-disk cache.
constexpr int diskcache_schema_version = 0;

// NOLINTBEGIN(concurrency-mt-unsafe)

// Helper to determine the default on-disk cache dir.
std::filesystem::path get_default_diskcache_dir()
{
#ifdef _WIN32
    // NOTE: in principle for max reliability we could use something like:
    //
    // SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &path_tmp)
    //
    // instead of getenv(). Keep in mind if this ever becomes an issue.
    if (const auto *p = std::getenv("LOCALAPPDATA")) {
        if (*p != '\0') {
            return std::filesystem::path(p) / "heyoka" / "cache";
        }
    }
#elif defined(__APPLE__)
    if (const auto *p = std::getenv("HOME")) {
        if (*p != '\0') {
            return std::filesystem::path(p) / "Library" / "Caches" / "heyoka";
        }
    }
#else
    if (const auto *p = std::getenv("XDG_CACHE_HOME")) {
        if (*p != '\0') {
            return std::filesystem::path(p) / "heyoka";
        }
    }
    if (const auto *p = std::getenv("HOME")) {
        if (*p != '\0') {
            return std::filesystem::path(p) / ".cache" / "heyoka";
        }
    }
#endif

    // Empty fallback.
    return {};
}

// Helper to determine the on-disk cache dir on startup. It will use the HEYOKA_CACHE_DIR env variable if provided,
// otherwise it will fall back to get_default_diskcache_dir().
std::filesystem::path get_initial_diskcache_dir()
{
    if (const auto *p = std::getenv("HEYOKA_CACHE_DIR")) {
        if (*p != '\0') {
            return {p};
        }
    }

    return get_default_diskcache_dir();
}

// NOLINTEND(concurrency-mt-unsafe)

// Small wrapper to safely call sqlite3_errmsg(). It will always return a non-null pointer.
[[nodiscard]] const char *sqlite3_errmsg_non_null(sqlite3 *db)
{
    constexpr auto def_msg = "Unknown error";

    if (db == nullptr) {
        return def_msg;
    }

    const auto *ret = sqlite3_errmsg(db);
    return (ret == nullptr) ? def_msg : ret;
}

// Struct encapsulating the state necessary to manage the on-disk cache.
struct diskcache_state {
    // RAII wrapper for a SQLite database connection.
    struct diskcache_connection {
        // Custom deleters for SQLite resources.
        struct db_closer {
            void operator()(sqlite3 *db) const noexcept
            {
                sqlite3_close_v2(db);
            }
        };
        struct stmt_finalizer {
            void operator()(sqlite3_stmt *s) const noexcept
            {
                sqlite3_finalize(s);
            }
        };
        struct errmsg_freer {
            void operator()(char *p) const noexcept
            {
                sqlite3_free(p);
            }
        };

        using db_ptr = std::unique_ptr<sqlite3, db_closer>;
        using stmt_ptr = std::unique_ptr<sqlite3_stmt, stmt_finalizer>;
        using errmsg_ptr = std::unique_ptr<char, errmsg_freer>;

        // NOTE: m_db must be declared before the prepared statements so that it is destroyed after them.
        db_ptr m_db;
        stmt_ptr m_lookup_stmt;
        stmt_ptr m_insert_stmt;
        stmt_ptr m_evict_stmt;

        // Common open flags for sqlite3_open_v2():
        //
        // - READWRITE: open for both reading and writing.
        // - CREATE: create the database if it does not exist.
        // - NOMUTEX: we protect all access with our own std::mutex, no need for SQLite's internal locking.
        // - EXRESCODE: return extended result codes (e.g., SQLITE_CONSTRAINT_UNIQUE instead of SQLITE_CONSTRAINT).
        static constexpr int open_flags
            = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_EXRESCODE;

        // The constructor opens the db connection.
        //
        // NOTE: eventually in the future we should also handle schema changes in this constructor, probably by dropping
        // the existing tables and reconstructing them. For now we do not bother.
        explicit diskcache_connection(const std::filesystem::path &dir)
        {
            if (dir.empty()) [[unlikely]] {
                throw std::runtime_error(
                    "Unable to open the llvm_state on-disk cache: the path to the database dir is empty");
            }

            // Create the cache directory if it does not exist.
            std::filesystem::create_directories(dir);

            // Open the database.
            const auto db_path = (std::filesystem::canonical(dir) / "cache.db").string();
            open_db(db_path);

            // Configure the connection.
            configure_db(db_path);

            // Integrity check: if the database is corrupt, throw. The user will need to manually delete the corrupt
            // cache directory in order to restore disk caching.
            if (!integrity_check()) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("The llvm_state on-disk cache database '{}' is corrupt. Please delete "
                                "the cache directory to restore disk caching.",
                                db_path));
            }

            // Create the tables.

            // First, the "cache" table.
            exec("CREATE TABLE IF NOT EXISTS cache ("
                 // The hash value of the cache entry, computed by hashing together the original bitcode, the
                 // compilation flags and the environment fingerprint. This is the primary key, hence unique.
                 "hash INTEGER PRIMARY KEY, "
                 // The original (unoptimised) bitcode.
                 "bitcode BLOB NOT NULL, "
                 // The compilation flags.
                 "comp_flag INTEGER NOT NULL, "
                 // The environment fingerprint.
                 "env_fingerprint TEXT NOT NULL, "
                 // The optimised bitcode.
                 "opt_bc BLOB NOT NULL, "
                 // The optimised IR.
                 "opt_ir BLOB NOT NULL, "
                 // The optimised object code.
                 "obj BLOB NOT NULL, "
                 // Approximate total size (in bytes) of the data stored in the entry.
                 "size INTEGER NOT NULL, "
                 // An integer signalling when the entry was last looked up.
                 "last_access INTEGER NOT NULL)");
            exec("CREATE INDEX IF NOT EXISTS idx_last_access ON cache(last_access)");

            // Config table: persistent key-value store for cache bookkeeping. This contains 3 rows:
            //
            // - "schema_version": version of the table layout, for future upgrade paths.
            // - "total_size": sum of all entries' data sizes.
            // - "size_limit": maximum total size of all entries' data.
            exec("CREATE TABLE IF NOT EXISTS config ("
                 "key TEXT PRIMARY KEY, "
                 "value INTEGER NOT NULL)");
            exec(fmt::format("INSERT OR IGNORE INTO config (key, value) VALUES ('schema_version', {})",
                             diskcache_schema_version)
                     .c_str());
            exec("INSERT OR IGNORE INTO config (key, value) VALUES ('total_size', 0)");
            exec(fmt::format("INSERT OR IGNORE INTO config (key, value) VALUES ('size_limit', {})",
                             default_diskcache_size)
                     .c_str());

            // Prepare the cached statements.

            // Lookup statement: atomically finds an entry by hash, bumps its last_access counter, and returns the full
            // entry. The new last_access is derived inline as MAX(last_access) + 1, eliminating the need for a separate
            // persistent counter.
            //
            // NOTE: if the hash matches but the bitcode/comp_flag/env_fingerprint don't, we have a collision, and we
            // have uselessly bumped the entry's last_access. This is a harmless pessimisation - it will just delay the
            // eventual eviction of the entry, and hopefully it should be a rare occurrence. The upshot is that we can
            // perform all the work in a single atomic statement without resorting to transactions.
            //
            // NOTE: MAX(last_access) should be efficient (O(log(n)) instead of O(n)) because last_access has an INDEX.
            m_lookup_stmt = prepare("WITH new_lru AS (SELECT COALESCE(MAX(last_access), 0) + 1 AS val FROM cache) "
                                    "UPDATE cache SET last_access = (SELECT val FROM new_lru) WHERE hash = ? RETURNING "
                                    "bitcode, comp_flag, env_fingerprint, opt_bc, opt_ir, obj");
        }

        // Helper to execute a simple SQL statement with no parameters or results.
        void exec(const char *sql) const
        {
            assert(m_db);

            char *err_msg = nullptr;
            if (sqlite3_exec(m_db.get(), sql, nullptr, nullptr, &err_msg) != SQLITE_OK) [[unlikely]] {
                const errmsg_ptr msg_guard(err_msg);
                throw std::runtime_error(fmt::format("SQLite exec failed for '{}': {}", sql,
                                                     (err_msg == nullptr) ? "unknown error" : err_msg));
            }
        }

        // Helper to prepare a persistent statement.
        stmt_ptr prepare(const char *sql) const
        {
            assert(m_db);

            sqlite3_stmt *stmt_raw = nullptr;
            // NOTE: SQLITE_PREPARE_PERSISTENT hints that these statements are long-lived, preventing SQLite from using
            // lookaside memory (a limited per-connection fast allocator) for them.
            if (sqlite3_prepare_v3(m_db.get(), sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt_raw, nullptr) != SQLITE_OK)
                [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("SQLite prepare failed for '{}': {}", sql, sqlite3_errmsg_non_null(m_db.get())));
            }
            return stmt_ptr(stmt_raw);
        }

    private:
        // Helper to open the database.
        void open_db(const std::string &db_path)
        {
            sqlite3 *db_raw = nullptr;
            // NOTE: even on failure, sqlite3_open_v2() may allocate a handle that needs closing.
            if (sqlite3_open_v2(db_path.c_str(), &db_raw, open_flags, nullptr) != SQLITE_OK) [[unlikely]] {
                // NOTE: this takes care of cleanup.
                const db_ptr tmp(db_raw);
                throw std::runtime_error(
                    fmt::format("Failed to open the llvm_state on-disk cache database at the path '{}': {}", db_path,
                                sqlite3_errmsg_non_null(db_raw)));
            }
            m_db.reset(db_raw);
        }

        // Helper to configure a freshly-opened database connection.
        void configure_db(const std::string &db_path) const
        {
            assert(m_db);

            // Enable defensive mode: prevents SQL from deliberately corrupting the database.
            if (sqlite3_db_config(m_db.get(), SQLITE_DBCONFIG_DEFENSIVE, 1, nullptr) != SQLITE_OK) [[unlikely]] {
                throw std::runtime_error(fmt::format("Unable to set the SQLITE_DBCONFIG_DEFENSIVE option for the "
                                                     "llvm_state on-disk cache database at the path '{}'",
                                                     db_path));
            }

            // Set a busy timeout for cross-process WAL contention.
            //
            // NOTE: this comes into play only if the database is being used from multiple processes. E.g., if a process
            // is inserting a large compilation artifact, the other process will have to wait until this operation is
            // finished. In case of a timeout, we will throw and catch the exception in the outer layer and treat it as
            // a cache miss.
            if (sqlite3_busy_timeout(m_db.get(), 5000) != SQLITE_OK) [[unlikely]] {
                throw std::runtime_error(fmt::format("Unable to set the busy timeout for the "
                                                     "llvm_state on-disk cache database at the path '{}'",
                                                     db_path));
            }

            // Enable WAL mode and verify it was actually activated.
            //
            // NOTE: the 'PRAGMA journal_mode=WAL' statement can fail silently (e.g., on network filesystems that lack
            // shared memory support). It returns the *current* journal mode, which may still be "delete" if WAL could
            // not be enabled. Therefore, we will be checking the return value of the command.
            //
            // NOTE: we have to manually issue the statement here because it has a return value (otherwise we could just
            // use exec()).
            {
                sqlite3_stmt *stmt_raw = nullptr;
                if (sqlite3_prepare_v3(m_db.get(), "PRAGMA journal_mode = WAL", -1, 0, &stmt_raw, nullptr) != SQLITE_OK)
                    [[unlikely]] {
                    throw std::runtime_error(fmt::format("Failed to prepare the WAL pragma for the llvm_state on-disk "
                                                         "cache database at the path '{}': {}",
                                                         db_path, sqlite3_errmsg_non_null(m_db.get())));
                }
                const stmt_ptr stmt(stmt_raw);
                if (sqlite3_step(stmt.get()) != SQLITE_ROW) [[unlikely]] {
                    throw std::runtime_error(fmt::format(
                        "Failed to set the WAL journal mode for the llvm_state on-disk cache database at the path '{}'",
                        db_path));
                }

                // NOTE: this is the extra step we perform - we check the return value of the statement.
                const auto *const mode = reinterpret_cast<const char *>(sqlite3_column_text(stmt.get(), 0));
                if (mode == nullptr || std::string_view(mode) != "wal") [[unlikely]] {
                    throw std::runtime_error(fmt::format("Failed to enable the WAL journal mode for the llvm_state "
                                                         "on-disk cache database at the path '{}' (got '{}')",
                                                         db_path, mode == nullptr ? "null" : mode));
                }
            }

            // NOTE: this reduces the aggressiveness of disk flushes in order to increase performance. This is an
            // acceptable tradeoff for a cache.
            exec("PRAGMA synchronous = NORMAL");
        }

        // Helper to run PRAGMA quick_check. Returns true if the database is healthy.
        [[nodiscard]] bool integrity_check() const
        {
            assert(m_db);

            sqlite3_stmt *stmt_raw = nullptr;
            // NOTE: quick_check is faster than integrity_check (skips index verification).
            if (sqlite3_prepare_v3(m_db.get(), "PRAGMA quick_check", -1, 0, &stmt_raw, nullptr) != SQLITE_OK)
                [[unlikely]] {
                return false;
            }
            const stmt_ptr stmt(stmt_raw);

            if (sqlite3_step(stmt.get()) != SQLITE_ROW) [[unlikely]] {
                return false;
            }

            // quick_check returns "ok" as the first row if the database is healthy.
            const auto *const result = reinterpret_cast<const char *>(sqlite3_column_text(stmt.get(), 0));
            return result != nullptr && std::string_view(result) == "ok";
        }
    };

    std::filesystem::path dir = get_initial_diskcache_dir();
    bool enabled = false;
    std::optional<diskcache_connection> conn;
    // Scratch buffers used for serialisation of vector<string> into blobs.
    //
    // Four buffers are needed because on insert, all four blobs (bitcode, opt_bc, opt_ir, obj) must be alive
    // simultaneously when bound with SQLITE_STATIC.
    std::array<std::vector<char>, 4> s11n_bufs;

    [[nodiscard]] std::optional<llvm_mc_value> lookup(const std::vector<std::string> &bc, const std::uint32_t comp_flag)
    {
        if (!enabled) {
            // Disk cache disabled, nothing to do.
            return {};
        }

        try {
            // Establish the database connection, if needed.
            if (!conn) {
                conn.emplace(dir);
            }

            // Compute the hash.
            const auto hash = compute_hash(bc, comp_flag);

            // Fetch the statement.
            auto *const stmt = conn->m_lookup_stmt.get();

            // NOTE: the scope guard ensures sqlite3_reset() is called on all exit paths, releasing any write lock held
            // after SQLITE_ROW and resetting the statement for the next lookup invocation.
            const boost::scope::scope_exit reset_guard([stmt]() noexcept { sqlite3_reset(stmt); });

            // Bind the statement.
            if (sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(hash)) != SQLITE_OK) [[unlikely]] {
                throw std::runtime_error(fmt::format("Failed to bind hash in the llvm_state on-disk cache lookup: {}",
                                                     sqlite3_errmsg_non_null(conn->m_db.get())));
            }

            // Start execution.
            const auto rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                // No row with this hash - cache miss.
                //
                // NOTE: in this codepath, a single step is enough.
                return {};
            }
            if (rc != SQLITE_ROW) [[unlikely]] {
                throw std::runtime_error(fmt::format("Failed to execute lookup in the llvm_state on-disk cache: {}",
                                                     sqlite3_errmsg_non_null(conn->m_db.get())));
            }

            // Helper to complete the statement by stepping to SQLITE_DONE. This is necessary because the first step
            // returned SQLITE_ROW but the statement has not completed yet - the second step commits the UPDATE
            // (the last_access bump) and releases the write lock.
            const auto finish = [stmt, this]() {
                if (sqlite3_step(stmt) != SQLITE_DONE) [[unlikely]] {
                    throw std::runtime_error(
                        fmt::format("Failed to complete the lookup statement in the llvm_state on-disk cache: {}",
                                    sqlite3_errmsg_non_null(conn->m_db.get())));
                }
            };

            // Row found. Read columns for key verification.
            //
            // RETURNING columns: 0=bitcode, 1=comp_flag, 2=env_fingerprint, 3=opt_bc, 4=opt_ir, 5=obj.
            //
            // NOTE: this is a raw pointer to the bc blob data.
            const auto *const bc_blob = sqlite3_column_blob(stmt, 0);
            // NOTE: this is the total size in bytes of the raw blob data - it *must* be called after
            // sqlite3_column_blob().
            const auto bc_blob_len = sqlite3_column_bytes(stmt, 0);
            const auto db_comp_flag = static_cast<std::uint32_t>(sqlite3_column_int64(stmt, 1));
            const auto *const db_fp = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));

            // Key verification: check that the full key matches, not just the hash.
            assert(bc_blob != nullptr);
            assert(db_fp != nullptr);
            if (db_comp_flag != comp_flag || std::string_view(db_fp) != diskcache_fingerprint
                || !vec_str_matches_blob(bc_blob, bc_blob_len, bc)) {
                // Hashes match but the full key does not. Cache miss.
                finish();
                return {};
            }

            // Key matches. Deserialize the cached value.
            //
            // NOTE: sqlite3_column_bytes() must be called *after* sqlite3_column_blob() on the same column.
            const auto *const opt_bc_blob = sqlite3_column_blob(stmt, 3);
            const auto opt_bc_blob_len = sqlite3_column_bytes(stmt, 3);
            const auto *const opt_ir_blob = sqlite3_column_blob(stmt, 4);
            const auto opt_ir_blob_len = sqlite3_column_bytes(stmt, 4);
            const auto *const obj_blob = sqlite3_column_blob(stmt, 5);
            const auto obj_blob_len = sqlite3_column_bytes(stmt, 5);

            llvm_mc_value ret;
            ret.opt_bc = deserialize_vec_str(opt_bc_blob, opt_bc_blob_len);
            ret.opt_ir = deserialize_vec_str(opt_ir_blob, opt_ir_blob_len);
            ret.obj = deserialize_vec_str(obj_blob, obj_blob_len);

            // Complete the statement: commits the LRU bump and releases the write lock.
            finish();

            return ret;

            // LCOV_EXCL_START
        } catch (const std::exception &ex) {
            get_logger()->warn("exception thrown while attempting a lookup in the llvm_state on-disk cache: {}",
                               ex.what());
        } catch (...) {
            get_logger()->warn(
                "exception thrown while attempting a lookup in the llvm_state on-disk cache: unknown error message");
        }

        return {};
        // LCOV_EXCL_STOP
    }

    [[nodiscard]] std::int64_t get_limit()
    {
        // Establish the database connection, if needed.
        if (!conn) {
            conn.emplace(dir);
        }

        const auto stmt = conn->prepare("SELECT value FROM config WHERE key = 'size_limit'");
        if (sqlite3_step(stmt.get()) != SQLITE_ROW) [[unlikely]] {
            throw std::runtime_error("Failed to read size_limit from the llvm_state on-disk cache config");
        }

        return sqlite3_column_int64(stmt.get(), 0);
    }

    void set_limit(const std::int64_t limit)
    {
        if (limit < 0) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid negative size limit for the llvm_state on-disk cache: {}", limit));
        }

        // Establish the database connection, if needed.
        if (!conn) {
            conn.emplace(dir);
        }

        conn->exec(fmt::format("UPDATE config SET value = {} WHERE key = 'size_limit'", limit).c_str());
    }

    [[nodiscard]] std::int64_t get_size()
    {
        // Establish the database connection, if needed.
        if (!conn) {
            conn.emplace(dir);
        }

        const auto stmt = conn->prepare("SELECT value FROM config WHERE key = 'total_size'");
        if (sqlite3_step(stmt.get()) != SQLITE_ROW) [[unlikely]] {
            throw std::runtime_error("Failed to read total_size from the llvm_state on-disk cache config");
        }

        return sqlite3_column_int64(stmt.get(), 0);
    }

    void clear()
    {
        // Establish the database connection, if needed.
        if (!conn) {
            conn.emplace(dir);
        }

        conn->exec("BEGIN IMMEDIATE");
        try {
            conn->exec("DELETE FROM cache");
            conn->exec("UPDATE config SET value = 0 WHERE key = 'total_size'");
            conn->exec("COMMIT");
        } catch (...) {
            // NOTE: if ROLLBACK fails, destroy the connection to ensure cleanup. sqlite3_close_v2() always succeeds
            // (returns SQLITE_OK regardless) and automatically rolls back any open transaction. The connection will be
            // lazily re-opened as needed.
            if (sqlite3_exec(conn->m_db.get(), "ROLLBACK", nullptr, nullptr, nullptr) != SQLITE_OK) {
                conn.reset();
            }

            throw;
        }
    }

private:
    // Helper to compute the hash for a cache lookup.
    //
    // NOTE: here we are using Boost's hashing primitives on the hope/assumption that they are deterministic across
    // processes and program executions. The determinism is desirable for optimal performance, otherwise entries
    // inserted by one process would never be found by another, making the disk cache useless. Boost's hashing does not
    // use random salting (unlike, e.g., Python's default string hashing), so this should hold in practice. In case this
    // changes in the future, we can always switch to other deterministic hashing approaches.
    [[nodiscard]] static std::size_t compute_hash(const std::vector<std::string> &bc, const std::uint32_t comp_flag)
    {
        // Start with the bitcodes.
        std::size_t seed = 0;
        for (const auto &cur_bc : bc) {
            boost::hash_combine(seed, cur_bc);
        }

        // Mix in the compiler flags.
        boost::hash_combine(seed, comp_flag);

        // Mix in the environment fingerprint.
        boost::hash_combine(seed, diskcache_fingerprint);

        return seed;
    }

    // Serialisation helpers to convert between vector<string> and sqlite blob.
    //
    // In the cache table, we are storing bc, ir and object code as blobs (i.e., char buffers), but in C++ they are
    // vector<string>. We thus want to be able to convert between the two representations. A vector<string> is encoded
    // as a char array in the following way:
    //
    // [uint64_t len_0][bytes_0][uint64_t len_1][bytes_1]...
    //
    // I.e., no element count prefix - the number of elements is implicit from reading until the end of the blob.
    //
    // NOTE: endianness is not a concern because the cache is local to the machine.

    // Serialize a vector<string> into buf. Returns a pointer+size pair suitable for sqlite3_bind_blob().
    [[nodiscard]] static std::pair<const void *, int> serialize_vec_str(std::vector<char> &buf,
                                                                        const std::vector<std::string> &vs)
    {
        buf.clear();

        for (const auto &s : vs) {
            const auto len = boost::numeric_cast<std::uint64_t>(s.size());
            const auto *const len_ptr = reinterpret_cast<const char *>(&len);
            buf.insert(buf.end(), len_ptr, len_ptr + sizeof(len));
            buf.insert(buf.end(), s.begin(), s.end());
        }

        return {buf.data(), boost::numeric_cast<int>(buf.size())};
    }

    // Check if a serialized blob matches a vector<string>, comparing element-by-element without allocating.
    //
    // NOTE: here we are assuming that the data in blob was created by serialize_vec_str().
    [[nodiscard]] static bool vec_str_matches_blob(const void *const blob, const int blob_len,
                                                   const std::vector<std::string> &bc)
    {
        assert(blob != nullptr);
        assert(blob_len >= 0);

        // Convert to char for byte iteration.
        const auto *blob_char = static_cast<const char *>(blob);

        // Init the remaining number of bytes to be read from blob.
        auto blob_rem = static_cast<unsigned>(blob_len);

        for (const auto &s : bc) {
            if (blob_rem == 0u) {
                // NOTE: we have finished reading from the blob but not from bc - they cannot be equal.
                return false;
            }

            // Extract the size of the current blob chunk.
            assert(blob_rem >= sizeof(std::uint64_t));
            std::uint64_t cur_blob_len = 0;
            std::ranges::copy(blob_char, blob_char + sizeof(std::uint64_t), reinterpret_cast<char *>(&cur_blob_len));

            // We have read the value of cur_blob_len, update blob_char and blob_rem.
            blob_char += sizeof(std::uint64_t);
            blob_rem -= sizeof(std::uint64_t);

            // The current blob chunk must fit in the remaining blob length.
            assert(cur_blob_len <= blob_rem);

            if (!std::ranges::equal(s, std::ranges::subrange(blob_char, blob_char + cur_blob_len))) {
                // NOTE: the sizes or the contents do not match between s and the current blob chunk - blob and bc must
                // differ.
                return false;
            }

            // We have read the content of the current blob chunk, update blob_char and blob_rem.
            blob_char += cur_blob_len;
            blob_rem -= static_cast<unsigned>(cur_blob_len);
        }

        // If the blob is fully consumed, it's an exact match.
        return blob_rem == 0u;
    }

    // Deserialise a blob into a vector<string>.
    [[nodiscard]] static std::vector<std::string> deserialize_vec_str(const void *const blob, const int blob_len)
    {
        assert(blob != nullptr);
        assert(blob_len >= 0);

        std::vector<std::string> out;

        // Convert to char for byte iteration.
        const auto *blob_char = static_cast<const char *>(blob);

        // Init the remaining number of bytes to be read from blob.
        auto blob_rem = static_cast<unsigned>(blob_len);

        while (blob_rem > 0u) {
            // Extract the size of the current blob chunk.
            assert(blob_rem >= sizeof(std::uint64_t));
            std::uint64_t cur_blob_len = 0;
            std::ranges::copy(blob_char, blob_char + sizeof(std::uint64_t), reinterpret_cast<char *>(&cur_blob_len));

            // We have read the value of cur_blob_len, update blob_char and blob_rem.
            blob_char += sizeof(std::uint64_t);
            blob_rem -= sizeof(std::uint64_t);

            // The current blob chunk must fit in the remaining blob length.
            assert(cur_blob_len <= blob_rem);

            // Append the current blob chunk into out.
            out.emplace_back(blob_char, blob_char + cur_blob_len);

            // We have read the content of the current blob chunk, update blob_char and blob_rem.
            blob_char += cur_blob_len;
            blob_rem -= static_cast<unsigned>(cur_blob_len);
        }

        return out;
    }
};

// Global instance of diskcache_state.
//
// NOTE: *all* accesses to this instance need to be protected via a lock on llvm_state_cache_mutex.
//
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp,bugprone-throwing-static-initialization)
diskcache_state diskcache_st;

} // namespace

std::optional<llvm_mc_value> llvm_state_memcache_lookup(const std::vector<std::string> &bc,
                                                        const std::uint32_t comp_flag)
{
    // Lock down.
    const std::scoped_lock lock(llvm_state_cache_mutex);

    // Sanity checks.
    llvm_state_memcache_sanity_checks();

    if (const auto it = llvm_state_memcache_hl(bc, comp_flag); it == lru_map.end()) {
        // Cache miss. Try the disk cache.
        auto diskcache_value = diskcache_st.lookup(bc, comp_flag);
        if (diskcache_value) {
            // With a disk cache hit, we need to insert the hit into the in-memory cache before returning.
            llvm_state_memcache_insert_impl(bc, comp_flag, *diskcache_value);
        }

        return diskcache_value;
    } else {
        // Cache hit.

        // Move the item to the front of the queue, if needed.
        if (const auto queue_it = it->first; queue_it != lru_queue.begin()) {
            // NOTE: splice() won't throw.
            lru_queue.splice(lru_queue.begin(), lru_queue, queue_it, std::next(queue_it));
        }

        return it->second;
    }
}

void llvm_state_memcache_try_insert(std::vector<std::string> bc, const std::uint32_t comp_flag, llvm_mc_value val)
{
    // Lock down.
    const std::scoped_lock lock(llvm_state_cache_mutex);

    // Sanity checks.
    llvm_state_memcache_sanity_checks();

    // Do a first lookup to check if bc is already in the cache. This could happen, e.g., if two threads are compiling
    // the same code concurrently.
    if (const auto it = llvm_state_memcache_hl(bc, comp_flag); it != lru_map.end()) {
        assert(val.opt_bc == it->second.opt_bc);
        assert(val.opt_ir == it->second.opt_ir);
        assert(val.obj == it->second.obj);

        return;
    }

    // Do the insertion.
    llvm_state_memcache_insert_impl(std::move(bc), comp_flag, std::move(val));
}

} // namespace detail

std::size_t llvm_state::get_memcache_size()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return detail::memcache_size;
}

std::size_t llvm_state::get_memcache_limit()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return boost::numeric_cast<std::size_t>(detail::memcache_limit);
}

void llvm_state::set_memcache_limit(std::size_t new_limit)
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    detail::memcache_limit = boost::numeric_cast<std::uint64_t>(new_limit);
}

void llvm_state::clear_memcache()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    // Sanity checks.
    detail::llvm_state_memcache_sanity_checks();

    detail::lru_map.clear();
    detail::lru_queue.clear();
    detail::memcache_size = 0;
}

std::filesystem::path llvm_state::get_diskcache_path()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return detail::diskcache_st.dir;
}

void llvm_state::set_diskcache_path(std::filesystem::path path)
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    // NOTE: if the user changes path after the db connection has been established, reset the connection. It will be
    // lazily re-established with the updated path the next time we need to look into the on-disk cache.
    //
    // NOTE: the reset and move() are all noexcept, no risk of leaving diskcache_st in an intermediate state.
    if (detail::diskcache_st.conn) {
        detail::diskcache_st.conn.reset();
    }

    detail::diskcache_st.dir = std::move(path);
}

bool llvm_state::get_diskcache_enabled()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return detail::diskcache_st.enabled;
}

void llvm_state::set_diskcache_enabled(const bool flag)
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    detail::diskcache_st.enabled = flag;
}

std::int64_t llvm_state::get_diskcache_limit()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return detail::diskcache_st.get_limit();
}

void llvm_state::set_diskcache_limit(const std::int64_t limit)
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    detail::diskcache_st.set_limit(limit);
}

std::int64_t llvm_state::get_diskcache_size()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    return detail::diskcache_st.get_size();
}

void llvm_state::clear_diskcache()
{
    // Lock down.
    const std::scoped_lock lock(detail::llvm_state_cache_mutex);

    detail::diskcache_st.clear();
}

HEYOKA_END_NAMESPACE
