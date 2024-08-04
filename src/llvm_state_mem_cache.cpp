// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <boost/container_hash/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>
#include <boost/unordered_map.hpp>

#include <heyoka/config.hpp>
#include <heyoka/llvm_state.hpp>

// This in-memory cache maps the bitcode
// of one or more LLVM modules and an integer flag
// (representing several compilation settings) to:
//
// - the optimised version of the bitcode,
// - the textual IR corresponding
//   to the optimised bitcode,
// - the object code of the optimised bitcode.
//
// The cache invalidation policy is LRU, implemented
// by pairing a linked list to an unordered_map.

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper to compute the total size in bytes
// of the data contained in an llvm_mc_value.
// Will throw on overflow.
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

// Global mutex for thread-safe operations.
// NOTE: std::mutex constructor not constexpr on MinGW:
// https://github.com/bluescarni/heyoka/issues/403
#if !defined(__MINGW32__)
HEYOKA_CONSTINIT
#endif
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex mem_cache_mutex;

// Definition of the data structures for the cache.
using lru_queue_t = std::list<std::pair<std::vector<std::string>, unsigned>>;

using lru_key_t = lru_queue_t::iterator;

// Implementation of hashing for std::pair<std::vector<std::string>, unsigned> and
// its heterogeneous counterpart.
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
    boost::hash_combine(seed, static_cast<std::size_t>(k.second));

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

// Global variables for the implementation of the cache.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
lru_queue_t lru_queue;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp)
lru_map_t lru_map;

// Size of the cache.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::size_t mem_cache_size = 0;

// NOTE: default cache size limit is 2GB.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::uint64_t mem_cache_limit = 2147483648ull;

// Machinery for heterogeneous lookup into the cache.
// NOTE: this function MUST be invoked while holding the global lock.
auto llvm_state_mem_cache_hl(const std::vector<std::string> &bc, unsigned comp_flag)
{
    // NOTE: the heterogeneous version of the key replaces std::vector<std::string>
    // with a const reference.
    using compat_key_t = std::pair<const std::vector<std::string> &, unsigned>;

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

// Debug function to run sanity checks on the cache.
// NOTE: this function MUST be invoked while holding the global lock.
void llvm_state_mem_cache_sanity_checks()
{
    assert(lru_queue.size() == lru_map.size());

    // Check that the computed size of the cache is consistent with mem_cache_size.
    assert(std::accumulate(lru_map.begin(), lru_map.end(), boost::safe_numerics::safe<std::size_t>(0),
                           [](const auto &a, const auto &p) { return a + p.second.total_size(); })
           == mem_cache_size);
}

} // namespace

std::optional<llvm_mc_value> llvm_state_mem_cache_lookup(const std::vector<std::string> &bc, unsigned comp_flag)
{
    // Lock down.
    const std::lock_guard lock(mem_cache_mutex);

    // Sanity checks.
    llvm_state_mem_cache_sanity_checks();

    if (const auto it = llvm_state_mem_cache_hl(bc, comp_flag); it == lru_map.end()) {
        // Cache miss.
        return {};
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

void llvm_state_mem_cache_try_insert(std::vector<std::string> bc, unsigned comp_flag, llvm_mc_value val)
{
    // Lock down.
    const std::lock_guard lock(mem_cache_mutex);

    // Sanity checks.
    llvm_state_mem_cache_sanity_checks();

    // Do a first lookup to check if bc is already in the cache.
    // This could happen, e.g., if two threads are compiling the same
    // code concurrently.
    if (const auto it = llvm_state_mem_cache_hl(bc, comp_flag); it != lru_map.end()) {
        assert(val.opt_bc == it->second.opt_bc);
        assert(val.opt_ir == it->second.opt_ir);
        assert(val.obj == it->second.obj);

        return;
    }

    // Compute the new cache size.
    auto new_cache_size = boost::safe_numerics::safe<std::size_t>(mem_cache_size) + val.total_size();

    // Remove items from the cache if we are exceeding
    // the limit.
    while (new_cache_size > mem_cache_limit && !lru_queue.empty()) {
        // Compute the size of the last item in the queue.
        const auto cur_it = lru_map.find(std::prev(lru_queue.end()));
        assert(cur_it != lru_map.end());
        const auto &cur_val = cur_it->second;
        // NOTE: no possibility of overflow here, as cur_size is guaranteed
        // not to be greater than mem_cache_size.
        const auto cur_size = cur_val.total_size();

        // NOTE: the next 4 lines cannot throw, which ensures that the
        // cache cannot be left in an inconsistent state.

        // Remove the last item in the queue.
        lru_map.erase(cur_it);
        lru_queue.pop_back();

        // Update new_cache_size and mem_cache_size.
        new_cache_size -= cur_size;
        mem_cache_size -= cur_size;
    }

    if (new_cache_size > mem_cache_limit) {
        // We cleared out the cache and yet insertion of
        // bc would still exceed the limit. Exit.
        assert(lru_queue.empty());
        assert(mem_cache_size == 0u);

        return;
    }

    // Add the new item to the front of the queue.
    // NOTE: if this throws, we have not modified lru_map yet,
    // no cleanup needed.
    lru_queue.emplace_front(std::move(bc), comp_flag);

    // Add the new item to the map.
    try {
        const auto [new_it, ins_flag] = lru_map.emplace(lru_queue.begin(), std::move(val));
        assert(ins_flag);

        // Update mem_cache_size.
        mem_cache_size = new_cache_size;

        // LCOV_EXCL_START
    } catch (...) {
        // Emplacement in lru_map failed, make sure to remove
        // the item we just added to lru_queue before re-throwing.
        lru_queue.pop_front();

        throw;
    }
    // LCOV_EXCL_STOP
}

} // namespace detail

std::size_t llvm_state::get_memcache_size()
{
    // Lock down.
    const std::lock_guard lock(detail::mem_cache_mutex);

    return detail::mem_cache_size;
}

std::size_t llvm_state::get_memcache_limit()
{
    // Lock down.
    const std::lock_guard lock(detail::mem_cache_mutex);

    return boost::numeric_cast<std::size_t>(detail::mem_cache_limit);
}

void llvm_state::set_memcache_limit(std::size_t new_limit)
{
    // Lock down.
    const std::lock_guard lock(detail::mem_cache_mutex);

    detail::mem_cache_limit = boost::numeric_cast<std::uint64_t>(new_limit);
}

void llvm_state::clear_memcache()
{
    // Lock down.
    const std::lock_guard lock(detail::mem_cache_mutex);

    // Sanity checks.
    detail::llvm_state_mem_cache_sanity_checks();

    detail::lru_map.clear();
    detail::lru_queue.clear();
    detail::mem_cache_size = 0;
}

HEYOKA_END_NAMESPACE
