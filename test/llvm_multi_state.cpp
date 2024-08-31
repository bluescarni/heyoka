// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <list>
#include <ranges>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <llvm/Config/llvm-config.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/erf.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    // Default construction.
    {
        REQUIRE_NOTHROW(llvm_multi_state{});
    }

    // No states in input.
    REQUIRE_THROWS_MATCHES(llvm_multi_state{{}}, std::invalid_argument,
                           Message("At least 1 llvm_state object is needed to construct an llvm_multi_state"));

    // Inconsistent settings.
    REQUIRE_THROWS_MATCHES(
        (llvm_multi_state{{llvm_state{kw::opt_level = 1u}, llvm_state{kw::opt_level = 2u}}}), std::invalid_argument,
        Message("Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state"));

    REQUIRE_THROWS_MATCHES(
        (llvm_multi_state{{llvm_state{kw::fast_math = true}, llvm_state{}}}), std::invalid_argument,
        Message("Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state"));

    REQUIRE_THROWS_MATCHES(
        (llvm_multi_state{{llvm_state{}, llvm_state{kw::force_avx512 = true}}}), std::invalid_argument,
        Message("Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state"));

    REQUIRE_THROWS_MATCHES(
        (llvm_multi_state{{llvm_state{}, llvm_state{}, llvm_state{kw::slp_vectorize = true}}}), std::invalid_argument,
        Message("Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state"));
    REQUIRE_THROWS_MATCHES(
        (llvm_multi_state{{llvm_state{}, llvm_state{kw::code_model = code_model::large}, llvm_state{}}}),
        std::invalid_argument,
        Message("Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state"));

    {
        // Construction from compiled modules.
        llvm_state s;
        s.compile();

        REQUIRE_THROWS_MATCHES(
            (llvm_multi_state{{s, llvm_state{}}}), std::invalid_argument,
            Message("An llvm_multi_state can be constructed only from uncompiled llvm_state objects"));
        REQUIRE_THROWS_MATCHES(
            (llvm_multi_state{{llvm_state{}, s}}), std::invalid_argument,
            Message("An llvm_multi_state can be constructed only from uncompiled llvm_state objects"));
    }

    {
        // Invalid module name.
        llvm_state s{kw::mname = "heyoka.master"};
        REQUIRE_THROWS_MATCHES(
            (llvm_multi_state{{s, llvm_state{}}}), std::invalid_argument,
            Message("An invalid llvm_state was passed to the constructor of an llvm_multi_state: the module name "
                    "'heyoka.master' is reserved for internal use by llvm_multi_state"));
    }

    // Test the property getters.
    {
        llvm_state s{kw::opt_level = 1u, kw::fast_math = true, kw::force_avx512 = true, kw::slp_vectorize = true,
                     kw::code_model = code_model::large};

        llvm_multi_state ms{{s, s, s, s}};

        REQUIRE(ms.get_opt_level() == 1u);
        REQUIRE(ms.fast_math());
        REQUIRE(ms.force_avx512());
        REQUIRE(ms.get_slp_vectorize());
        REQUIRE(ms.get_code_model() == code_model::large);
        REQUIRE(ms.get_n_modules() == 5u);
        REQUIRE(!ms.is_compiled());
        REQUIRE(ms.get_parjit() == detail::default_parjit);

        ms.compile();

        REQUIRE(ms.is_compiled());
        REQUIRE(ms.get_opt_level() == 1u);
        REQUIRE(ms.fast_math());
        REQUIRE(ms.force_avx512());
        REQUIRE(ms.get_slp_vectorize());
        REQUIRE(ms.get_code_model() == code_model::large);
        REQUIRE(ms.get_n_modules() == 5u);

        REQUIRE_THROWS_MATCHES(
            ms.compile(), std::invalid_argument,
            Message("The function 'compile' can be invoked only if the llvm_multi_state has not been compiled yet"));
    }

    // Move construction/assignment.
    {
        llvm_state s{kw::opt_level = 1u, kw::fast_math = true, kw::force_avx512 = true, kw::slp_vectorize = true,
                     kw::code_model = code_model::large};

        llvm_multi_state ms{{s, s, s, s}, false};

        auto ms2 = std::move(ms);

        REQUIRE(ms2.get_opt_level() == 1u);
        REQUIRE(ms2.fast_math());
        REQUIRE(ms2.force_avx512());
        REQUIRE(ms2.get_slp_vectorize());
        REQUIRE(ms2.get_code_model() == code_model::large);
        REQUIRE(ms2.get_n_modules() == 5u);
        REQUIRE(!ms2.is_compiled());
        REQUIRE(!ms2.get_parjit());

        ms2.compile();

        llvm_multi_state ms3;
        ms3 = std::move(ms2);

        REQUIRE(ms3.is_compiled());
        REQUIRE(ms3.get_opt_level() == 1u);
        REQUIRE(ms3.fast_math());
        REQUIRE(ms3.force_avx512());
        REQUIRE(ms3.get_slp_vectorize());
        REQUIRE(ms3.get_code_model() == code_model::large);
        REQUIRE(ms3.get_n_modules() == 5u);
    }
}

TEST_CASE("copy semantics")
{
    using Catch::Matchers::Message;

    // NOTE: in order to properly test this, we have to disable the cache.
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(0);

    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    llvm_multi_state ms{{s1, s2}, false};

    auto ms_copy = ms;

    REQUIRE(ms_copy.get_bc() == ms.get_bc());
    REQUIRE(ms_copy.get_ir() == ms.get_ir());
    REQUIRE(ms_copy.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy.fast_math() == ms.fast_math());
    REQUIRE(ms_copy.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy.get_code_model() == ms.get_code_model());
    REQUIRE(!ms_copy.get_parjit());
    REQUIRE_THROWS_MATCHES(
        ms_copy.get_object_code(), std::invalid_argument,
        Message("The function 'get_object_code' can be invoked only after the llvm_multi_state has been compiled"));
    REQUIRE_THROWS_MATCHES(
        ms_copy.jit_lookup("foo"), std::invalid_argument,
        Message("The function 'jit_lookup' can be invoked only after the llvm_multi_state has been compiled"));

    ms.compile();
    ms_copy.compile();

    REQUIRE(ms_copy.get_bc() == ms.get_bc());
    REQUIRE(ms_copy.get_ir() == ms.get_ir());
    REQUIRE(ms_copy.get_object_code() == ms.get_object_code());
    REQUIRE(ms_copy.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy.fast_math() == ms.fast_math());
    REQUIRE(ms_copy.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy.get_code_model() == ms.get_code_model());
    REQUIRE_NOTHROW(ms_copy.jit_lookup("f1"));
    REQUIRE_NOTHROW(ms_copy.jit_lookup("f2"));

    {
        auto *cf1_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy.jit_lookup("f1"));
        auto *cf2_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy.jit_lookup("f2"));

        REQUIRE_THROWS_MATCHES(ms_copy.jit_lookup("f3"), std::invalid_argument,
                               Message("Could not find the symbol 'f3' in an llvm_multi_state"));

        const double ins[] = {2., 3.};
        double outs[2] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
    }

    auto ms_copy2 = ms;

    REQUIRE(ms_copy2.get_bc() == ms.get_bc());
    REQUIRE(ms_copy2.get_ir() == ms.get_ir());
    REQUIRE(ms_copy2.get_object_code() == ms.get_object_code());
    REQUIRE(ms_copy2.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy2.fast_math() == ms.fast_math());
    REQUIRE(ms_copy2.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy2.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy2.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy2.get_code_model() == ms.get_code_model());
    REQUIRE(!ms_copy2.get_parjit());
    REQUIRE_NOTHROW(ms_copy2.jit_lookup("f1"));
    REQUIRE_NOTHROW(ms_copy2.jit_lookup("f2"));

    {
        auto *cf1_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy2.jit_lookup("f1"));
        auto *cf2_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy2.jit_lookup("f2"));

        const double ins[] = {2., 3.};
        double outs[2] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
    }

    // Test also copy assignment.
    llvm_multi_state ms_copy3;
    ms_copy3 = ms_copy2;

    REQUIRE(ms_copy3.get_bc() == ms.get_bc());
    REQUIRE(ms_copy3.get_ir() == ms.get_ir());
    REQUIRE(ms_copy3.get_object_code() == ms.get_object_code());
    REQUIRE(ms_copy3.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy3.fast_math() == ms.fast_math());
    REQUIRE(ms_copy3.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy3.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy3.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy3.get_code_model() == ms.get_code_model());
    REQUIRE(!ms_copy3.get_parjit());
    REQUIRE_NOTHROW(ms_copy3.jit_lookup("f1"));
    REQUIRE_NOTHROW(ms_copy3.jit_lookup("f2"));

    {
        auto *cf1_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy3.jit_lookup("f1"));
        auto *cf2_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy3.jit_lookup("f2"));

        const double ins[] = {2., 3.};
        double outs[2] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
    }

    // Restore the cache.
    llvm_state::set_memcache_limit(100'000'000ull);
}

TEST_CASE("s11n")
{
    using Catch::Matchers::Message;

    // NOTE: in order to properly test this, we have to disable the cache.
    llvm_state::clear_memcache();
    llvm_state::set_memcache_limit(0);

    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    // Uncompiled.
    llvm_multi_state ms{{s1, s2}, false};

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);
        oa << ms;
    }

    llvm_multi_state ms_copy{{llvm_state{}}};

    {
        boost::archive::binary_iarchive ia(ss);
        ia >> ms_copy;
    }

    REQUIRE(ms_copy.get_bc() == ms.get_bc());
    REQUIRE(ms_copy.get_ir() == ms.get_ir());
    REQUIRE(ms_copy.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy.fast_math() == ms.fast_math());
    REQUIRE(ms_copy.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy.get_code_model() == ms.get_code_model());
    REQUIRE(!ms_copy.get_parjit());
    REQUIRE_THROWS_MATCHES(
        ms_copy.get_object_code(), std::invalid_argument,
        Message("The function 'get_object_code' can be invoked only after the llvm_multi_state has been compiled"));
    REQUIRE_THROWS_MATCHES(
        ms_copy.jit_lookup("foo"), std::invalid_argument,
        Message("The function 'jit_lookup' can be invoked only after the llvm_multi_state has been compiled"));

    // Compiled.
    ms.compile();

    ss.str("");

    {
        boost::archive::binary_oarchive oa(ss);
        oa << ms;
    }

    {
        boost::archive::binary_iarchive ia(ss);
        ia >> ms_copy;
    }

    REQUIRE(ms_copy.get_bc() == ms.get_bc());
    REQUIRE(ms_copy.get_ir() == ms.get_ir());
    REQUIRE(ms_copy.get_object_code() == ms.get_object_code());
    REQUIRE(ms_copy.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy.fast_math() == ms.fast_math());
    REQUIRE(ms_copy.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy.get_code_model() == ms.get_code_model());
    REQUIRE(!ms_copy.get_parjit());
    REQUIRE_NOTHROW(ms_copy.jit_lookup("f1"));
    REQUIRE_NOTHROW(ms_copy.jit_lookup("f2"));

    {
        auto *cf1_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy.jit_lookup("f1"));
        auto *cf2_ptr = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(
            ms_copy.jit_lookup("f2"));

        const double ins[] = {2., 3.};
        double outs[2] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
    }

    // Restore the cache.
    llvm_state::set_memcache_limit(100'000'000ull);
}

// Test about s11n of a default-cted llvm_multi_state.
TEST_CASE("empty s11n")
{
    llvm_multi_state ms;

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);
        oa << ms;
    }

    ms = llvm_multi_state{{llvm_state{}}};

    {
        boost::archive::binary_iarchive ia(ss);
        REQUIRE_NOTHROW(ia >> ms);
    }
}

TEST_CASE("cfunc")
{
    using Catch::Matchers::Message;

    // Basic test.
    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    const auto orig_ir1 = s1.get_ir();
    const auto orig_ir2 = s2.get_ir();

    const auto orig_bc1 = s1.get_bc();
    const auto orig_bc2 = s2.get_bc();

    llvm_multi_state ms{{s1, s2}};

    REQUIRE(ms.get_ir().size() == 3u);
    REQUIRE(ms.get_bc().size() == 3u);
    REQUIRE_THROWS_MATCHES(
        ms.get_object_code(), std::invalid_argument,
        Message("The function 'get_object_code' can be invoked only after the llvm_multi_state has been compiled"));

    REQUIRE(orig_ir1 == ms.get_ir()[0]);
    REQUIRE(orig_ir2 == ms.get_ir()[1]);

    REQUIRE(orig_bc1 == ms.get_bc()[0]);
    REQUIRE(orig_bc2 == ms.get_bc()[1]);

    ms.compile();
    s1.compile();
    s2.compile();

    REQUIRE(ms.get_ir().size() == 3u);
    REQUIRE(ms.get_bc().size() == 3u);

    auto *cf1_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f1"));
    auto *cf2_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f2"));

    const double ins[] = {2., 3.};
    double outs[2] = {};

    cf1_ptr(outs, ins, nullptr, nullptr);
    cf2_ptr(outs + 1, ins, nullptr, nullptr);

    REQUIRE(outs[0] == 6);
    REQUIRE(outs[1] == 2. / 3.);
}

TEST_CASE("stream op")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    const auto orig_ir1 = s1.get_ir();
    const auto orig_ir2 = s2.get_ir();

    const auto orig_bc1 = s1.get_bc();
    const auto orig_bc2 = s2.get_bc();

    llvm_multi_state ms{{s1, s2}};

    std::ostringstream oss;
    oss << ms;

    REQUIRE(!oss.str().empty());
}

// A test to check that, post compilation, snapshots and object files
// are ordered deterministically.
TEST_CASE("post compile ordering")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2, s3, s4;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y});
    add_cfunc<double>(s2, "f2", {x / y}, {x, y});
    add_cfunc<double>(s3, "f3", {x + y}, {x, y});
    add_cfunc<double>(s4, "f4", {x - y}, {x, y});

    llvm_state::clear_memcache();

    llvm_multi_state ms{{s1, s2, s3, s4}};
    ms.compile();

    const auto orig_obj = ms.get_object_code();
    const auto orig_ir = ms.get_ir();
    const auto orig_bc = ms.get_bc();

    for (auto i = 0; i < 20; ++i) {
        llvm_state::clear_memcache();

        llvm_multi_state ms2{{s1, s2, s3, s4}};
        ms2.compile();

        REQUIRE(ms2.get_object_code() == orig_obj);
        REQUIRE(ms2.get_ir() == orig_ir);
        REQUIRE(ms2.get_bc() == orig_bc);
    }
}

TEST_CASE("memcache testing")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state s1, s2, s3, s4;

    add_cfunc<double>(s1, "f1", {x * y}, {x, y});
    add_cfunc<double>(s2, "f2", {x / y}, {x, y});
    add_cfunc<double>(s3, "f3", {x + y}, {x, y});
    add_cfunc<double>(s4, "f4", {x - y}, {x, y});

    llvm_state::clear_memcache();

    llvm_multi_state ms{{s1, s2, s3, s4}};
    ms.compile();

    const auto cur_cache_size = llvm_state::get_memcache_size();

    llvm_multi_state ms2{{s1, s2, s3, s4}};
    ms2.compile();

    REQUIRE(cur_cache_size == llvm_state::get_memcache_size());

    auto *cf1_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f1"));
    auto *cf2_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f2"));
    auto *cf3_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f3"));
    auto *cf4_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f4"));

    const double ins[] = {2., 3.};
    double outs[4] = {};

    cf1_ptr(outs, ins, nullptr, nullptr);
    cf2_ptr(outs + 1, ins, nullptr, nullptr);
    cf3_ptr(outs + 2, ins, nullptr, nullptr);
    cf4_ptr(outs + 3, ins, nullptr, nullptr);

    REQUIRE(outs[0] == 6);
    REQUIRE(outs[1] == 2. / 3.);
    REQUIRE(outs[2] == 5);
    REQUIRE(outs[3] == -1);
}

// Tests to check vectorisation via the vector-function-abi-variant machinery.
TEST_CASE("vfabi double")
{
    for (auto fast_math : {false, true}) {
        llvm_state s1{kw::slp_vectorize = true, kw::fast_math = fast_math};
        llvm_state s2{kw::slp_vectorize = true, kw::fast_math = fast_math};

        auto [a, b] = make_vars("a", "b");

        add_cfunc<double>(s1, "cfunc", {erf(a), erf(b)}, {a, b});
        add_cfunc<double>(s2, "cfuncs", {erf(a), erf(b)}, {a, b}, kw::strided = true);

        llvm_multi_state ms{{s1, s2}};

        ms.compile();

        // NOTE: autovec with external scalar functions seems to work
        // only since LLVM 16.
#if defined(HEYOKA_WITH_SLEEF) && LLVM_VERSION_MAJOR >= 16

        for (auto ir : ms.get_ir()) {
            using string_find_iterator = boost::find_iterator<std::string::iterator>;

            auto count = 0u;
            for (auto it = boost::make_find_iterator(ir, boost::first_finder("@erf", boost::is_iequal()));
                 it != string_find_iterator(); ++it) {
                ++count;
            }

            // NOTE: in the master module or in the "cfunc" module, we don't
            // expect any @erf: the master module contains only the trigger,
            // the "cfunc" module should have vectorised everything and
            // there should be no more references to the scalar @erf.
            if (count == 0u) {
                continue;
            }

            // NOTE: occurrences of the scalar version:
            // - 2 calls in the strided cfunc,
            // - 1 declaration.
            REQUIRE(count == 3u);
        }

#endif
    }
}

// Test for the range constructor.
TEST_CASE("range ctor")
{
    auto [x, y] = make_vars("x", "y");

    {
        std::list<llvm_state> slist;

        slist.emplace_back();
        add_cfunc<double>(slist.back(), "f1", {x * y}, {x, y});

        slist.emplace_back();
        add_cfunc<double>(slist.back(), "f2", {x / y}, {x, y});

        slist.emplace_back();
        add_cfunc<double>(slist.back(), "f3", {x + y}, {x, y});

        slist.emplace_back();
        add_cfunc<double>(slist.back(), "f4", {x - y}, {x, y});

        llvm_state::clear_memcache();

        llvm_multi_state ms{slist | std::views::transform([](auto &s) -> auto && { return std::move(s); })};
        ms.compile();

        auto *cf1_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f1"));
        auto *cf2_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f2"));
        auto *cf3_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f3"));
        auto *cf4_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f4"));

        const double ins[] = {2., 3.};
        double outs[4] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);
        cf3_ptr(outs + 2, ins, nullptr, nullptr);
        cf4_ptr(outs + 3, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
        REQUIRE(outs[2] == 5);
        REQUIRE(outs[3] == -1);
    }

    {
        std::array<llvm_state, 4> slist;

        add_cfunc<double>(slist[0], "f1", {x * y}, {x, y});
        add_cfunc<double>(slist[1], "f2", {x / y}, {x, y});
        add_cfunc<double>(slist[2], "f3", {x + y}, {x, y});
        add_cfunc<double>(slist[3], "f4", {x - y}, {x, y});

        llvm_state::clear_memcache();

        llvm_multi_state ms{slist | std::views::transform([](auto &s) -> auto && { return std::move(s); })};
        ms.compile();

        auto *cf1_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f1"));
        auto *cf2_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f2"));
        auto *cf3_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f3"));
        auto *cf4_ptr
            = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(ms.jit_lookup("f4"));

        const double ins[] = {2., 3.};
        double outs[4] = {};

        cf1_ptr(outs, ins, nullptr, nullptr);
        cf2_ptr(outs + 1, ins, nullptr, nullptr);
        cf3_ptr(outs + 2, ins, nullptr, nullptr);
        cf4_ptr(outs + 3, ins, nullptr, nullptr);

        REQUIRE(outs[0] == 6);
        REQUIRE(outs[1] == 2. / 3.);
        REQUIRE(outs[2] == 5);
        REQUIRE(outs[3] == -1);
    }
}
