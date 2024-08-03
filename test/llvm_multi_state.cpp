// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <stdexcept>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

#include <fmt/ranges.h>
#include <ranges>

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

        ms.compile();

        REQUIRE(ms.is_compiled());

        REQUIRE(ms.get_opt_level() == 1u);
        REQUIRE(ms.fast_math());
        REQUIRE(ms.force_avx512());
        REQUIRE(ms.get_slp_vectorize());
        REQUIRE(ms.get_code_model() == code_model::large);
        REQUIRE(ms.get_n_modules() == 5u);
    }
}

TEST_CASE("copy semantics")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    llvm_state s1{kw::mname = "module_0"}, s2{kw::mname = "module_1"};

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    llvm_multi_state ms{{s1, s2}};

    auto ms_copy = ms;

    REQUIRE(ms_copy.get_bc() == ms.get_bc());
    REQUIRE(ms_copy.get_ir() == ms.get_ir());
    REQUIRE(ms_copy.is_compiled() == ms.is_compiled());
    REQUIRE(ms_copy.fast_math() == ms.fast_math());
    REQUIRE(ms_copy.force_avx512() == ms.force_avx512());
    REQUIRE(ms_copy.get_opt_level() == ms.get_opt_level());
    REQUIRE(ms_copy.get_slp_vectorize() == ms.get_slp_vectorize());
    REQUIRE(ms_copy.get_code_model() == ms.get_code_model());
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
}

TEST_CASE("s11n")
{
    using Catch::Matchers::Message;

    auto [x, y] = make_vars("x", "y");

    llvm_state s1{kw::mname = "module_0"}, s2{kw::mname = "module_1"};

    add_cfunc<double>(s1, "f1", {x * y}, {x, y}, kw::compact_mode = true);
    add_cfunc<double>(s2, "f2", {x / y}, {x, y}, kw::compact_mode = true);

    // Uncompiled.
    llvm_multi_state ms{{s1, s2}};

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
}

TEST_CASE("cfunc")
{
    using Catch::Matchers::Message;

    // Basic test.
    auto [x, y] = make_vars("x", "y");

    llvm_state s1{kw::mname = "module_0"}, s2{kw::mname = "module_1"};

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

    // Check the first few characters of the optimised ir/bc match.
    // Cannot check the entire ir/bc because of the difference in trigger name.
    REQUIRE((s1.get_ir().substr(0, 100) == ms.get_ir()[0].substr(0, 100)
             || s1.get_ir().substr(0, 100) == ms.get_ir()[1].substr(0, 100)
             || s1.get_ir().substr(0, 100) == ms.get_ir()[2].substr(0, 100)));
    REQUIRE((s2.get_ir().substr(0, 100) == ms.get_ir()[0].substr(0, 100)
             || s2.get_ir().substr(0, 100) == ms.get_ir()[1].substr(0, 100)
             || s2.get_ir().substr(0, 100) == ms.get_ir()[2].substr(0, 100)));

    std::cout << "orig:" << std::endl << std::endl;
    fmt::print("{}\n\n\n",
               s1.get_bc().substr(0, 25) | std::views::transform([](auto c) { return static_cast<int>(c); }));

    std::cout << "pos0:" << std::endl << std::endl;
    fmt::print("{}\n\n\n",
               ms.get_bc()[0].substr(0, 25) | std::views::transform([](auto c) { return static_cast<int>(c); }));

    std::cout << "pos1:" << std::endl << std::endl;
    fmt::print("{}\n\n\n",
               ms.get_bc()[1].substr(0, 25) | std::views::transform([](auto c) { return static_cast<int>(c); }));

    std::cout << "pos2:" << std::endl << std::endl;
    fmt::print("{}\n\n\n",
               ms.get_bc()[2].substr(0, 25) | std::views::transform([](auto c) { return static_cast<int>(c); }));

    REQUIRE((s1.get_bc().substr(0, 25) == ms.get_bc()[0].substr(0, 25)
             || s1.get_bc().substr(0, 25) == ms.get_bc()[1].substr(0, 25)
             || s1.get_bc().substr(0, 25) == ms.get_bc()[2].substr(0, 25)));
    REQUIRE((s2.get_bc().substr(0, 25) == ms.get_bc()[0].substr(0, 25)
             || s2.get_bc().substr(0, 25) == ms.get_bc()[1].substr(0, 25)
             || s2.get_bc().substr(0, 25) == ms.get_bc()[2].substr(0, 25)));

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

    llvm_state s1{kw::mname = "module_0"}, s2{kw::mname = "module_1"};

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
