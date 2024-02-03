// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <llvm/Config/llvm-config.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

constexpr bool skip_batch_ld =
#if LLVM_VERSION_MAJOR <= 17
    std::numeric_limits<long double>::digits == 64
#else
    false
#endif
    ;

// Basic API test.
TEST_CASE("basic")
{
    auto [x, y] = make_vars("x", "y");

    auto tester = [&](auto fp_x, unsigned opt_level, bool high_accuracy, bool compact_mode) {
        using fp_t = decltype(fp_x);

        const std::uint32_t custom_batch_size = (skip_batch_ld && std::same_as<fp_t, long double>) ? 1 : 2;

        const std::array<fp_t, 2> inputs = {1, 2};
        std::array<fp_t, 2> outputs{};

        // A few tests for def-cted object.
        cfunc<fp_t> cf0;

        REQUIRE_THROWS_AS(cf0.get_vars(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_fn(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_dc(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_llvm_state_scalar(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_llvm_state_scalar_s(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_llvm_state_batch_s(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_high_accuracy(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_compact_mode(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_parallel_mode(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf0.get_batch_size(), std::invalid_argument);

        auto cf1 = cf0;

        REQUIRE_THROWS_AS(cf1.get_vars(), std::invalid_argument);

        auto cf2 = std::move(cf1);

        REQUIRE_THROWS_AS(cf1.get_vars(), std::invalid_argument);
        REQUIRE_THROWS_AS(cf2.get_vars(), std::invalid_argument);

        REQUIRE_THROWS_AS(cf2({}, {}), std::invalid_argument);

        // Main constructor.
        cf0 = cfunc<fp_t>{{x + y, x - y}, {y, x}, kw::parallel_mode = true};

        REQUIRE(cf0.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf0.get_vars() == std::vector{y, x});
        REQUIRE(!cf0.get_dc().empty());
        REQUIRE(cf0.get_llvm_state_scalar().get_opt_level() == 3u);
        REQUIRE(cf0.get_llvm_state_scalar_s().get_opt_level() == 3u);
        REQUIRE(cf0.get_llvm_state_batch_s().get_opt_level() == 3u);
        REQUIRE(cf0.get_high_accuracy() == false);
        REQUIRE(cf0.get_compact_mode() == false);
        REQUIRE(cf0.get_parallel_mode() == true);
        REQUIRE(cf0.get_batch_size() == recommended_simd_size<fp_t>());

        cf0(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));

        cf0 = cfunc<fp_t>{{x + y, x - y},
                          {y, x},
                          kw::batch_size = custom_batch_size,
                          kw::opt_level = opt_level,
                          kw::high_accuracy = high_accuracy,
                          kw::compact_mode = compact_mode};

        REQUIRE(cf0.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf0.get_vars() == std::vector{y, x});
        REQUIRE(!cf0.get_dc().empty());
        REQUIRE(cf0.get_llvm_state_scalar().get_opt_level() == opt_level);
        REQUIRE(cf0.get_llvm_state_scalar_s().get_opt_level() == opt_level);
        REQUIRE(cf0.get_llvm_state_batch_s().get_opt_level() == opt_level);
        REQUIRE(cf0.get_high_accuracy() == high_accuracy);
        REQUIRE(cf0.get_compact_mode() == compact_mode);
        REQUIRE(cf0.get_parallel_mode() == false);
        REQUIRE(cf0.get_batch_size() == custom_batch_size);

        cf0(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));

        // Copy constructor.
        auto cf0_copy = cf0;

        REQUIRE(cf0_copy.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf0_copy.get_vars() == std::vector{y, x});
        REQUIRE(!cf0_copy.get_dc().empty());
        REQUIRE(cf0_copy.get_llvm_state_scalar().get_opt_level() == opt_level);
        REQUIRE(cf0_copy.get_llvm_state_scalar_s().get_opt_level() == opt_level);
        REQUIRE(cf0_copy.get_llvm_state_batch_s().get_opt_level() == opt_level);
        REQUIRE(cf0_copy.get_high_accuracy() == high_accuracy);
        REQUIRE(cf0_copy.get_compact_mode() == compact_mode);
        REQUIRE(cf0_copy.get_parallel_mode() == false);
        REQUIRE(cf0_copy.get_batch_size() == custom_batch_size);

        cf0_copy(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));

        // Move ctor.
        auto cf0_move = std::move(cf0_copy);

        REQUIRE_THROWS_AS(cf0_copy.get_vars(), std::invalid_argument);

        REQUIRE(cf0_move.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf0_move.get_vars() == std::vector{y, x});
        REQUIRE(!cf0_move.get_dc().empty());
        REQUIRE(cf0_move.get_llvm_state_scalar().get_opt_level() == opt_level);
        REQUIRE(cf0_move.get_llvm_state_scalar_s().get_opt_level() == opt_level);
        REQUIRE(cf0_move.get_llvm_state_batch_s().get_opt_level() == opt_level);
        REQUIRE(cf0_move.get_high_accuracy() == high_accuracy);
        REQUIRE(cf0_move.get_compact_mode() == compact_mode);
        REQUIRE(cf0_move.get_parallel_mode() == false);
        REQUIRE(cf0_move.get_batch_size() == custom_batch_size);

        cf0_move(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));

        // Copy assignment.
        cf1 = cf0;

        REQUIRE(cf1.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf1.get_vars() == std::vector{y, x});
        REQUIRE(!cf1.get_dc().empty());
        REQUIRE(cf1.get_llvm_state_scalar().get_opt_level() == opt_level);
        REQUIRE(cf1.get_llvm_state_scalar_s().get_opt_level() == opt_level);
        REQUIRE(cf1.get_llvm_state_batch_s().get_opt_level() == opt_level);
        REQUIRE(cf1.get_high_accuracy() == high_accuracy);
        REQUIRE(cf1.get_compact_mode() == compact_mode);
        REQUIRE(cf1.get_parallel_mode() == false);
        REQUIRE(cf1.get_batch_size() == custom_batch_size);

        cf1(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));

        // Move assignment.
        cf2 = std::move(cf1);

        REQUIRE_THROWS_AS(cf1.get_vars(), std::invalid_argument);

        REQUIRE(cf2.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf2.get_vars() == std::vector{y, x});
        REQUIRE(!cf2.get_dc().empty());
        REQUIRE(cf2.get_llvm_state_scalar().get_opt_level() == opt_level);
        REQUIRE(cf2.get_llvm_state_scalar_s().get_opt_level() == opt_level);
        REQUIRE(cf2.get_llvm_state_batch_s().get_opt_level() == opt_level);
        REQUIRE(cf2.get_high_accuracy() == high_accuracy);
        REQUIRE(cf2.get_compact_mode() == compact_mode);
        REQUIRE(cf2.get_parallel_mode() == false);
        REQUIRE(cf2.get_batch_size() == custom_batch_size);

        cf2(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
        std::ranges::fill(outputs, fp_t(0));
    };

    for (auto cm : {false, true}) {
        for (auto f : {false, true}) {
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 0, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 1, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 2, f, cm); });
            tuple_for_each(fp_types, [&tester, f, cm](auto x) { tester(x, 3, f, cm); });
        }
    }
}

TEST_CASE("s11n")
{
    {
        // s11n on def cted object.
        cfunc<double> cf0;

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);
            oa << cf0;
        }

        cf0 = cfunc<double>({1_dbl}, {"x"_var});
        REQUIRE(cf0.get_vars() == std::vector{"x"_var});

        {
            boost::archive::binary_iarchive ia(ss);
            ia >> cf0;
        }

        REQUIRE_THROWS_AS(cf0.get_vars(), std::invalid_argument);
    }

    {
        // s11n on an object with several custom settings.
        const std::array<double, 2> inputs = {1, 2};
        std::array<double, 2> outputs{};

        auto [x, y] = make_vars("x", "y");

        auto cf0 = cfunc<double>{{x + y, x - y},           {y, x},
                                 kw::batch_size = 1,       kw::opt_level = 1,
                                 kw::high_accuracy = true, kw::compact_mode = true};

        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);
            oa << cf0;
        }

        cf0 = cfunc<double>{};
        REQUIRE_THROWS_AS(cf0.get_vars(), std::invalid_argument);

        {
            boost::archive::binary_iarchive ia(ss);
            ia >> cf0;
        }

        REQUIRE(cf0.get_fn() == std::vector{x + y, x - y});
        REQUIRE(cf0.get_vars() == std::vector{y, x});
        REQUIRE(!cf0.get_dc().empty());
        REQUIRE(cf0.get_llvm_state_scalar().get_opt_level() == 1);
        REQUIRE(cf0.get_llvm_state_scalar_s().get_opt_level() == 1);
        REQUIRE(cf0.get_llvm_state_batch_s().get_opt_level() == 1);
        REQUIRE(cf0.get_high_accuracy() == true);
        REQUIRE(cf0.get_compact_mode() == true);
        REQUIRE(cf0.get_parallel_mode() == false);
        REQUIRE(cf0.get_batch_size() == 1);

        cf0(outputs, inputs);
        REQUIRE(outputs[0] == 3);
        REQUIRE(outputs[1] == 1);
    }
}
