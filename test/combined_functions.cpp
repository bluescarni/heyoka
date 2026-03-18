// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/finder.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/mdspan.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

using heyoka::detail::combined_cos;
using heyoka::detail::combined_sin;

static std::mt19937 rng;

// NOTE: we limit testing to float and double, since SIMD acceleration is available only for these types at this time.
const auto fp_types = std::tuple<float, double>{};

#if defined(HEYOKA_WITH_SLEEF)

TEST_CASE("sincos fusion")
{
    const auto &tf = detail::get_target_features();

    // NOTE: this test verifies that combined_sin(x) and combined_cos(x) on the same argument are fused into a single
    // SLEEF sincos call via CSE in the explicitly vectorized codepath. With either sse2 or aarch64, we know that some
    // vectorization will take place.
    if (tf.sse2 || tf.aarch64) {
        for (auto compact_mode : {false, true}) {
            tuple_for_each(fp_types, [compact_mode](auto fp_x) {
                using fp_t = decltype(fp_x);

                auto x = make_vars("x");

                cfunc<fp_t> cf({combined_sin(x), combined_cos(x)}, {x}, kw::compact_mode = compact_mode);

                using string_find_iterator = boost::find_iterator<std::string::const_iterator>;
                auto call_count = 0u;

                // Find the IR containing the SIMD driver and count sincos wrapper references. In compact mode, get_ir()
                // returns a vector of strings (one per module) - we look for the module containing the SIMD batch
                // driver (has "driver_0" and a batch_size > 1). In non-compact mode, the SIMD IR is the third
                // llvm_state. Scalar driver modules contain dead-code sincos infrastructure kept alive by llvm.used for
                // VFABI/SLP, so we must skip them.
                std::string ir;

                if (compact_mode) {
                    const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                    for (const auto &mod_ir : irs) {
                        if (mod_ir.find("@heyoka.Sleef_sincos") != std::string::npos
                            && mod_ir.find("batch_size_1") == std::string::npos) {
                            ir = mod_ir;
                            break;
                        }
                    }
                    REQUIRE(!ir.empty());
                } else {
                    ir = std::get<0>(cf.get_llvm_states())[2].get_ir();
                }

                for (auto it
                     = boost::make_find_iterator(std::as_const(ir), boost::first_finder("@heyoka.Sleef_sincos"));
                     it != string_find_iterator(); ++it) {
                    ++call_count;
                }

                // We expect 2 occurrences: 1 call + 1 definition.
                REQUIRE(call_count == 2u);
            });
        }
    }
}

#endif

#if defined(__linux__)

// Test that LLVM's backend fuses separate sin+cos calls into a single sincos library call on the scalar path.
TEST_CASE("scalar sincos backend fusion")
{
    auto x = make_vars("x");

    tuple_for_each(fp_types, [&x](auto fp_x) {
        using fp_t = decltype(fp_x);

        cfunc<fp_t> cf({combined_sin(x), combined_cos(x)}, {x});

        // Check the scalar llvm_state's object code for the "sincos" symbol.
        const auto &obj0 = std::get<0>(cf.get_llvm_states())[0].get_object_code();
        REQUIRE(obj0.find("sincos") != std::string::npos);
        const auto &obj1 = std::get<0>(cf.get_llvm_states())[1].get_object_code();
        REQUIRE(obj1.find("sincos") != std::string::npos);
    });
}

#endif

TEST_CASE("sincos correctness")
{
    const auto tester = [](auto fp_x, unsigned opt_level, bool compact_mode) {
        using fp_t = decltype(fp_x);

        auto [x, y] = make_vars("x", "y");

        std::uniform_real_distribution<double> rdist(-10., 10.);

        auto gen = [&rdist]() { return static_cast<fp_t>(rdist(rng)); };

        for (auto batch_size : {1u, 2u, 4u, 5u}) {
            std::vector<fp_t> outs(batch_size * 4u);
            std::vector<fp_t> ins(batch_size * 2u);

            std::ranges::generate(ins, gen);

            // Use both combined_sin and combined_cos on the same argument,
            // and also on different arguments.
            cfunc<fp_t> cf({combined_sin(x), combined_cos(x), combined_sin(y), combined_cos(y)}, {x, y},
                           kw::batch_size = batch_size, kw::compact_mode = compact_mode, kw::opt_level = opt_level);

            cf(mdspan<fp_t, dextents<std::size_t, 2>>(outs.data(), 4u, batch_size),
               mdspan<const fp_t, dextents<std::size_t, 2>>(ins.data(), 2u, batch_size));

            using std::cos;
            using std::sin;
            for (auto i = 0u; i < batch_size; ++i) {
                auto x_val = ins[i];
                auto y_val = ins[i + batch_size];

                REQUIRE(outs[i] == approximately(sin(x_val), fp_t(100)));
                REQUIRE(outs[i + batch_size] == approximately(cos(x_val), fp_t(100)));
                REQUIRE(outs[i + 2u * batch_size] == approximately(sin(y_val), fp_t(100)));
                REQUIRE(outs[i + 3u * batch_size] == approximately(cos(y_val), fp_t(100)));
            }
        }
    };

    for (auto cm : {false, true}) {
        for (auto opt_level : {0u, 1u, 2u, 3u}) {
            tuple_for_each(fp_types, [&tester, opt_level, cm](auto x) { tester(x, opt_level, cm); });
        }
    }
}
