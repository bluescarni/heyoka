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

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/kepE.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/taylor.hpp>

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

                // Find the IR containing the SIMD codegen and verify that sincos fusion happened:
                // at least one call to the sincos array wrapper must be present, and no separate
                // llvm.sin/llvm.cos calls should remain.
                std::string ir;

                if (compact_mode) {
                    // In compact mode, get_ir() returns a vector of strings (one per module).
                    // Find the SIMD batch driver module.
                    const auto irs = std::get<1>(cf.get_llvm_states()).get_ir();
                    for (const auto &mod_ir : irs) {
                        if (mod_ir.find("@heyoka.Sleef_sincos") != std::string::npos
                            && mod_ir.find("batch_size_1(") == std::string::npos) {
                            ir = mod_ir;
                            break;
                        }
                    }
                    REQUIRE(!ir.empty());
                } else {
                    ir = std::get<0>(cf.get_llvm_states())[2].get_ir();
                }

                // Sincos fusion is happening.
                REQUIRE(ir.find("@heyoka.Sleef_sincos") != std::string::npos);
                // No separate sin/cos calls remain.
                REQUIRE(ir.find("@llvm.sin.") == std::string::npos);
                REQUIRE(ir.find("@llvm.cos.") == std::string::npos);
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

#if defined(HEYOKA_HAVE_REAL128)

// Test that combined_sin/combined_cos on the same argument are fused into a single sincosq wrapper call via CSE for
// real128.
TEST_CASE("sincos fusion real128")
{
    auto x = make_vars("x");

    cfunc<mppp::real128> cf({combined_sin(x), combined_cos(x)}, {x});

    // Check fusion in the IR.
    const auto ir = std::get<0>(cf.get_llvm_states())[0].get_ir();

    using string_find_iterator = boost::find_iterator<std::string::const_iterator>;
    auto call_count = 0u;
    for (auto it = boost::make_find_iterator(std::as_const(ir), boost::first_finder("@heyoka.sincosq_wrapper"));
         it != string_find_iterator(); ++it) {
        ++call_count;
    }

    // We expect 2 occurrences: 1 call + 1 definition.
    REQUIRE(call_count == 2u);

    // Check correctness of the produced values.
    using namespace mppp::literals;

    const std::vector<mppp::real128> ins = {1.23_rq};
    std::vector<mppp::real128> outs(2u);

    cf(outs, ins);

    REQUIRE(outs[0] == approximately(sin(1.23_rq)));
    REQUIRE(outs[1] == approximately(cos(1.23_rq)));
}

#endif

#if defined(HEYOKA_HAVE_REAL)

// Test that combined_sin/combined_cos on the same argument are fused into a single mpfr sincos wrapper call via CSE for
// mppp::real.
TEST_CASE("sincos fusion real")
{
    const auto prec = 237u;

    auto x = make_vars("x");

    cfunc<mppp::real> cf({combined_sin(x), combined_cos(x)}, {x}, kw::prec = prec, kw::compact_mode = false);

    // Check fusion in the IR.
    const auto ir = std::get<0>(cf.get_llvm_states())[0].get_ir();

    const auto search_str = fmt::format("@heyoka.real.{}.sincos", prec);

    using string_find_iterator = boost::find_iterator<std::string::const_iterator>;
    auto call_count = 0u;
    for (auto it = boost::make_find_iterator(ir, boost::first_finder(search_str)); it != string_find_iterator(); ++it) {
        ++call_count;
    }

    // We expect 2 occurrences: 1 call + 1 definition.
    REQUIRE(call_count == 2u);

    // Check correctness of the produced values.
    const std::vector ins{mppp::real{"1.23", prec}};
    std::vector<mppp::real> outs(2u, mppp::real{0, prec});

    cf(outs, ins);

    REQUIRE(outs[0] == approximately(sin(mppp::real{"1.23", prec})));
    REQUIRE(outs[1] == approximately(cos(mppp::real{"1.23", prec})));
}

#endif

#if defined(HEYOKA_WITH_SLEEF)

// Test that kepE's internal sincos calls are fused via the combined primitives
// in the explicitly vectorized codepath.
TEST_CASE("kepE sincos fusion")
{
    const auto &tf = detail::get_target_features();

    if (tf.sse2 || tf.aarch64) {
        tuple_for_each(fp_types, [](auto fp_x) {
            using fp_t = decltype(fp_x);

            auto [ecc, M] = make_vars("ecc", "M");

            cfunc<fp_t> cf({kepE(ecc, M)}, {ecc, M});

            // Non-compact mode: check the SIMD llvm_state (third state).
            const auto ir = std::get<0>(cf.get_llvm_states())[2].get_ir();

            // Sincos fusion is happening.
            REQUIRE(ir.find("@heyoka.Sleef_sincos") != std::string::npos);
            // No separate sin/cos calls remain.
            REQUIRE(ir.find("@llvm.sin.") == std::string::npos);
            REQUIRE(ir.find("@llvm.cos.") == std::string::npos);
        });
    }
}

#endif

#if defined(__linux__)

// Test that kepE's scalar codegen fuses sin+cos into sincos via the backend.
TEST_CASE("kepE scalar sincos backend fusion")
{
    auto [ecc, M] = make_vars("ecc", "M");

    tuple_for_each(fp_types, [&](auto fp_x) {
        using fp_t = decltype(fp_x);

        cfunc<fp_t> cf({kepE(ecc, M)}, {ecc, M});

        const auto &obj0 = std::get<0>(cf.get_llvm_states())[0].get_object_code();
        REQUIRE(obj0.find("sincos") != std::string::npos);
        const auto &obj1 = std::get<0>(cf.get_llvm_states())[1].get_object_code();
        REQUIRE(obj1.find("sincos") != std::string::npos);
    });
}

#endif

#if defined(HEYOKA_HAVE_REAL128)

// Test that kepE's sincos calls are fused via CSE for real128.
TEST_CASE("kepE sincos fusion real128")
{
    auto [ecc, M] = make_vars("ecc", "M");

    cfunc<mppp::real128> cf({kepE(ecc, M)}, {ecc, M});

    const auto ir = std::get<0>(cf.get_llvm_states())[0].get_ir();

    using string_find_iterator = boost::find_iterator<std::string::const_iterator>;
    auto call_count = 0u;
    for (auto it = boost::make_find_iterator(std::as_const(ir), boost::first_finder("@heyoka.sincosq_wrapper"));
         it != string_find_iterator(); ++it) {
        ++call_count;
    }

    // With CSE, each sincos on the same argument is fused: 3 calls (initial guess + first Newton step + loop body) + 1
    // definition.
    REQUIRE(call_count == 4u);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

// Test that kepE's sincos calls are fused via CSE for mppp::real.
TEST_CASE("kepE sincos fusion real")
{
    const auto prec = 237u;

    auto [ecc, M] = make_vars("ecc", "M");

    cfunc<mppp::real> cf({kepE(ecc, M)}, {ecc, M}, kw::prec = prec, kw::compact_mode = false);

    const auto ir = std::get<0>(cf.get_llvm_states())[0].get_ir();

    const auto search_str = fmt::format("@heyoka.real.{}.sincos", prec);

    using string_find_iterator = boost::find_iterator<std::string::const_iterator>;
    auto call_count = 0u;
    for (auto it = boost::make_find_iterator(std::as_const(ir), boost::first_finder(search_str));
         it != string_find_iterator(); ++it) {
        ++call_count;
    }

    // With CSE, each sincos on the same argument is fused: 3 calls (initial guess + first Newton step + loop body) + 1
    // definition.
    REQUIRE(call_count == 4u);
}

#endif

// Test that the sincos combine pass correctly modifies cfunc and Taylor decompositions.
TEST_CASE("sincos combine pass")
{
    auto [x, y] = make_vars("x", "y");

    // cfunc decomposition: sin(x), cos(x), cos(x), sin(y).
    //
    // sin(x) and cos(x) share argument x -> both should be combined. The second cos(x) should be CSE'd away (same as
    // first cos(x)). sin(y) has no matching cos(y) -> should remain non-combined.
    {
        cfunc<double> cf({sin(x), cos(x), cos(x), sin(y)}, {x, y});

        const auto &dc = cf.get_dc();

        auto n_combined_sin = 0, n_combined_cos = 0, n_uncombined_sin = 0;

        for (const auto &ex : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    } else {
                        ++n_uncombined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        REQUIRE(n_combined_sin == 1);
        REQUIRE(n_combined_cos == 1);
        REQUIRE(n_uncombined_sin == 1);
    }

    // Do also a test with par.
    {
        cfunc<double> cf({sin(par[0]), cos(par[0]), cos(par[0]), sin(par[1])}, {x, y});

        const auto &dc = cf.get_dc();

        auto n_combined_sin = 0, n_combined_cos = 0, n_uncombined_sin = 0;

        for (const auto &ex : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    } else {
                        ++n_uncombined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        REQUIRE(n_combined_sin == 1);
        REQUIRE(n_combined_cos == 1);
        REQUIRE(n_uncombined_sin == 1);
    }

    // And one with numbers.
    {
        const auto s3 = expression{func{detail::sin_impl{3_dbl, false}}};
        const auto c3 = expression{func{detail::cos_impl{3_dbl, false}}};
        const auto s2 = expression{func{detail::sin_impl{2_dbl, false}}};

        cfunc<double> cf({s3, c3, c3, s2}, {x, y});

        const auto &dc = cf.get_dc();

        auto n_combined_sin = 0, n_combined_cos = 0, n_uncombined_sin = 0;

        for (const auto &ex : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    } else {
                        ++n_uncombined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        REQUIRE(n_combined_sin == 1);
        REQUIRE(n_combined_cos == 1);
        REQUIRE(n_uncombined_sin == 1);
    }

    // Taylor decomposition: x' = sin(x).
    //
    // Taylor decomposition injects cos(x) as a hidden dependency, so the combine pass should find the pair and combine
    // both.
    {
        auto ta = taylor_adaptive<double>{{prime(x) = sin(x)}, {0.1}, kw::tol = 1.};

        const auto &dc = ta.get_decomposition();

        auto n_combined_sin = 0, n_combined_cos = 0;

        for (const auto &[ex, deps] : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        REQUIRE(n_combined_sin == 1);
        REQUIRE(n_combined_cos == 1);
    }

    // Do also a test with par.
    {
        auto ta = taylor_adaptive<double>{{prime(x) = sin(par[0])}, {0.1}, kw::tol = 1.};

        const auto &dc = ta.get_decomposition();

        auto n_combined_sin = 0, n_combined_cos = 0;

        for (const auto &[ex, deps] : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        REQUIRE(n_combined_sin == 1);
        REQUIRE(n_combined_cos == 1);
    }

    // And one with numbers.
    {
        const auto s3 = expression{func{detail::sin_impl{3_dbl, false}}};
        auto ta = taylor_adaptive<double>{{prime(x) = s3}, {0.1}, kw::tol = 1.};

        const auto &dc = ta.get_decomposition();

        auto n_combined_sin = 0, n_combined_cos = 0;

        for (const auto &[ex, deps] : dc) {
            if (const auto *f = std::get_if<func>(&ex.value())) {
                if (const auto *si = f->extract<detail::sin_impl>()) {
                    if (si->is_combined()) {
                        ++n_combined_sin;
                    }
                }
                if (const auto *ci = f->extract<detail::cos_impl>()) {
                    if (ci->is_combined()) {
                        ++n_combined_cos;
                    }
                }
            }
        }

        // NOTE: here the checks are 0 because the creation of the Taylor decomposition triggers constant folding on the
        // hidden dep, so the sin/cos pair is not recognised as acting on the same number.
        REQUIRE(n_combined_sin == 0);
        REQUIRE(n_combined_cos == 0);
    }
}

#if defined(HEYOKA_WITH_SLEEF)

// Test that combined sin/cos are individually auto-vectorized via SLP. The combined scalar wrappers inline, exposing
// llvm.sin/llvm.cos calls, which SLP vectorizes separately with regular SLEEF. The combined sincos fusion does NOT
// happen in the SLP path, but individual vectorization should still work.
TEST_CASE("sincos SLP vectorization")
{
    const auto &tf = detail::get_target_features();

    if (tf.sse2 || tf.aarch64) {
        auto [a, b, c, d] = make_vars("a", "b", "c", "d");

        // 4 combined sin + 4 combined cos on different arguments. SLP should bundle the sin calls together and the cos
        // calls together, vectorizing each group with regular SLEEF.
        cfunc<double> cf({combined_sin(a), combined_sin(b), combined_sin(c), combined_sin(d), combined_cos(a),
                          combined_cos(b), combined_cos(c), combined_cos(d)},
                         {a, b, c, d}, kw::slp_vectorize = true);

        // Verify correctness.
        const std::vector ins{1., 2., 3., 4.};
        std::vector<double> outs(8u, 0.);

        cf(outs, ins);

        REQUIRE(outs[0] == approximately(std::sin(1.)));
        REQUIRE(outs[1] == approximately(std::sin(2.)));
        REQUIRE(outs[2] == approximately(std::sin(3.)));
        REQUIRE(outs[3] == approximately(std::sin(4.)));
        REQUIRE(outs[4] == approximately(std::cos(1.)));
        REQUIRE(outs[5] == approximately(std::cos(2.)));
        REQUIRE(outs[6] == approximately(std::cos(3.)));
        REQUIRE(outs[7] == approximately(std::cos(4.)));

        // NOTE: this test currently fails on the CI on OSX arm64 with llvm 19 - the SLP vectorisation does not happen.
        // It is not clear at this time if this is an issue specific to the platform or to the llvm version or to a
        // combination of the two. Let us revisit this in the future.
#if !defined(__APPLE__)
        // Verify SLP vectorization happened: the scalar llvm.sin/llvm.cos intrinsics should be completely absent from
        // the unstrided scalar module.
        const auto ir = std::get<0>(cf.get_llvm_states())[0].get_ir();

        REQUIRE(ir.find("@llvm.sin.f64") == std::string::npos);
        REQUIRE(ir.find("@llvm.cos.f64") == std::string::npos);
#endif
    }
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

#if defined(HEYOKA_HAVE_REAL128)

// Test combined sincos with real128 in a batch Taylor integrator,
// exercising the real128 vector codepath.
TEST_CASE("sincos real128 batch")
{
    using fp_t = mppp::real128;

    auto [x, y] = make_vars("x", "y");

    auto ta = taylor_adaptive_batch<fp_t>{
        {prime(x) = sin(y), prime(y) = cos(x)}, {fp_t{2}, fp_t{-1}, fp_t{3}, fp_t{-4}}, 2, kw::tol = .5};

    ta.step(true);

    const auto jet = tc_to_jet(ta);

    REQUIRE(jet[0] == 2);
    REQUIRE(jet[1] == -1);
    REQUIRE(jet[2] == 3);
    REQUIRE(jet[3] == -4);

    REQUIRE(jet[4] == approximately(sin(jet[2])));
    REQUIRE(jet[5] == approximately(sin(jet[3])));
    REQUIRE(jet[6] == approximately(cos(jet[0])));
    REQUIRE(jet[7] == approximately(cos(jet[1])));
}

#endif
