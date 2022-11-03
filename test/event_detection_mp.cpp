// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>

#include <fmt/format.h>

#include <mp++/real.hpp>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("poly translator 1")
{
    using fp_t = mppp::real;

    auto poly_eval5 = [](const auto &a, const auto &x) {
        return ((((a[5] * x + a[4]) * x + a[3]) * x + a[2]) * x + a[1]) * x + a[0];
    };

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30u, 123u}) {
            std::vector<fp_t> output, input;
            output.resize(6u, fp_t{0, prec});
            input.resize(6u, fp_t{0, prec});

            for (auto i = 0u; i < 6u; ++i) {
                mppp::set(input[i], i + 1u);
            }

            llvm_state s{kw::opt_level = opt_level};

            detail::add_poly_translator_1(s, detail::llvm_type_like(s, input[0]), 5, 1);

            s.optimise();

            s.compile();

            auto *pt1 = reinterpret_cast<void (*)(fp_t *, const fp_t *)>(s.jit_lookup("poly_translate_1"));

            pt1(output.data(), input.data());

            REQUIRE(poly_eval5(output, fp_t{"1.1", prec})
                    == approximately(poly_eval5(input, fp_t{"1.1", prec} + fp_t{1, prec})));
        }
    }
}

TEST_CASE("poly csc")
{
    using fp_t = mppp::real;

    for (auto opt_level : {0u, 3u}) {
        for (auto prec : {30u, 123u}) {
            std::vector<fp_t> input;
            input.resize(6u, fp_t{0, prec});

            for (auto i = 0u; i < 6u; ++i) {
                mppp::set(input[i], i + 1u);
            }

            llvm_state s{kw::opt_level = opt_level};

            detail::llvm_add_csc(s, detail::llvm_type_like(s, input[0]), 5, 1);

            s.optimise();

            s.compile();

            auto *pt1 = reinterpret_cast<void (*)(std::uint32_t *, const fp_t *)>(s.jit_lookup(
                fmt::format("heyoka_csc_degree_5_{}", detail::llvm_mangle_type(detail::llvm_type_like(s, input[0])))));

            std::uint32_t out = 1;
            pt1(&out, input.data());

            REQUIRE(out == 0u);

            mppp::set(input[0], -1);

            pt1(&out, input.data());

            REQUIRE(out == 1u);

            mppp::set(input[1], -2);
            mppp::set(input[3], -1);

            pt1(&out, input.data());

            REQUIRE(out == 3u);
        }
    }
}
