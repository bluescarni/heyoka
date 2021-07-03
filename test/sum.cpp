// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <heyoka/config.hpp>

// #include <cmath>
// #include <cstdint>
// #include <sstream>
// #include <variant>
// #include <vector>

// #include <fmt/format.h>
// #include <fmt/ranges.h>

// #if defined(HEYOKA_HAVE_REAL128)

// #include <mp++/real128.hpp>

// #endif

#include <heyoka/expression.hpp>
// #include <heyoka/func.hpp>
// #include <heyoka/llvm_state.hpp>
// #include <heyoka/math/cos.hpp>
// #include <heyoka/math/kepE.hpp>
// #include <heyoka/math/pow.hpp>
// #include <heyoka/math/sin.hpp>
// #include <heyoka/math/sqrt.hpp>
// #include <heyoka/number.hpp>
// #include <heyoka/taylor.hpp>
#include <heyoka/math/sum.hpp>

#include "catch.hpp"
// #include "test_utils.hpp"

// #if defined(_MSC_VER) && !defined(__clang__)

// // NOTE: MSVC has issues with the other "using"
// // statement form.
// using namespace fmt::literals;

// #else

// using fmt::literals::operator""_format;

// #endif

using namespace heyoka;
// using namespace heyoka_test;

TEST_CASE("sum ctor")
{
    sum({1_dbl, 2_dbl, 3_dbl});
}
