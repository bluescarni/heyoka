// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <heyoka/config.hpp>

// #include <algorithm>
// #include <cmath>
// #include <initializer_list>
// #include <limits>
// #include <random>
// #include <sstream>
// #include <tuple>
// #include <type_traits>
// #include <vector>

// #include <boost/algorithm/string/find_iterator.hpp>
// #include <boost/algorithm/string/finder.hpp>
// #include <boost/algorithm/string/predicate.hpp>

// #include <llvm/Config/llvm-config.h>

// #if defined(HEYOKA_HAVE_REAL128)

// #include <mp++/real128.hpp>

// #endif

// #if defined(HEYOKA_HAVE_REAL)

// #include <mp++/real.hpp>

// #endif

#include <heyoka/expression.hpp>
// #include <heyoka/kw.hpp>
// #include <heyoka/llvm_state.hpp>
#include <heyoka/math/dfun.hpp>
// #include <heyoka/math/sin.hpp>
// #include <heyoka/s11n.hpp>

#include "catch.hpp"
// #include "test_utils.hpp"

// static std::mt19937 rng;

using namespace heyoka;
// using namespace heyoka_test;

TEST_CASE("basic")
{
    auto [x, y, z] = make_vars("x", "y", "z");

    std::cout << dfun(x, {y, z}) << '\n';
    std::cout << dfun(x, {y, z}, {{1, 1}}) << '\n';
    std::cout << dfun(x, {y, z}, {{0, 1}, {1, 1}}) << '\n';
}
