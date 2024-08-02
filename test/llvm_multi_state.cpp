// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    llvm_multi_state ms({llvm_state{}, llvm_state{}, llvm_state{}, llvm_state{}, llvm_state{}, llvm_state{}});
    ms.compile();
}
