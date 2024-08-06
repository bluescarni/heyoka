// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <vector>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/sgp4.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("basic")
{
    auto [x, y] = make_vars("x", "y");

    llvm_state tplt;

    auto [ms, dc] = detail::make_multi_cfunc<double>(tplt, "test", {x + y}, {x, y}, 1, false, false, 0);

    ms.compile();
}

TEST_CASE("sgp4")
{
    detail::edb_disabler ed;

    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");

    auto dt = diff_tensors(model::sgp4(), std::vector(inputs.begin(), inputs.end()));

    // auto cf = cfunc<double>(dt.get_jacobian(), inputs, kw::compact_mode = true);

    // return;

    llvm_state tplt;

    auto [ms, dc] = detail::make_multi_cfunc<double>(tplt, "test", dt.get_jacobian(),
                                                     std::vector(inputs.begin(), inputs.end()), 4, false, false, 0);

    ms.compile();
}
