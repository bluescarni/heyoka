// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #include <cmath>
// #include <initializer_list>
// #include <iostream>
// #include <limits>
// #include <stdexcept>
// #include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/elp2000.hpp>
// #include <heyoka/taylor.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace heyoka::model;

TEST_CASE("basic")
{
    llvm_state s;

    auto dc = add_cfunc<double>(s, "func", model::elp2000_cartesian_e2000(kw::thresh = 1e-5));
    s.compile();

    fmt::println("{}", fmt::join(dc, "\n"));

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("func"));

    double out[3]{};
    const double tm = (2469000.5 - 2451545.0) / (365.25 * 100);

    cf_ptr(out, nullptr, nullptr, &tm);

    fmt::println("{}", out);
}
