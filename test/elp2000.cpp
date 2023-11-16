// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/model/elp2000.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace heyoka::model;

TEST_CASE("basic")
{
    llvm_state s;

    auto dc = add_cfunc<double>(s, "func", model::elp2000_cartesian_e2000(kw::thresh = 1e-5), kw::compact_mode = true);
    s.compile();

    auto *cf_ptr
        = reinterpret_cast<void (*)(double *, const double *, const double *, const double *)>(s.jit_lookup("func"));

    double out[3]{};

    // NOTE: these values have been computed after having checked
    // that the full solution coincides with the values provided
    // in the README up to ~10cm of precision.
    const double ref[5][3] = {{-361605.79234692274, 44981.04302003427, -30693.19198820311},
                              {-363123.49639910535, 35877.2078378671, -33194.68710267386},
                              {-371572.8001113177, 75278.04108874535, -32227.390108194537},
                              {-373885.8585384737, 127397.62026596011, -30039.82215825389},
                              {-346323.78767959465, 206374.952164921, -28496.523303933904}};

    const auto dates = {2469000.5, 2449000.5, 2429000.5, 2409000.5, 2389000.5};

    for (auto i = 0u; i < 5u; ++i) {
        const auto date = *(dates.begin() + i);

        const double tm = (date - 2451545.0) / (36525);
        cf_ptr(out, nullptr, nullptr, &tm);

        REQUIRE(std::abs(out[0] - ref[i][0]) < 1e-10);
        REQUIRE(std::abs(out[1] - ref[i][1]) < 1e-10);
        REQUIRE(std::abs(out[2] - ref[i][2]) < 1e-10);
    }
}
