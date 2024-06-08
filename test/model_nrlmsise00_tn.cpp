// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/nrlmsise00_tn.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("impl")
{

    using Catch::Matchers::Message;
    auto [h, lat, lon, f107, f107a, ap] = make_vars("h", "lat", "lon", "f107", "f107a", "ap");

    // First, we test malformed cases and their throws.
    // 1 - Size of the Geodetic coordinates is wrong
    REQUIRE_THROWS_AS(model::detail::nrlmsise00_tn_impl({h, lat, lon, h}, f107, f107a, ap, heyoka::time / 86400_dbl),
                      std::invalid_argument);

    // Then we test the numerical output
    // Construct the expression for the thermospheric density
    auto rho = model::detail::nrlmsise00_tn_impl({h, lat, lon}, f107, f107a, ap, heyoka::time / 86400_dbl);
    // Case 1 - 1st January 00:00:00
    {
        // Prepare the input-output buffers.
        std::array<double, 6> in{600, 1.2, 3.9,  21.2, 12.2, 22.};
        std::array<double, 1> out{};
        // Produce the compiled function
        cfunc<double> rho_cf{{rho}, {h, lat, lon, f107, f107a, ap}};
        // Call the model
        rho_cf(out, in, kw::time = 0.);
        REQUIRE(out[0] == approximately(9.599548606663777e-15));
    }
    // Case 2 - 123.23 days later (different alts etc....)
    {
        // Prepare the input-output buffers.
        std::array<double, 6> in{234, 4.5, 1.02, 4, 3, 5};
        std::array<double, 1> out{};
        // Produce the compiled function
        cfunc<double> rho_cf{{rho}, {h, lat, lon, f107, f107a, ap}};
        // Call the model
        rho_cf(out, in, kw::time = 123.23 * 86400.);
        REQUIRE(out[0] == approximately(3.549961466488851e-11));
    }
}

TEST_CASE("igor_iface")
{
    auto [h, lat, lon, f107, f107a, ap] = make_vars("h", "lat", "lon", "f107", "f107a", "ap");
    {
        auto igor_v = model::nrlmsise00_tn(kw::geodetic = {h, lat, lon}, kw::f107 = f107, kw::f107a = f107a, kw::ap = ap, kw::time_expr = heyoka::time / 86400_dbl);
        auto vanilla_v = model::detail::nrlmsise00_tn_impl({h, lat, lon}, f107, f107a, ap, heyoka::time / 86400_dbl);
        REQUIRE(igor_v == vanilla_v);
    }
}
