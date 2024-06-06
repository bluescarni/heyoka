// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/jb08_tn.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("impl")
{

    using Catch::Matchers::Message;
    auto [h, lat, lon, f107a, f107, s107a, s107, m107a, m107, y107a, y107, dDstdT]
        = make_vars("h", "lat", "lon", "f107a", "f107", "s107a", "s107", "m107a", "m107", "y107a", "y107", "dDstdT");

    // First, we test malformed cases and their throws.
    // 1 - Size of the Geodetic coordinates is wrong
    REQUIRE_THROWS_AS(model::detail::jb08_tn_impl({h, lat, lon, h}, f107a, f107, s107a, s107, m107a, m107, y107a, y107,
                                                  dDstdT, heyoka::time / 86400_dbl),
                      std::invalid_argument);

    // Then we test the numerical output
    // Construct the expression for the thermospheric density
    auto rho = model::detail::jb08_tn_impl({h, lat, lon}, f107a, f107, s107a, s107, m107a, m107, y107a, y107, dDstdT,
                                           heyoka::time / 86400_dbl);
    // Produce the compiled function
    cfunc<double> rho_cf{{rho}, {h, lat, lon, f107a, f107, s107a, s107, m107a, m107, y107a, y107, dDstdT}};

    // Case 1 - 1st January 00:00:00
    {
        // Prepare the input-output buffers.
        std::array<double, 12> in{600, 1.2, 3.9, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::array<double, 1> out{};
        // Call the compiled function
        rho_cf(out, in, kw::time = 0.);
        REQUIRE(out[0] == approximately(6.805408788157112e-15));
    }
    // Case 2 - 123.23 days later (different alts etc....)
    {
        // Prepare the input-output buffers.
        std::array<double, 12> in{234, 4.5, 1.02, 11, 10, 9, 8, 7, 6, 5, 4, 3};
        std::array<double, 1> out{};
        // Call the model
        rho_cf(out, in, kw::time = 123.23 * 86400.);
        REQUIRE(out[0] == approximately(1.3364825974582714e-11));
    }
    // Case 2 - 2300.92 days later (different alts etc....)
    {
        // Prepare the input-output buffers.
        std::array<double, 12> in{90., 2., 1., 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
        std::array<double, 1> out{};
        // Call the model
        rho_cf(out, in, kw::time = 2300.92 * 86400.);
        REQUIRE(out[0] == approximately(5.173701008824741e-08));
    }
}

TEST_CASE("igor_iface")
{
    auto [h, lat, lon, f107a, f107, s107a, s107, m107a, m107, y107a, y107, dDstdT]
        = make_vars("h", "lat", "lon", "f107a", "f107", "s107a", "s107", "m107a", "m107", "y107a", "y107", "dDstdT");
    {
        auto igor_v = model::jb08_tn(kw::geodetic = {h, lat, lon}, kw::f107 = f107, kw::f107a = f107a, kw::s107 = s107,
                                     kw::s107a = s107a, kw::m107 = m107, kw::m107a = m107a, kw::y107 = y107,
                                     kw::y107a = y107a, kw::dDstdT = dDstdT, kw::time_expr = heyoka::time / 86400_dbl);
        auto vanilla_v = model::detail::jb08_tn_impl({h, lat, lon}, f107a, f107, s107a, s107, m107a, m107, y107a, y107,
                                                     dDstdT, heyoka::time / 86400_dbl);
        REQUIRE(igor_v == vanilla_v);
    }
}
