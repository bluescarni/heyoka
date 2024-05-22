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
    auto [h, lat, lon, f107a, f107, ap] = make_vars("h", "lat", "lon", "f107a", "f107", "ap");
    auto rho
        = model::detail::nrlmsise00_tn_impl({h, lat, lon}, f107a, f107, ap, heyoka::time / 60_dbl / 60_dbl / 24_dbl);
    cfunc<double> rho_cf{{rho}, {h, lat, lon, f107a, f107, ap}};
    std::array<double, 6> in{600, 1.2, 3.9, 12.2, 21.2, 22.};
    std::array<double, 1> out{};
    rho_cf(out, in, kw::time = 0.);
    fmt::print("{}", out);
}
