// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/iau2006.hpp>

#include "catch.hpp"

using namespace heyoka;
using namespace heyoka::model;

// NOTE: these are the erfa routines used for validation.
extern "C" {
void eraXy06(double date1, double date2, double *x, double *y);
double eraS06(double date1, double date2, double x, double y);
}

std::mt19937 rng;

static const int ntrials = 1000;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    heyoka::detail::edb_disabler ed;

    // Error modes.
    REQUIRE_THROWS_MATCHES(model::iau2006(kw::thresh = -1.), std::invalid_argument,
                           Message("Invalid threshold value passed to iau2006(): "
                                   "the value must be finite and non-negative, but it is -1 instead"));

    // Create a compiled function containing the full solution.
    const auto sol = iau2006(kw::thresh = 0.);
    cfunc<double> cf(sol, {}, kw::compact_mode = true);

    // Evaluate at several time coordinates around J2000 and validate against erfa.
    std::vector<double> input, output = {0., 0., 0.};
    std::uniform_real_distribution<double> dist(-50 * 365.25, 50 * 365.25);
    for (auto i = 0; i < ntrials; ++i) {
        // Generate a random number of days since J2000.
        const auto delta_days = dist(rng);

        // heyoka.
        cf(output, input, kw::time = delta_days / 36525);

        // erfa.
        double x{}, y{};
        eraXy06(2451545.0, delta_days, &x, &y);
        const auto s = eraS06(2451545.0, delta_days, x, y);

        // Compare.
        REQUIRE(std::abs((output[0] - x) / x) <= 1e-10);
        REQUIRE(std::abs((output[1] - y) / y) <= 1e-10);
        REQUIRE(std::abs((output[2] - s) / s) <= 1e-10);
    }
}

// A test for the threshold.
TEST_CASE("thresh")
{
    using Catch::Matchers::Message;

    heyoka::detail::edb_disabler ed;

    const auto sol = iau2006();
    cfunc<double> cf(sol, {}, kw::compact_mode = true);

    // Evaluate at several time coordinates around J2000 and validate against erfa.
    std::vector<double> input, output = {0., 0., 0.};
    std::uniform_real_distribution<double> dist(-50 * 365.25, 50 * 365.25);
    for (auto i = 0; i < ntrials; ++i) {
        // Generate a random number of days since J2000.
        const auto delta_days = dist(rng);

        // heyoka.
        cf(output, input, kw::time = delta_days / 36525);

        // erfa.
        double x{}, y{};
        eraXy06(2451545.0, delta_days, &x, &y);
        const auto s = eraS06(2451545.0, delta_days, x, y);

        // Compare.
        REQUIRE(std::abs((output[0] - x) / x) <= 1e-8);
        REQUIRE(std::abs((output[1] - y) / y) <= 1e-8);
        REQUIRE(std::abs((output[2] - s) / s) <= 1e-8);
    }
}
