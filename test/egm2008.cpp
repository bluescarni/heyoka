// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <fmt/core.h>

#include <heyoka/detail/egm2008.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/model/egm2008.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("error handling")
{
    using Catch::Matchers::Message;

    const auto [x, y, z] = make_vars("x", "y", "z");

    REQUIRE_THROWS_MATCHES(
        model::egm2008_pot({x, y, z}, model::detail::egm2008_max_degree + 1u, 0), std::invalid_argument,
        Message(fmt::format("Invalid degree specified for the EGM2008 geopotential model: the maximum degree is "
                            "{}, but a degree of {} was specified",
                            model::detail::egm2008_max_degree, model::detail::egm2008_max_degree + 1u)));

    REQUIRE_THROWS_MATCHES(model::egm2008_pot({x, y, z}, 0, 1), std::invalid_argument,
                           Message("Invalid combination of degree and order specified for the EGM2008 "
                                   "geopotential model: the order 1 is greater than the degree 0"));
}

// A few basic checks against manually-computed results.
TEST_CASE("basic")
{
    using std::cos;
    using std::sin;

    const auto [x, y, z] = make_vars("x", "y", "z");

    // NOTE: up to order 1 only the Keplerian term must survive.
    REQUIRE(model::egm2008_pot({x, y, z}, 0, 0) == model::egm2008_pot({x, y, z}, 1, 0));
    REQUIRE(model::egm2008_pot({x, y, z}, 0, 0) == model::egm2008_pot({x, y, z}, 1, 1));

    // Prepare input/output for the compiled functions.
    std::vector<double> out(1), in(3);

    // A simple test with J2 only, computed manually.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 2, 0)}, {x, y, z});
        // Pick some longitude (irrelevant) and colatitude.
        const auto phi = 42.;
        const auto theta = 0.123;
        // Pick a radius.
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62418910.63181418));
    }

    // A simple test up to C22/S22, computed manually.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 2, 2)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62418905.70696879));
    }

    // Test with n=2, m=1.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 2, 1)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62418910.596871205));
    }

    // A simple test up to C33/S33, computed manually.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 3, 3)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62419001.277551234));
    }

    // Test with n=3, m=0.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 3, 0)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62419061.693081796));
    }

    // Test with n=3, m=1.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 3, 1)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62419011.00138944));
    }

    // Test with n=3, m=2.
    {
        auto cf = cfunc<double>({model::egm2008_pot({x, y, z}, 3, 2)}, {x, y, z});
        const auto phi = 42.;
        const auto theta = 0.123;
        const auto r = 6378137.0 + 1000.0;

        const auto X = r * sin(theta) * cos(phi);
        const auto Y = r * sin(theta) * sin(phi);
        const auto Z = r * cos(theta);

        in[0] = X;
        in[1] = Y;
        in[2] = Z;

        cf(out, in);

        REQUIRE(out[0] == approximately(62419001.00066965));
    }
}
