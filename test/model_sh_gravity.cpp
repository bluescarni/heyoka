// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/sh_gravity.hpp>

#include "catch.hpp"

using namespace heyoka;

template <typename T>
concept can_sh_gravity_pot = requires(const std::array<expression, 3> &xyz, double a, double mu, T coeffs) {
    model::sh_gravity_pot(xyz, kw::a = a, kw::mu = mu, kw::sh_coefficients = std::forward<T>(coeffs));
};

TEST_CASE("basics")
{
    const auto xyz = make_vars("x", "y", "z");

    // Keplerian potential tests.
    {
        auto pot = model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                                         kw::sh_coefficients = std::vector{std::array{2., 1.}});
        REQUIRE(pot == par[0] * (2. / sqrt(sum({pow(xyz[0], 2.), pow(xyz[1], 2.), pow(xyz[2], 2.)}))));
    }
    {
        auto pot = model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                                         kw::sh_coefficients = std::vector{std::array{2_dbl, 1_dbl}});
        REQUIRE(pot == par[0] * (2. / sqrt(sum({pow(xyz[0], 2.), pow(xyz[1], 2.), pow(xyz[2], 2.)}))));
    }
    {
        auto pot = model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                                         kw::sh_coefficients = std::vector{std::array{"par0", "par1"}});
        REQUIRE(pot == par[0] * ("par0"_var / sqrt(sum({pow(xyz[0], 2.), pow(xyz[1], 2.), pow(xyz[2], 2.)}))));
    }
    {
        const std::array<double[2], 1> cfs = {{{2., 3.}}};

        auto pot = model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0], kw::sh_coefficients = cfs);
        REQUIRE(pot == par[0] * (2. / sqrt(sum({pow(xyz[0], 2.), pow(xyz[1], 2.), pow(xyz[2], 2.)}))));
    }

    // Keplerian acceleration tests.
    {
        auto acc = model::sh_gravity_acc(xyz, kw::a = 1., kw::mu = 1_dbl,
                                         kw::sh_coefficients = std::vector{std::array{1., 1.}});
        const auto spow = sum({pow(xyz[0], 2.), pow(xyz[1], 2.), pow(xyz[2], 2.)});
        REQUIRE(acc[0] == (-0.57735026918962573 * (1.7320508075688772 * ((xyz[0] / spow) / pow(spow, 0.5)))));
        REQUIRE(acc[1] == (-0.57735026918962573 * (1.7320508075688772 * ((xyz[1] / spow) / pow(spow, 0.5)))));
        REQUIRE(acc[2] == (-0.57735026918962573 * ((1.7320508075688772 * (xyz[2] / spow) / pow(spow, 0.5)))));
    }

    // Concept checking tests.
    REQUIRE(can_sh_gravity_pot<std::vector<std::array<double, 2>>>);
    REQUIRE(can_sh_gravity_pot<std::vector<std::array<long double, 2>>>);
    REQUIRE(can_sh_gravity_pot<std::vector<std::array<float, 2>>>);
    REQUIRE(can_sh_gravity_pot<std::vector<std::array<const char *, 2>>>);
    REQUIRE(can_sh_gravity_pot<std::vector<std::array<std::string, 2>>>);
    REQUIRE(can_sh_gravity_pot<std::array<double[2], 1>>);

    REQUIRE(!can_sh_gravity_pot<std::vector<std::array<double, 1>>>);
    REQUIRE(!can_sh_gravity_pot<std::vector<std::array<double, 3>>>);
    REQUIRE(!can_sh_gravity_pot<std::vector<int>>);
}

TEST_CASE("failure modes")
{
    using Catch::Matchers::Message;

    const auto xyz = make_vars("x", "y", "z");

    REQUIRE_THROWS_MATCHES(
        model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                              kw::sh_coefficients = std::vector<std::array<double, 2>>{}),
        std::invalid_argument,
        Message("A custom spherical harmonics gravity model cannot be created from an empty list of C/S coefficients"));
    REQUIRE_THROWS_MATCHES(
        model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                              kw::sh_coefficients = std::vector{std::array{1., 2.}, std::array{3., 4.}}),
        std::invalid_argument,
        Message("Invalid custom spherical harmonics gravity model: the list of coefficients has a size of 2, which is "
                "not equal to n*(n+1)/2 for any natural number n"));
    REQUIRE_THROWS_MATCHES(
        model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                              kw::sh_coefficients
                              = std::vector{std::array{1., 2.}, std::array{3., 4.}, std::array{3., 4.}},
                              kw::max_degree = 4),
        std::invalid_argument,
        Message("Invalid maximum degree 4 specified for a custom spherical harmonics gravity model: it is larger "
                "than the maximum degree 1 supported by the provided C/S coefficients"));
    REQUIRE_THROWS_MATCHES(
        model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                              kw::sh_coefficients
                              = std::vector{std::array{1., 2.}, std::array{3., 4.}, std::array{3., 4.}},
                              kw::max_order = 4),
        std::invalid_argument,
        Message("Cannot instantiate a custom spherical harmonics gravity model when only the maximum order is "
                "specified - please provide a maximum degree as well"));
    REQUIRE_THROWS_MATCHES(
        model::sh_gravity_pot(xyz, kw::a = 1., kw::mu = par[0],
                              kw::sh_coefficients
                              = std::vector{std::array{1., 2.}, std::array{3., 4.}, std::array{3., 4.}},
                              kw::max_degree = 1, kw::max_order = 4),
        std::invalid_argument,
        Message("Invalid combination of degree and order specified for a spherical harmonics gravity model: the order "
                "4 is greater than the degree 1"));
}
