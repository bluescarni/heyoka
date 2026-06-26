// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/model/sh_gravity.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

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
    // Array-like element whose type cannot be used to construct an expression (fails expression_array_ctible).
    REQUIRE(!can_sh_gravity_pot<std::vector<std::array<std::vector<int>, 2>>>);
    // Not a range at all.
    REQUIRE(!can_sh_gravity_pot<int>);
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

// Check that a custom model built from a subset of the egm2008 coefficients reproduces the output of the egm2008  model
// functions. This also exercises the std::optional code paths for the max_degree/max_order kwargs.
TEST_CASE("egm2008 comparison")
{
    heyoka::detail::edb_disabler ed;

    const auto xyz = make_vars("x", "y", "z");
    const std::vector vars{xyz[0], xyz[1], xyz[2]};

    // Use the same gravitational parameter/reference radius as the egm2008 model.
    const auto mu = model::get_egm2008_mu();
    const auto a = model::get_egm2008_a();

    // Assemble the full flattened list of C/S coefficients up to degree 16 from the egm2008 dataset. The dataset begins
    // at degree 2, so we manually prepend the degree 0 and 1 terms.
    constexpr std::uint32_t max_deg = 16;
    const auto cs = model::get_egm2008_CS();

    std::vector<std::array<expression, 2>> coeffs;
    coeffs.push_back({1_dbl, 0_dbl}); // (0, 0)
    coeffs.push_back({0_dbl, 0_dbl}); // (1, 0)
    coeffs.push_back({0_dbl, 0_dbl}); // (1, 1)
    // Number of (degree, order) pairs for degrees 2..max_deg.
    const auto n_rows = (max_deg + 1u) * (max_deg + 2u) / 2u - 3u;
    for (std::size_t k = 0; k < n_rows; ++k) {
        coeffs.push_back({expression{cs[k, 0]}, expression{cs[k, 1]}});
    }

    // A handful of evaluation points (all away from the origin).
    const std::vector<std::array<double, 3>> pts{{a * 1.5, 0., 0.},      {0., a * 2., 0.},
                                                 {0., 0., a * 1.2},      {a, a, a},
                                                 {a * 0.8, a * 1.3, -a}, {-a * 1.1, a * 0.4, a * 0.9}};

    // Helper: compare the custom potential/acceleration against the egm2008 ones at degree n, order m.
    auto compare = [&](const expression &c_pot, const std::array<expression, 3> &c_acc, const std::uint32_t n,
                       const std::uint32_t m) {
        auto cf_c_pot = cfunc<double>({c_pot}, vars, kw::compact_mode = true);
        auto cf_e_pot = cfunc<double>({model::egm2008_pot(xyz, n, m)}, vars, kw::compact_mode = true);
        auto cf_c_acc = cfunc<double>(c_acc, vars, kw::compact_mode = true);
        auto cf_e_acc = cfunc<double>(model::egm2008_acc(xyz, n, m), vars, kw::compact_mode = true);

        std::vector<double> in(3), c_po(1), e_po(1), c_ac(3), e_ac(3);
        for (const auto &p : pts) {
            in[0] = p[0];
            in[1] = p[1];
            in[2] = p[2];

            cf_c_pot(c_po, in);
            cf_e_pot(e_po, in);
            REQUIRE(c_po[0] == approximately(e_po[0]));

            cf_c_acc(c_ac, in);
            cf_e_acc(e_ac, in);
            REQUIRE(c_ac[0] == approximately(e_ac[0]));
            REQUIRE(c_ac[1] == approximately(e_ac[1]));
            REQUIRE(c_ac[2] == approximately(e_ac[2]));
        }
    };

    // Case 1: max_degree/max_order passed as engaged optionals - full 16x16 model.
    {
        const std::optional<std::uint32_t> md{16}, mo{16};
        auto c_pot = model::sh_gravity_pot(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs,
                                           kw::max_degree = md, kw::max_order = mo);
        auto c_acc = model::sh_gravity_acc(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs,
                                           kw::max_degree = md, kw::max_order = mo);
        compare(c_pot, c_acc, 16, 16);
    }

    // Case 2: max_degree/max_order passed as empty optionals - defaults to the full 16x16 model.
    {
        const std::optional<std::uint32_t> md{}, mo{};
        auto c_pot = model::sh_gravity_pot(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs,
                                           kw::max_degree = md, kw::max_order = mo);
        auto c_acc = model::sh_gravity_acc(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs,
                                           kw::max_degree = md, kw::max_order = mo);
        compare(c_pot, c_acc, 16, 16);
    }

    // Case 3: only max_degree provided (engaged optional), order defaults to the degree - truncated 8x8 model.
    {
        const std::optional<std::uint32_t> md{8};
        auto c_pot
            = model::sh_gravity_pot(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs, kw::max_degree = md);
        auto c_acc
            = model::sh_gravity_acc(xyz, kw::mu = mu, kw::a = a, kw::sh_coefficients = coeffs, kw::max_degree = md);
        compare(c_pot, c_acc, 8, 8);
    }
}

// Check that the C/S coefficients can be provided via a lazy input range whose reference type is a prvalue (rather than
// an lvalue reference), exercising the value-category handling in sh_gravity_common_opts().
TEST_CASE("prvalue coefficients")
{
    const auto xyz = make_vars("x", "y", "z");

    // Map a flat coefficient index to a C/S pair.
    auto make_cf = [](std::size_t i) { return std::array{static_cast<double>(i) + 1., static_cast<double>(i) + 2.}; };

    constexpr std::size_t n = 6; // degree 2 -> 6 coefficient pairs.

    // Reference: a materialised vector of coefficients (lvalue references when iterated).
    std::vector<std::array<double, 2>> ref_coeffs;
    for (std::size_t i = 0; i < n; ++i) {
        ref_coeffs.push_back(make_cf(i));
    }

    // A lazy range yielding the same coefficients as prvalue std::array on the fly.
    auto lazy_coeffs = std::views::iota(std::size_t(0), n) | std::views::transform(make_cf);

    // Building the model from the lazy prvalue range must yield exactly the same expressions as from the materialised
    // vector.
    REQUIRE(model::sh_gravity_pot(xyz, kw::mu = par[0], kw::a = par[1], kw::sh_coefficients = ref_coeffs)
            == model::sh_gravity_pot(xyz, kw::mu = par[0], kw::a = par[1], kw::sh_coefficients = lazy_coeffs));

    REQUIRE(model::sh_gravity_acc(xyz, kw::mu = par[0], kw::a = par[1], kw::sh_coefficients = ref_coeffs)
            == model::sh_gravity_acc(xyz, kw::mu = par[0], kw::a = par[1], kw::sh_coefficients = lazy_coeffs));

    // The lazy prvalue range is also accepted by the input-range concept.
    REQUIRE(can_sh_gravity_pot<decltype(lazy_coeffs)>);
}
