// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/detail/debug.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/s11n.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

const auto revday2radmin = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 1440.; };
const auto deg2rad = [](auto x) { return x * 2. * boost::math::constants::pi<double>() / 360.; };

TEST_CASE("model expression")
{
    using namespace heyoka::literals;
    using Catch::Matchers::Message;

    detail::edb_disabler ed;

    auto outputs = model::sgp4();
    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");

    auto sgp4_cf = cfunc<double>(outputs, inputs);

    {
        std::vector<double> ins = {revday2radmin(15.50103472202482),
                                   0.0007417,
                                   deg2rad(51.6439),
                                   deg2rad(211.2001),
                                   deg2rad(17.6667),
                                   deg2rad(85.6398),
                                   .38792e-4,
                                   0.},
                            outs(7u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(3469.947984145807, 10000.));
        REQUIRE(outs[1] == approximately(-2690.388430131083, 10000.));
        REQUIRE(outs[2] == approximately(5175.831924199492, 10000.));
        REQUIRE(outs[3] == approximately(5.810229142351453, 10000.));
        REQUIRE(outs[4] == approximately(4.802261184784617, 10000.));
        REQUIRE(outs[5] == approximately(-1.388280333072693, 10000.));
        REQUIRE(outs[6] == 0.);
    }

    {
        std::vector<double> ins = {revday2radmin(15.50103472202482),
                                   0.0007417,
                                   deg2rad(51.6439),
                                   deg2rad(211.2001),
                                   deg2rad(17.6667),
                                   deg2rad(85.6398),
                                   .38792e-4,
                                   1440.},
                            outs(7u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(-3591.82683131782, 10000.));
        REQUIRE(outs[1] == approximately(2723.666407193435, 10000.));
        REQUIRE(outs[2] == approximately(-5090.448264983512, 10000.));
        REQUIRE(outs[3] == approximately(-5.927709516654264, 10000.));
        REQUIRE(outs[4] == approximately(-4.496384419253211, 10000.));
        REQUIRE(outs[5] == approximately(1.785277174529374, 10000.));
        REQUIRE(outs[6] == 0.);
    }

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738),
                                   0.75863e-3,
                                   0.},
                            outs(7u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(2561.223660636298, 10000.));
        REQUIRE(outs[1] == approximately(3698.797144057697, 10000.));
        REQUIRE(outs[2] == approximately(5818.772215708888, 10000.));
        REQUIRE(outs[3] == approximately(-3.276142513618007, 10000.));
        REQUIRE(outs[4] == approximately(-4.806489082829041, 10000.));
        REQUIRE(outs[5] == approximately(4.511134501638151, 10000.));
        REQUIRE(outs[6] == 0.);
    }

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738),
                                   0.75863e-3,
                                   1440.},
                            outs(7u);

        sgp4_cf(outs, ins);

        REQUIRE(outs[0] == approximately(3134.2015620939, 10000.));
        REQUIRE(outs[1] == approximately(4604.963663328277, 10000.));
        REQUIRE(outs[2] == approximately(-4791.661126560278, 10000.));
        REQUIRE(outs[3] == approximately(2.732034613044249, 10000.));
        REQUIRE(outs[4] == approximately(3.952589777415254, 10000.));
        REQUIRE(outs[5] == approximately(5.588906721377138, 10000.));
        REQUIRE(outs[6] == 0.);
    }

    // Test also non-default expressions for the sgp4 inputs.
    outputs = model::sgp4({"a"_var, "b"_var, "c"_var, "d"_var, "e"_var, "f"_var, par[0], heyoka::time});
    const auto inputs2 = make_vars("a", "b", "c", "d", "e", "f");

    sgp4_cf = cfunc<double>(outputs, inputs2);

    {
        std::vector<double> ins = {revday2radmin(13.75091047972192),
                                   0.0024963,
                                   deg2rad(90.2039),
                                   deg2rad(55.5633),
                                   deg2rad(320.5956),
                                   deg2rad(91.4738)},
                            pars = {0.75863e-3}, outs(7u);
        double time = 1440.;

        sgp4_cf(outs, ins, kw::pars = pars, kw::time = time);

        REQUIRE(outs[0] == approximately(3134.2015620939, 10000.));
        REQUIRE(outs[1] == approximately(4604.963663328277, 10000.));
        REQUIRE(outs[2] == approximately(-4791.661126560278, 10000.));
        REQUIRE(outs[3] == approximately(2.732034613044249, 10000.));
        REQUIRE(outs[4] == approximately(3.952589777415254, 10000.));
        REQUIRE(outs[5] == approximately(5.588906721377138, 10000.));
        REQUIRE(outs[6] == 0.);
    }

    // Error checking.
    REQUIRE_THROWS_MATCHES(model::sgp4({"a"_var, "b"_var, "c"_var, "d"_var}), std::invalid_argument,
                           Message("Invalid number of inputs passed to the sgp4() function: 8 "
                                   "expressions are expected but 4 were provided instead"));
}

TEST_CASE("propagator basics")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    REQUIRE_NOTHROW(prop_t{});

    // Copy construction.
    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    const auto tm = std::array{1440., 0.};
    const prop_t::in_1d<double> tm_in{tm.data(), 2};

    prop_t prop{md_input_t{ins.data(), 2}};
    REQUIRE(prop.get_diff_order() == 0u);
    auto prop2 = prop;
    REQUIRE(prop2.get_nsats() == 2u);
    REQUIRE(prop.get_sat_data().extent(0) == 9u);
    REQUIRE(prop.get_sat_data().extent(1) == 2u);
    REQUIRE(prop.get_sat_data()(0, 0) == revday2radmin(13.75091047972192));
    REQUIRE(prop.get_sat_data()(0, 1) == revday2radmin(15.50103472202482));
    REQUIRE(prop.get_sat_data()(1, 0) == 0.0024963);
    REQUIRE(prop.get_sat_data()(1, 1) == 0.0007417);

    // Move construction.
    auto prop3 = std::move(prop2);
    REQUIRE(prop3.get_nsats() == 2u);

    // Revive prop2 via copy assignment.
    prop2 = prop3;
    REQUIRE(prop2.get_nsats() == 2u);

    // Revive via move assignment.
    prop_t prop4;
    prop4 = std::move(prop2);
    REQUIRE(prop4.get_nsats() == 2u);
}

TEST_CASE("propagator single")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    const auto tm = std::array{1440., 0.};
    const prop_t::in_1d<double> tm_in{tm.data(), 2};

    for (auto cm : {false, true}) {
        prop_t prop{md_input_t{ins.data(), 2}, kw::compact_mode = cm};
        REQUIRE(prop.get_nouts() == 7u);

        std::vector<double> outs(14u);
        prop_t::out_2d out{outs.data(), 7, 2};

        prop(out, tm_in);

        REQUIRE(out(0, 0) == approximately(3134.2015620939, 10000.));
        REQUIRE(out(1, 0) == approximately(4604.963663328277, 10000.));
        REQUIRE(out(2, 0) == approximately(-4791.661126560278, 10000.));
        REQUIRE(out(3, 0) == approximately(2.732034613044249, 10000.));
        REQUIRE(out(4, 0) == approximately(3.952589777415254, 10000.));
        REQUIRE(out(5, 0) == approximately(5.588906721377138, 10000.));
        REQUIRE(out(6, 0) == 0.);
        REQUIRE(out(0, 1) == approximately(3469.947984145807, 10000.));
        REQUIRE(out(1, 1) == approximately(-2690.388430131083, 10000.));
        REQUIRE(out(2, 1) == approximately(5175.831924199492, 10000.));
        REQUIRE(out(3, 1) == approximately(5.810229142351453, 10000.));
        REQUIRE(out(4, 1) == approximately(4.802261184784617, 10000.));
        REQUIRE(out(5, 1) == approximately(-1.388280333072693, 10000.));
        REQUIRE(out(6, 1) == 0.);

        auto dates
            = std::array<prop_t::date, 2>{{{2460486.5 + 1, 0.6478633000000116}, {2458826.5, 0.6933954099999937}}};
        prop_t::in_1d<prop_t::date> date_in{dates.data(), 2};

        prop(out, date_in);

        REQUIRE(out(0, 0) == approximately(3134.2015620939, 10000.));
        REQUIRE(out(1, 0) == approximately(4604.963663328277, 10000.));
        REQUIRE(out(2, 0) == approximately(-4791.661126560278, 10000.));
        REQUIRE(out(3, 0) == approximately(2.732034613044249, 10000.));
        REQUIRE(out(4, 0) == approximately(3.952589777415254, 10000.));
        REQUIRE(out(5, 0) == approximately(5.588906721377138, 10000.));
        REQUIRE(out(6, 0) == 0.);
        REQUIRE(out(0, 1) == approximately(3469.947984145807, 10000.));
        REQUIRE(out(1, 1) == approximately(-2690.388430131083, 10000.));
        REQUIRE(out(2, 1) == approximately(5175.831924199492, 10000.));
        REQUIRE(out(3, 1) == approximately(5.810229142351453, 10000.));
        REQUIRE(out(4, 1) == approximately(4.802261184784617, 10000.));
        REQUIRE(out(5, 1) == approximately(-1.388280333072693, 10000.));
        REQUIRE(out(6, 1) == 0.);

        // Try with several bogus input spans.
        REQUIRE_THROWS_AS(prop(prop_t::out_2d{outs.data(), 5, 2}, date_in), std::invalid_argument);
        REQUIRE_THROWS_AS(prop(out, prop_t::in_1d<double>{ins.data(), 1}), std::invalid_argument);
    }
}

TEST_CASE("propagator batch")
{
    detail::edb_disabler ed;

    using Catch::Matchers::Message;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    const auto tm = std::array{1440., 0., 0., 1440.};
    const prop_t::in_2d<double> tm_in{tm.data(), 2, 2};

    for (auto cm : {false, true}) {
        prop_t prop{md_input_t{ins.data(), 2}, kw::compact_mode = cm};

        std::vector<double> outs(28u);
        prop_t::out_3d out{outs.data(), 2, 7, 2};

        prop(out, tm_in);

        REQUIRE(out(0, 0, 0) == approximately(3134.2015620939, 10000.));
        REQUIRE(out(0, 1, 0) == approximately(4604.963663328277, 10000.));
        REQUIRE(out(0, 2, 0) == approximately(-4791.661126560278, 10000.));
        REQUIRE(out(0, 3, 0) == approximately(2.732034613044249, 10000.));
        REQUIRE(out(0, 4, 0) == approximately(3.952589777415254, 10000.));
        REQUIRE(out(0, 5, 0) == approximately(5.588906721377138, 10000.));
        REQUIRE(out(0, 6, 0) == 0.);
        REQUIRE(out(0, 0, 1) == approximately(3469.947984145807, 10000.));
        REQUIRE(out(0, 1, 1) == approximately(-2690.388430131083, 10000.));
        REQUIRE(out(0, 2, 1) == approximately(5175.831924199492, 10000.));
        REQUIRE(out(0, 3, 1) == approximately(5.810229142351453, 10000.));
        REQUIRE(out(0, 4, 1) == approximately(4.802261184784617, 10000.));
        REQUIRE(out(0, 5, 1) == approximately(-1.388280333072693, 10000.));
        REQUIRE(out(0, 6, 1) == 0.);
        REQUIRE(out(1, 0, 0) == approximately(2561.223660636298, 10000.));
        REQUIRE(out(1, 1, 0) == approximately(3698.797144057697, 10000.));
        REQUIRE(out(1, 2, 0) == approximately(5818.772215708888, 10000.));
        REQUIRE(out(1, 3, 0) == approximately(-3.276142513618007, 10000.));
        REQUIRE(out(1, 4, 0) == approximately(-4.806489082829041, 10000.));
        REQUIRE(out(1, 5, 0) == approximately(4.511134501638151, 10000.));
        REQUIRE(out(1, 6, 0) == 0.);
        REQUIRE(out(1, 0, 1) == approximately(-3591.82683131782, 10000.));
        REQUIRE(out(1, 1, 1) == approximately(2723.666407193435, 10000.));
        REQUIRE(out(1, 2, 1) == approximately(-5090.448264983512, 10000.));
        REQUIRE(out(1, 3, 1) == approximately(-5.927709516654264, 10000.));
        REQUIRE(out(1, 4, 1) == approximately(-4.496384419253211, 10000.));
        REQUIRE(out(1, 5, 1) == approximately(1.785277174529374, 10000.));
        REQUIRE(out(1, 6, 1) == 0.);

        auto dates = std::array<prop_t::date, 4>{{{2460486.5 + 1, 0.6478633000000116},
                                                  {2458826.5, 0.6933954099999937},
                                                  {2460486.5, 0.6478633000000116},
                                                  {2458826.5 + 1, 0.6933954099999937}}};
        prop_t::in_2d<prop_t::date> date_in{dates.data(), 2, 2};

        prop(out, date_in);

        REQUIRE(out(0, 0, 0) == approximately(3134.2015620939, 10000.));
        REQUIRE(out(0, 1, 0) == approximately(4604.963663328277, 10000.));
        REQUIRE(out(0, 2, 0) == approximately(-4791.661126560278, 10000.));
        REQUIRE(out(0, 3, 0) == approximately(2.732034613044249, 10000.));
        REQUIRE(out(0, 4, 0) == approximately(3.952589777415254, 10000.));
        REQUIRE(out(0, 5, 0) == approximately(5.588906721377138, 10000.));
        REQUIRE(out(0, 6, 0) == 0.);
        REQUIRE(out(0, 0, 1) == approximately(3469.947984145807, 10000.));
        REQUIRE(out(0, 1, 1) == approximately(-2690.388430131083, 10000.));
        REQUIRE(out(0, 2, 1) == approximately(5175.831924199492, 10000.));
        REQUIRE(out(0, 3, 1) == approximately(5.810229142351453, 10000.));
        REQUIRE(out(0, 4, 1) == approximately(4.802261184784617, 10000.));
        REQUIRE(out(0, 5, 1) == approximately(-1.388280333072693, 10000.));
        REQUIRE(out(0, 6, 1) == 0.);
        REQUIRE(out(1, 0, 0) == approximately(2561.223660636298, 10000.));
        REQUIRE(out(1, 1, 0) == approximately(3698.797144057697, 10000.));
        REQUIRE(out(1, 2, 0) == approximately(5818.772215708888, 10000.));
        REQUIRE(out(1, 3, 0) == approximately(-3.276142513618007, 10000.));
        REQUIRE(out(1, 4, 0) == approximately(-4.806489082829041, 10000.));
        REQUIRE(out(1, 5, 0) == approximately(4.511134501638151, 10000.));
        REQUIRE(out(1, 6, 0) == 0.);
        REQUIRE(out(1, 0, 1) == approximately(-3591.82683131782, 10000.));
        REQUIRE(out(1, 1, 1) == approximately(2723.666407193435, 10000.));
        REQUIRE(out(1, 2, 1) == approximately(-5090.448264983512, 10000.));
        REQUIRE(out(1, 3, 1) == approximately(-5.927709516654264, 10000.));
        REQUIRE(out(1, 4, 1) == approximately(-4.496384419253211, 10000.));
        REQUIRE(out(1, 5, 1) == approximately(1.785277174529374, 10000.));
        REQUIRE(out(1, 6, 1) == 0.);

        // Check that nothing bad happens with zero evals.
        prop(prop_t::out_3d{outs.data(), 0, 7, 2}, prop_t::in_2d<double>{tm.data(), 0, 2});

        // Try with several bogus input spans.
        REQUIRE_THROWS_AS(prop(prop_t::out_3d{outs.data(), 2, 5, 2}, date_in), std::invalid_argument);
        REQUIRE_THROWS_AS(prop(prop_t::out_3d{outs.data(), 2, 4, 1}, date_in), std::invalid_argument);
        REQUIRE_THROWS_AS(prop(out, prop_t::in_2d<double>{ins.data(), 2, 1}), std::invalid_argument);
        REQUIRE_THROWS_AS(prop(out, prop_t::in_2d<double>{ins.data(), 2, 0}), std::invalid_argument);
    }
}

TEST_CASE("error handling")
{
    detail::edb_disabler ed;

    using Catch::Matchers::Message;
    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;
    using prop_t = model::sgp4_propagator<double>;

    // Propagator with null list or zero satellites.
    REQUIRE_THROWS_MATCHES((prop_t{md_input_t{nullptr, 0}}), std::invalid_argument,
                           Message("Cannot initialise an sgp4_propagator with an empty list of satellites"));

    std::vector<double> input(9u);

    REQUIRE_THROWS_MATCHES((prop_t{md_input_t{input.data(), 0}}), std::invalid_argument,
                           Message("Cannot initialise an sgp4_propagator with an empty list of satellites"));

    std::vector<double> ins = {revday2radmin(13.75091047972192),
                               revday2radmin(15.50103472202482),
                               0.0024963,
                               0.0007417,
                               deg2rad(90.2039),
                               deg2rad(51.6439),
                               deg2rad(55.5633),
                               deg2rad(211.2001),
                               deg2rad(320.5956),
                               deg2rad(17.6667),
                               deg2rad(91.4738),
                               deg2rad(85.6398),
                               0.75863e-3,
                               .38792e-4,
                               2460486.5,
                               2458826.5,
                               0.6478633000000116,
                               0.6933954099999937};

    prop_t prop{md_input_t{ins.data(), 2}};

    auto dates = std::array<prop_t::date, 2>{{{2460486.5 + 1, 0.6478633000000116}, {0., 1.}}};

    std::vector<double> outs(12u);
    prop_t::out_2d out{outs.data(), 7, 2};

    prop_t::in_1d<prop_t::date> date_in2{dates.data(), 1};

    REQUIRE_THROWS_MATCHES(
        prop(out, date_in2), std::invalid_argument,
        Message("Invalid array of dates passed to the call operator of an sgp4_propagator: the number of "
                "satellites is 2, while the number of dates is 1"));

    auto dates_batch = std::array<prop_t::date, 4>{{{2460486.5 + 1, 0.6478633000000116},
                                                    {2458826.5, 0.6933954099999937},
                                                    {2460486.5, 0.6478633000000116},
                                                    {2458826.5 + 1, 0.6933954099999937}}};
    prop_t::in_2d<prop_t::date> date_b{dates_batch.data(), 1, 2};

    std::vector<double> outs_batch(24u);
    prop_t::out_3d out_batch{outs.data(), 2, 7, 2};

    REQUIRE_THROWS_MATCHES(
        prop(out_batch, date_b), std::invalid_argument,
        Message("Invalid dimensions detected in batch-mode sgp4 propagation: the number of evaluations "
                "inferred from the output array is 2, which is not consistent with the number of evaluations "
                "inferred from the times array (1)"));

    date_b = prop_t::in_2d<prop_t::date>{dates_batch.data(), 1, 1};

    REQUIRE_THROWS_MATCHES(
        prop(out_batch, date_b), std::invalid_argument,
        Message("Invalid array of dates passed to the batch-mode call operator of an sgp4_propagator: the number of "
                "satellites is 2, while the number of dates is per evaluation is 1"));

    // Check that requesting diff information throws on a propagator
    // which was constructed without derivatives.
    REQUIRE_THROWS_MATCHES(
        prop.get_dslice(0), std::invalid_argument,
        Message("The function 'get_dslice()' cannot be invoked on an sgp4 propagator without derivatives"));
    REQUIRE_THROWS_MATCHES(
        prop.get_dslice(0, 0), std::invalid_argument,
        Message("The function 'get_dslice()' cannot be invoked on an sgp4 propagator without derivatives"));
    REQUIRE_THROWS_MATCHES(
        prop.get_mindex(0), std::invalid_argument,
        Message("The function 'get_mindex()' cannot be invoked on an sgp4 propagator without derivatives"));
    REQUIRE_THROWS_MATCHES(
        prop.get_diff_args(), std::invalid_argument,
        Message("The function 'get_diff_args()' cannot be invoked on an sgp4 propagator without derivatives"));
}

TEST_CASE("derivatives")
{
    using Catch::Matchers::Message;

    detail::edb_disabler ed;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;
    using prop_t = model::sgp4_propagator<double>;

    // First compute the order-2 derivatives of the whole model.
    const auto sgp4_func = model::sgp4();
    const auto inputs = make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar", "tsince");
    const auto dt = diff_tensors(sgp4_func, std::vector(inputs.begin(), inputs.begin() + 7), kw::diff_order = 2);

    // Make a compiled function with the derivatives.
    auto diff_cf = cfunc<double>(dt | std::views::transform([](const auto &p) { return p.second; }), inputs,
                                 kw::compact_mode = true);

    // Create a propagator with derivatives.
    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    const auto tm = std::array{1440., 0.};
    const prop_t::in_1d<double> tm_in{tm.data(), 2};

    prop_t prop{md_input_t{ins.data(), 2}, kw::diff_order = 2, kw::compact_mode = true};

    REQUIRE(prop.get_nouts() == 252u);
    REQUIRE(prop.get_diff_order() == 2u);
    auto sl = prop.get_dslice(1);
    REQUIRE(sl.first == 7u);
    REQUIRE(sl.second == 7u + 7u * 7u);
    sl = prop.get_dslice(3, 1);
    REQUIRE(sl.first == 7u + 3u * 7u);
    REQUIRE(sl.second == 7u + 4u * 7u);
    REQUIRE(prop.get_mindex(7u + 4u * 7u) == dtens::sv_idx_t{4, {{0, 1}}});
    REQUIRE_THROWS_MATCHES(prop.get_mindex(1000u), std::invalid_argument,
                           Message("Cannot fetch the multiindex of the derivative at index 1000: the index "
                                   "is not less than the total number of derivatives (252)"));
    REQUIRE(prop.get_diff_args() == std::vector(inputs.begin(), inputs.begin() + 7));

    // Prepare the input buffer for the cfunc.
    std::vector<double> cf_in(ins.begin(), ins.begin() + 14);
    cf_in.insert(cf_in.end(), tm.begin(), tm.end());
    cfunc<double>::in_2d cf_in_span(cf_in.data(), 8, 2);

    // Prepare the output buffers.
    std::vector<double> cf_out(dt.size() * 2u), prop_out(cf_out);
    cfunc<double>::out_2d cf_out_span(cf_out.data(), dt.size(), 2);
    cfunc<double>::out_2d prop_out_span(prop_out.data(), dt.size(), 2);

    // Evaluate the cfunc.
    diff_cf(cf_out_span, cf_in_span);

    // Evaluate the propagation.
    prop(prop_out_span, tm_in);

    for (std::size_t i = 0; i < prop_out_span.extent(0); ++i) {
        for (std::size_t j = 0; j < prop_out_span.extent(1); ++j) {
            REQUIRE(prop_out_span(i, j) == approximately(cf_out_span(i, j), 10000.));
        }
    }
}

// A test with several satellites to test parallelisation.
TEST_CASE("large")
{
    const auto tot_N = 1000;
    const auto n_evals = 100;

    detail::edb_disabler ed;

    using md_input_t = mdspan<double, extents<std::size_t, 9, std::dynamic_extent>>;
    using prop_t = model::sgp4_propagator<double>;

    for (auto cm : {false, true}) {
        std::vector<double> ins;
        ins.resize(tot_N * 9);
        md_input_t in(ins.data(), tot_N);
        for (auto i = 0; i < tot_N; ++i) {
            in(0, i) = revday2radmin(13.75091047972192);
            in(1, i) = 0.0024963;
            in(2, i) = deg2rad(90.2039);
            in(3, i) = deg2rad(55.5633);
            in(4, i) = deg2rad(320.5956);
            in(5, i) = deg2rad(91.4738);
            in(6, i) = 0.75863e-3;
            in(7, i) = 2460486.5;
            in(8, i) = 0.6478633000000116;
        }

        prop_t prop{mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>{ins.data(), tot_N},
                    kw::compact_mode = cm};

        std::vector<prop_t::date> times;
        times.resize(tot_N, prop_t::date{2460486.5, 0.6478633000000116});

        std::vector<double> outs;
        outs.resize(tot_N * 7);
        prop_t::out_2d out_span{outs.data(), 7, tot_N};

        prop(out_span, prop_t::in_1d<prop_t::date>{times.data(), tot_N});

        for (auto i = 0; i < tot_N; ++i) {
            REQUIRE(out_span(0, i) == approximately(2561.223660636298, 10000.));
            REQUIRE(out_span(1, i) == approximately(3698.797144057697, 10000.));
            REQUIRE(out_span(2, i) == approximately(5818.772215708888, 10000.));
            REQUIRE(out_span(3, i) == approximately(-3.276142513618007, 10000.));
            REQUIRE(out_span(4, i) == approximately(-4.806489082829041, 10000.));
            REQUIRE(out_span(5, i) == approximately(4.511134501638151, 10000.));
            REQUIRE(out_span(6, i) == 0.);
        }

        times.resize(tot_N * n_evals, prop_t::date{2460486.5, 0.6478633000000116});
        outs.resize(tot_N * 7 * n_evals);
        prop_t::out_3d out_span_batch{outs.data(), n_evals, 7, tot_N};

        prop(out_span_batch, prop_t::in_2d<prop_t::date>{times.data(), n_evals, tot_N});

        for (auto i = 0; i < tot_N; ++i) {
            for (auto k = 0; k < n_evals; ++k) {
                REQUIRE(out_span_batch(k, 0, i) == approximately(2561.223660636298, 10000.));
                REQUIRE(out_span_batch(k, 1, i) == approximately(3698.797144057697, 10000.));
                REQUIRE(out_span_batch(k, 2, i) == approximately(5818.772215708888, 10000.));
                REQUIRE(out_span_batch(k, 3, i) == approximately(-3.276142513618007, 10000.));
                REQUIRE(out_span_batch(k, 4, i) == approximately(-4.806489082829041, 10000.));
                REQUIRE(out_span_batch(k, 5, i) == approximately(4.511134501638151, 10000.));
                REQUIRE(out_span_batch(k, 6, i) == 0.);
            }
        }
    }
}

TEST_CASE("s11n")
{
    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    prop_t prop{md_input_t{ins.data(), 2}};

    const std::vector sat_data(prop.get_sat_data().data_handle(),
                               prop.get_sat_data().data_handle() + prop.get_sat_data().size());

    std::stringstream ss;

    {
        boost::archive::binary_oarchive oa(ss);
        oa << prop;
    }

    prop = prop_t{};

    {
        boost::archive::binary_iarchive ia(ss);
        ia >> prop;
    }

    const std::vector new_sat_data(prop.get_sat_data().data_handle(),
                                   prop.get_sat_data().data_handle() + prop.get_sat_data().size());

    REQUIRE(prop.get_nsats() == 2u);
    REQUIRE(sat_data == new_sat_data);
}

TEST_CASE("replace_sat_data")
{
    using Catch::Matchers::Message;

    detail::edb_disabler ed;

    using prop_t = model::sgp4_propagator<double>;

    using md_input_t = mdspan<const double, extents<std::size_t, 9, std::dynamic_extent>>;

    const std::vector<double> ins = {revday2radmin(13.75091047972192),
                                     revday2radmin(15.50103472202482),
                                     0.0024963,
                                     0.0007417,
                                     deg2rad(90.2039),
                                     deg2rad(51.6439),
                                     deg2rad(55.5633),
                                     deg2rad(211.2001),
                                     deg2rad(320.5956),
                                     deg2rad(17.6667),
                                     deg2rad(91.4738),
                                     deg2rad(85.6398),
                                     0.75863e-3,
                                     .38792e-4,
                                     2460486.5,
                                     2458826.5,
                                     0.6478633000000116,
                                     0.6933954099999937};

    prop_t prop{md_input_t{ins.data(), 2}, kw::diff_order = 1};

    // Build a second propagator with the two satellites' data swapped.
    std::vector<double> ins2 = {revday2radmin(15.50103472202482),
                                revday2radmin(13.75091047972192),
                                0.0007417,
                                0.0024963,
                                deg2rad(51.6439),
                                deg2rad(90.2039),
                                deg2rad(211.2001),
                                deg2rad(55.5633),
                                deg2rad(17.6667),
                                deg2rad(320.5956),
                                deg2rad(85.6398),
                                deg2rad(91.4738),
                                .38792e-4,
                                0.75863e-3,
                                2458826.5,
                                2460486.5,
                                0.6933954099999937,
                                0.6478633000000116};

    prop_t prop2{md_input_t{ins2.data(), 2}, kw::diff_order = 1};

    // Replace the data in prop with the data in prop2.
    prop.replace_sat_data(md_input_t{ins2.data(), 2});

    // Check that the replacement worked.
    REQUIRE(std::vector(prop.get_sat_data().data_handle(), prop.get_sat_data().data_handle() + 18)
            == std::vector(prop2.get_sat_data().data_handle(), prop2.get_sat_data().data_handle() + 18));

    // Run evaluations in prop and prop2 and compare the results.
    std::vector<prop_t::date> times;
    times.resize(2u, prop_t::date{2460486.5, 0.6478633000000116});

    std::vector<double> outs;
    outs.resize(2 * prop.get_nouts());
    prop_t::out_2d out_span{outs.data(), prop.get_nouts(), 2};

    prop(out_span, prop_t::in_1d<prop_t::date>{times.data(), 2});

    const auto orig_out = outs;

    prop2(out_span, prop_t::in_1d<prop_t::date>{times.data(), 2});

    REQUIRE(orig_out == outs);

    // Also check with s11n.
    {
        std::stringstream ss;

        {
            boost::archive::binary_oarchive oa(ss);
            oa << prop;
        }

        prop = prop_t{};

        {
            boost::archive::binary_iarchive ia(ss);
            ia >> prop;
        }

        prop(out_span, prop_t::in_1d<prop_t::date>{times.data(), 2});

        REQUIRE(orig_out == outs);
    }

    REQUIRE(orig_out == outs);

    // Error throwing.
    REQUIRE_THROWS_MATCHES((prop.replace_sat_data(md_input_t{ins2.data(), 1})), std::invalid_argument,
                           Message("Invalid array provided to replace_sat_data(): the number of "
                                   "columns (1) does not match the number of satellites (2)"));
}

TEST_CASE("deep space detection")
{
    REQUIRE(model::gpe_is_deep_space(revday2radmin(6.), 0.0024963, deg2rad(90.2039)));
    REQUIRE(!model::gpe_is_deep_space(revday2radmin(13.75091047972192), 0.0024963, deg2rad(90.2039)));
}

TEST_CASE("jd_conversions")
{
    // Normal day: 2016-12-30T09:00:00.
    {
        // From UTC to TAI.
        const auto tai_jd = model::jd_utc_to_tai(2457752.875, 0.);
        const auto [tai_hi, tai_lo] = detail::eft_add_knuth(tai_jd.first, tai_jd.second);
        // NOTE: these are computed with astropy.
        const auto [tai_hi_cmp, tai_lo_cmp] = detail::eft_add_knuth(2457753.0, -0.12458333333333332);

        REQUIRE(tai_hi == tai_hi_cmp);
        REQUIRE(tai_lo == tai_lo_cmp);

        // From TAI to UTC.
        const auto utc_jd = model::jd_tai_to_utc(tai_hi, tai_lo);
        const auto [utc_hi, utc_lo] = detail::eft_add_knuth(utc_jd.first, utc_jd.second);
        const auto [utc_hi_cmp, utc_lo_cmp] = detail::eft_add_knuth(2457753.0, -0.125);

        REQUIRE(utc_hi == utc_hi_cmp);
        REQUIRE(utc_lo == utc_lo_cmp);
    }

    // Day with a leap second: 2016-12-31T09:00:00.
    {
        // From UTC to TAI.
        const auto tai_jd = model::jd_utc_to_tai(2457754.0, -0.12500434022754364);
        const auto [tai_hi, tai_lo] = detail::eft_add_knuth(tai_jd.first, tai_jd.second);
        // NOTE: these are computed with astropy.
        const auto [tai_hi_cmp, tai_lo_cmp] = detail::eft_add_knuth(2457754.0, -0.12458333333333332);

        REQUIRE(tai_hi == tai_hi_cmp);
        REQUIRE(tai_lo == tai_lo_cmp);

        // From TAI to UTC.
        const auto utc_jd = model::jd_tai_to_utc(tai_hi, tai_lo);
        const auto [utc_hi, utc_lo] = detail::eft_add_knuth(utc_jd.first, utc_jd.second);
        const auto [utc_hi_cmp, utc_lo_cmp] = detail::eft_add_knuth(2457754.0, -0.12500434022754364);

        REQUIRE(utc_hi == utc_hi_cmp);
        REQUIRE(utc_lo == utc_lo_cmp);
    }
}
