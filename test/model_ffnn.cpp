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

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>
#include <heyoka/model/ffnn.hpp>

#include "catch.hpp"

using namespace heyoka;

TEST_CASE("impl")
{
    using Catch::Matchers::Message;

    // A linear layer, just because
    auto linear = [](expression ret) -> expression { return ret; };
    // We also define a few symbols
    auto [x, y, z] = make_vars("x", "y", "z");

    // First, we test malformed cases and their throws.
    // 1 - number of activations function is wrong
    REQUIRE_THROWS_AS(model::detail::ffnn_impl({x}, {1}, 2, {heyoka::tanh, heyoka::tanh, linear},
                                               {1_dbl, 2_dbl, 3_dbl, 4_dbl, 5_dbl, 6_dbl}),
                      std::invalid_argument);
    // 2 - number of inputs is zero
    REQUIRE_THROWS_AS(
        model::detail::ffnn_impl({}, {1}, 2, {heyoka::tanh, heyoka::tanh}, {1_dbl, 2_dbl, 3_dbl, 4_dbl, 5_dbl, 6_dbl}),
        std::invalid_argument);
    // 3 - number of outputs is zero
    REQUIRE_THROWS_AS(model::detail::ffnn_impl({x}, {1}, 0, {heyoka::tanh, heyoka::tanh}, {1_dbl, 2_dbl, 3_dbl, 4_dbl}),
                      std::invalid_argument);
    // 4 - One of the hidden layers has zero neurons
    REQUIRE_THROWS_AS(model::detail::ffnn_impl({x}, {1, 0}, 2, {heyoka::tanh, heyoka::tanh, linear}, {1_dbl, 2_dbl}),
                      std::invalid_argument);
    // 5 - Wrong number of weights/biases
    REQUIRE_THROWS_AS(
        model::detail::ffnn_impl({x}, {1}, 1, {heyoka::tanh, heyoka::tanh}, {1_dbl, 2_dbl, 3_dbl, 5_dbl, 6_dbl}),
        std::invalid_argument);
    // 6 - empty activations list
    REQUIRE_THROWS_MATCHES(model::detail::ffnn_impl({x}, {1}, 1, {}, {1_dbl, 2_dbl, 3_dbl, 5_dbl, 6_dbl}),
                           std::invalid_argument,
                           Message("Cannot create a FFNN with an empty list of activation functions"));
    // 7 - empty activation functions
    REQUIRE_THROWS_MATCHES(
        model::detail::ffnn_impl({x}, {}, 1, {std::function<expression(const expression &)>{}}, {1_dbl, 2_dbl}),
        std::invalid_argument, Message("The list of activation functions cannot contain empty functions"));

    // We now check some hand coded networks
    {
        auto my_net = model::detail::ffnn_impl({x}, {}, 1, {linear}, {1_dbl, 2_dbl});
        REQUIRE(my_net[0] == expression(2_dbl + x));
    }
    {
        auto my_net = model::detail::ffnn_impl({x}, {}, 1, {heyoka::tanh}, {1_dbl, 2_dbl});
        REQUIRE(my_net[0] == expression(heyoka::tanh(2_dbl + x)));
    }
    {
        auto my_net = model::detail::ffnn_impl({x}, {1}, 1, {heyoka::tanh, linear}, {1_dbl, 2_dbl, 3_dbl, 4_dbl});
        REQUIRE(my_net[0] == expression(4_dbl + (2_dbl * heyoka::tanh(3_dbl + x))));
    }
    {
        auto my_net = model::detail::ffnn_impl({x}, {1}, 1, {heyoka::tanh, heyoka::sin}, {1_dbl, 2_dbl, 3_dbl, 4_dbl});
        REQUIRE(my_net[0] == expression(heyoka::sin(4_dbl + (2_dbl * heyoka::tanh(3_dbl + x)))));
    }
    {
        auto my_net = model::detail::ffnn_impl({x, y}, {2}, 1, {heyoka::sin, heyoka::cos},
                                               {1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl});
        REQUIRE(
            my_net[0]
            == expression(heyoka::cos(sum({1_dbl, heyoka::sin(sum({1_dbl, x, y})), heyoka::sin(sum({1_dbl, x, y}))}))));
    }

    // Check overflow detection.
    REQUIRE_THROWS_AS(
        model::detail::ffnn_impl({x}, {}, std::numeric_limits<std::uint32_t>::max(), {linear}, {1_dbl, 2_dbl}),
        std::system_error);
}

TEST_CASE("igor_iface")
{
    auto [x, y, z] = make_vars("x", "y", "z");
    {
        auto igor_v = model::ffnn(kw::inputs = {x, y}, kw::nn_hidden = {2u}, kw::n_out = 1u,
                                  kw::activations = {heyoka::sin, heyoka::cos},
                                  kw::nn_wb = {1., 1., 1., 1., 1., 1., 1., 1., 1.});
        auto vanilla_v = model::detail::ffnn_impl({x, y}, {2u}, 1u, {heyoka::sin, heyoka::cos},
                                                  {1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl, 1_dbl});
        REQUIRE(igor_v == vanilla_v);
    }
    // We test the expected setting for the default weights+biases expressions to par[i].
    {
        auto igor_v = model::ffnn(kw::inputs = {x, y}, kw::nn_hidden = {2u}, kw::n_out = 1u,
                                  kw::activations = {heyoka::sin, heyoka::cos});
        auto vanilla_v
            = model::detail::ffnn_impl({x, y}, {2u}, 1u, {heyoka::sin, heyoka::cos},
                                       {par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8]});
        REQUIRE(igor_v == vanilla_v);
    }
}
