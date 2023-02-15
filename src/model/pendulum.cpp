// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/cos.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/model/pendulum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

namespace
{

void pendulum_check_params(const expression &gconst, const expression &L)
{
    if (!std::holds_alternative<number>(gconst.value()) && !std::holds_alternative<param>(gconst.value())) {
        throw std::invalid_argument(
            "The gravitational acceleration of a pendulum model must be a number or a parameter");
    }

    if (!std::holds_alternative<number>(L.value()) && !std::holds_alternative<param>(L.value())) {
        throw std::invalid_argument("The length of a pendulum must be a number or a parameter");
    }
}

} // namespace

std::vector<std::pair<expression, expression>> pendulum_impl(const expression &gconst, const expression &L)
{
    pendulum_check_params(gconst, L);

    auto [x, v] = make_vars("x", "v");

    return {prime(x) = v, prime(v) = -gconst / L * sin(x)};
}

expression pendulum_energy_impl(const expression &gconst, const expression &L)
{
    pendulum_check_params(gconst, L);

    auto [x, v] = make_vars("x", "v");

    return 0.5_dbl * (L * L) * (v * v) + gconst * L * (1_dbl - cos(x));
}

} // namespace model::detail

HEYOKA_END_NAMESPACE
